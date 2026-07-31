"""End-to-end dashboard API tests against a live Postgres and Redis.

Marked `integration` (skipped when DATABASE_URL is unset), like the webhook e2e
suite: session auth, ownership scoping, and the append-only decision feed are
guarantees produced by the database and the cookie jar together, and faking
either would test the fake. No test touches a real network — the kill switch runs
against the in-memory FakeBroker.

Run with the stack up:
    docker compose exec api uv run pytest tests/api -q
"""

from __future__ import annotations

import base64
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL", "")
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

MASTER_KEY = base64.b64encode(secrets.token_bytes(32)).decode()
PEPPER = secrets.token_urlsafe(32)


@pytest.fixture()
async def app(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("CREDENTIALS_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("ENDPOINT_ID_PEPPER", PEPPER)
    monkeypatch.setenv("SESSION_SECRET", secrets.token_urlsafe(32))
    monkeypatch.setenv("APP_ENV", "test")

    from signalguard.config import get_settings
    from signalguard.db.session import dispose_engine, init_engine
    from signalguard.redis_client import close_redis, init_redis

    get_settings.cache_clear()
    init_engine(DATABASE_URL)
    init_redis(REDIS_URL)

    from signalguard.main import create_app

    yield create_app()

    await dispose_engine()
    await close_redis()
    get_settings.cache_clear()


def _client(app: Any) -> AsyncClient:
    """A fresh client (its own cookie jar) — one per simulated user."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _email() -> str:
    return f"{uuid.uuid4()}@example.test"


async def _register(client: AsyncClient) -> str:
    email = _email()
    resp = await client.post(
        "/api/auth/register", json={"email": email, "password": "correct horse staple"}
    )
    assert resp.status_code == 201, resp.text
    return email


# --- auth ---------------------------------------------------------------------


async def test_register_login_logout_lifecycle(app: Any) -> None:
    async with _client(app) as client:
        email = await _register(client)

        me = await client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["email"] == email

        assert (await client.post("/api/auth/logout")).status_code == 204
        # Cookie cleared and session revoked — /me now fails closed.
        assert (await client.get("/api/auth/me")).status_code == 401


async def test_me_requires_a_session(app: Any) -> None:
    async with _client(app) as client:
        assert (await client.get("/api/auth/me")).status_code == 401


async def test_login_wrong_password_is_401(app: Any) -> None:
    async with _client(app) as client:
        email = await _register(client)
        await client.post("/api/auth/logout")
        resp = await client.post(
            "/api/auth/login", json={"email": email, "password": "wrong wrong wrong"}
        )
        assert resp.status_code == 401


async def test_duplicate_email_is_rejected(app: Any) -> None:
    async with _client(app) as a, _client(app) as b:
        email = _email()
        body = {"email": email, "password": "correct horse staple"}
        assert (await a.post("/api/auth/register", json=body)).status_code == 201
        assert (await b.post("/api/auth/register", json=body)).status_code == 409


# --- risk profile -------------------------------------------------------------


async def test_default_profile_denies_all_symbols(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        resp = await client.get("/api/risk-profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["allowed_symbols"] == []
        assert body["version"] == 1


async def test_profile_update_bumps_version_and_normalises(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        resp = await client.put(
            "/api/risk-profile",
            json={"allowed_symbols": ["btcusdt"], "risk_per_trade_pct": "0.02"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["allowed_symbols"] == ["BTCUSDT"]
        assert body["risk_per_trade_pct"] == "0.02"
        assert body["version"] == 2


async def test_profile_rejects_out_of_range_percentage(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        # 50 meaning "50%" in a fraction column — must be refused, not stored.
        resp = await client.put("/api/risk-profile", json={"risk_per_trade_pct": "50"})
        assert resp.status_code == 422


async def test_profile_rejects_unknown_timezone(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        resp = await client.put("/api/risk-profile", json={"timezone": "Mars/Olympus"})
        assert resp.status_code == 422


# --- broker accounts ----------------------------------------------------------


async def _create_account(client: AsyncClient, label: str = "binance-testnet-1") -> str:
    resp = await client.post(
        "/api/broker-accounts",
        json={
            "broker": "binance_spot_testnet",
            "label": label,
            "api_key": "k-" + secrets.token_hex(4),
            "api_secret": "s-" + secrets.token_hex(4),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def test_broker_account_never_returns_credentials(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        await _create_account(client)
        resp = await client.get("/api/broker-accounts")
        assert resp.status_code == 200
        (account,) = resp.json()
        # No credential material in any form.
        for banned in ("api_key", "api_secret", "encrypted_credentials", "credentials"):
            assert banned not in account


async def test_duplicate_label_rejected(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        await _create_account(client, "dup")
        resp = await client.post(
            "/api/broker-accounts",
            json={
                "broker": "binance_spot_testnet",
                "label": "dup",
                "api_key": "k",
                "api_secret": "s",
            },
        )
        assert resp.status_code == 409


async def test_accounts_are_scoped_per_user(app: Any) -> None:
    async with _client(app) as alice, _client(app) as bob:
        await _register(alice)
        await _register(bob)
        alice_account = await _create_account(alice)

        # Bob sees none of Alice's accounts...
        assert (await bob.get("/api/broker-accounts")).json() == []
        # ...and cannot delete hers.
        assert (
            await bob.delete(f"/api/broker-accounts/{alice_account}")
        ).status_code == 404


async def test_delete_is_soft(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        account_id = await _create_account(client)
        assert (
            await client.delete(f"/api/broker-accounts/{account_id}")
        ).status_code == 204
        # Soft delete: it stops appearing in the active list.
        assert client and (await client.get("/api/broker-accounts")).json() == []


# --- dashboard feeds ----------------------------------------------------------


async def test_feeds_start_empty_and_scoped(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        for path in ("/api/decisions", "/api/orders", "/api/positions"):
            resp = await client.get(path)
            assert resp.status_code == 200
            assert resp.json() == []


async def test_decisions_reject_unknown_reason_code(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        resp = await client.get("/api/decisions", params={"reason_code": "NONSENSE"})
        assert resp.status_code == 422


# --- kill switch (FakeBroker — no network) ------------------------------------


async def test_kill_switch_locks_the_account(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakes.fake_broker import FakeBroker

    monkeypatch.setattr(
        "signalguard.api.killswitch.build_adapter",
        lambda account, key: FakeBroker(),
    )

    async with _client(app) as client:
        await _register(client)
        account_id = await _create_account(client)

        resp = await client.post(
            "/api/kill-switch", json={"account_id": account_id, "confirm": "CONFIRM"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["locked"] is True
        assert body["account_id"] == account_id


async def test_kill_switch_needs_confirmation(app: Any) -> None:
    async with _client(app) as client:
        await _register(client)
        account_id = await _create_account(client)
        resp = await client.post(
            "/api/kill-switch", json={"account_id": account_id, "confirm": "nope"}
        )
        assert resp.status_code == 422


# --- sizing preview -----------------------------------------------------------


async def test_sizing_preview_matches_the_engine(app: Any) -> None:
    """Preview returns the exact number the engine would size, with an equity override."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async with _client(app) as client:
        await _register(client)
        account_id = await _create_account(client)

        # Cache instrument filters the preview needs.
        engine = create_async_engine(DATABASE_URL)
        maker = async_sessionmaker(engine)
        async with maker() as s:
            await s.execute(
                text(
                    "INSERT INTO instruments (broker, symbol, base_asset, quote_asset,"
                    " tick_size, lot_step, min_qty, min_notional, status, fetched_at)"
                    " VALUES ('binance_spot_testnet','BTCUSDT','BTC','USDT',"
                    " 0.01, 0.00001, 0.00001, 10, 'TRADING', :now)"
                    " ON CONFLICT (broker, symbol) DO UPDATE SET fetched_at = :now"
                ),
                {"now": datetime.now(UTC)},
            )
            await s.commit()
        await engine.dispose()

        resp = await client.post(
            "/api/risk-profile/preview",
            json={
                "account_id": account_id,
                "symbol": "BTCUSDT",
                "entry_price": "62000",
                "stop_price": "61000",
                "equity": "10000",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Closed-form buffer (OQ-3): 100 / (1000 + 62000*0.002) = 0.08896..., round
        # down to the 0.00001 lot step.
        assert body["tradeable"] is True
        assert body["qty"] == "0.08896"
