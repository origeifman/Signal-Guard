"""Pure tests for the API request models and the credential envelope.

No database, no network — these run in the standard suite. They pin the
validation that keeps bad input out of the risk engine and the round-trip that
keeps broker secrets encrypted at rest.
"""

from __future__ import annotations

import base64
import secrets
from decimal import Decimal

import pytest
from pydantic import ValidationError

from signalguard.api.schemas import (
    BrokerAccountCreate,
    KillSwitchRequest,
    RegisterRequest,
    RiskProfileUpdate,
)
from signalguard.execution.credentials import (
    BrokerCredentials,
    open_credentials,
    seal_credentials,
)

MASTER_KEY = base64.b64encode(secrets.token_bytes(32)).decode()


# --- email normalisation ------------------------------------------------------


def test_register_lowercases_and_trims_email() -> None:
    req = RegisterRequest(email="  Alice@Example.COM ", password="a" * 12)
    assert req.email == "alice@example.com"


@pytest.mark.parametrize("bad", ["", "no-at-sign", "@nolocal.com", "a@nodot"])
def test_register_rejects_malformed_email(bad: str) -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email=bad, password="a" * 12)


def test_register_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="short")


# --- risk-profile bounds mirror the DB CHECK constraints ----------------------


@pytest.mark.parametrize(
    "field",
    ["risk_per_trade_pct", "max_daily_dd_pct", "min_stop_distance_pct"],
)
def test_percentages_must_be_a_fraction(field: str) -> None:
    # The classic mistake: typing "50" meaning 50%. A fraction column would then
    # size positions 5,000x too large — rejected at the edge.
    with pytest.raises(ValidationError):
        RiskProfileUpdate(**{field: Decimal("50")})
    with pytest.raises(ValidationError):
        RiskProfileUpdate(**{field: Decimal("0")})


def test_risk_profile_update_normalises_symbols() -> None:
    upd = RiskProfileUpdate(allowed_symbols=[" btcusdt ", "ethusdt", "  "])
    assert upd.allowed_symbols == ["BTCUSDT", "ETHUSDT"]


def test_risk_profile_update_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RiskProfileUpdate(risk_per_trade_ptc=Decimal("0.01"))  # typo'd key


def test_empty_update_is_allowed_by_the_model() -> None:
    # The "nothing to change" guard lives in the endpoint (400); the model itself
    # permits an all-None body.
    assert RiskProfileUpdate().model_dump(exclude_unset=True) == {}


# --- testnet-only and kill-switch confirmation --------------------------------


def test_broker_account_refuses_non_testnet() -> None:
    with pytest.raises(ValidationError):
        BrokerAccountCreate(
            broker="binance_spot_testnet",
            label="x",
            api_key="k",
            api_secret="s",
            is_testnet=False,
        )


def test_kill_switch_requires_exact_confirmation() -> None:
    ok = KillSwitchRequest(account_id="a", confirm="confirm")
    assert ok.confirm == "confirm"
    with pytest.raises(ValidationError):
        KillSwitchRequest(account_id="a", confirm="yes")


# --- credential envelope round-trip ------------------------------------------


def test_credentials_round_trip() -> None:
    creds = BrokerCredentials(api_key="my-key", api_secret="my-secret")
    ciphertext, nonce = seal_credentials(creds, MASTER_KEY)

    # The plaintext secret must not be findable in what gets stored.
    assert b"my-secret" not in ciphertext
    assert b"my-key" not in ciphertext

    recovered = open_credentials(ciphertext, nonce, MASTER_KEY)
    assert recovered == creds


def test_credentials_tamper_is_rejected() -> None:
    creds = BrokerCredentials(api_key="k", api_secret="s")
    ciphertext, nonce = seal_credentials(creds, MASTER_KEY)
    tampered = bytes([ciphertext[0] ^ 0x01]) + ciphertext[1:]
    with pytest.raises(Exception):  # noqa: B017 - AES-GCM raises InvalidTag
        open_credentials(tampered, nonce, MASTER_KEY)
