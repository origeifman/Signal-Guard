"""The kill switch, exposed over HTTP (CLAUDE.md §10).

Locking the account is the guarantee that must not fail: even if the broker
adapter cannot be built or the sweep errors, the account is still marked LOCKED so
no new signal is accepted, and the reconciler keeps trying to flatten it. So this
endpoint locks first and reports sweep failures rather than aborting on them.

Idempotent: firing it on an already-locked account re-runs the sweep.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy import select

from signalguard.api.deps import CurrentUser, DbSession
from signalguard.api.schemas import KillSwitchOut, KillSwitchRequest
from signalguard.config import get_settings
from signalguard.db.models import BrokerAccount
from signalguard.execution.factory import build_adapter
from signalguard.execution.killswitch import fire_kill_switch, set_locked
from signalguard.redis_client import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kill-switch", tags=["kill-switch"])


@router.post("", response_model=KillSwitchOut)
async def fire(
    body: KillSwitchRequest, user: CurrentUser, session: DbSession
) -> KillSwitchOut:
    account = await _load_owned_account(session, user.id, body.account_id)

    redis: Redis | None
    try:
        redis = get_redis()
    except RuntimeError:
        # Redis down is not a reason to refuse the kill switch — the durable lock
        # lives in Postgres. Proceed without the cache.
        redis = None

    reason = f"manual kill switch by user {user.id}"

    try:
        adapter = build_adapter(account, get_settings().credentials_master_key)
    except Exception:  # noqa: BLE001 - includes bad credentials / unknown broker
        # Cannot reach the broker to flatten — but we can still LOCK, which stops
        # anything new. The reconciler enforces flatness once the broker returns.
        logger.critical(
            "Kill switch: could not build broker adapter — locking only",
            extra={"broker_account_id": str(account.id)},
        )
        await set_locked(session, redis, account.id, reason)
        await session.commit()
        return KillSwitchOut(
            account_id=str(account.id),
            locked=True,
            orders_cancelled=0,
            positions_closed=0,
            clean=False,
            errors=["broker_adapter_unavailable"],
        )

    try:
        result = await fire_kill_switch(session, redis, adapter, account.id, reason)
        await session.commit()
    finally:
        await adapter.aclose()

    return KillSwitchOut(
        account_id=str(result.account_id),
        locked=result.locked,
        orders_cancelled=result.orders_cancelled,
        positions_closed=result.positions_closed,
        clean=result.is_clean,
        errors=result.errors,
    )


async def _load_owned_account(
    session: DbSession, user_id: uuid.UUID, account_id: str
) -> BrokerAccount:
    try:
        account_uuid = uuid.UUID(account_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        ) from exc
    result = await session.execute(
        select(BrokerAccount).where(
            BrokerAccount.id == account_uuid, BrokerAccount.user_id == user_id
        )
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        )
    return account
