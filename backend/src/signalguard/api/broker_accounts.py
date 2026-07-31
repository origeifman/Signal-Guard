"""Broker-account CRUD (CLAUDE.md §11).

Credentials are encrypted on the way in and never returned on the way out — not
the key, not the secret, not the ciphertext. Delete is a soft delete
(`is_active = false`): `decisions` and `orders` reference an account with
ON DELETE RESTRICT and are append-only, so a hard delete would either fail or
destroy an audit trail. A deactivated account stops routing new signals while its
history stays intact.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from signalguard.api.deps import CurrentUser, DbSession
from signalguard.api.schemas import BrokerAccountCreate, BrokerAccountOut
from signalguard.config import get_settings
from signalguard.db.models import BrokerAccount
from signalguard.execution.credentials import BrokerCredentials, seal_credentials

router = APIRouter(prefix="/api/broker-accounts", tags=["broker-accounts"])


def _account_out(a: BrokerAccount) -> BrokerAccountOut:
    return BrokerAccountOut(
        id=str(a.id),
        broker=a.broker,
        label=a.label,
        is_testnet=a.is_testnet,
        is_active=a.is_active,
        trading_state=a.trading_state,
        locked_at=a.locked_at,
        locked_reason=a.locked_reason,
        created_at=a.created_at,
    )


@router.get("", response_model=list[BrokerAccountOut])
async def list_accounts(
    user: CurrentUser, session: DbSession
) -> list[BrokerAccountOut]:
    result = await session.execute(
        select(BrokerAccount)
        .where(BrokerAccount.user_id == user.id)
        .order_by(BrokerAccount.created_at)
    )
    return [_account_out(a) for a in result.scalars().all()]


@router.post("", response_model=BrokerAccountOut, status_code=status.HTTP_201_CREATED)
async def create_account(
    body: BrokerAccountCreate, user: CurrentUser, session: DbSession
) -> BrokerAccountOut:
    settings = get_settings()
    ciphertext, nonce = seal_credentials(
        BrokerCredentials(api_key=body.api_key, api_secret=body.api_secret),
        settings.credentials_master_key,
    )
    account = BrokerAccount(
        user_id=user.id,
        broker=body.broker,
        label=body.label,
        encrypted_credentials=ciphertext,
        credentials_nonce=nonce,
        is_testnet=True,  # constraint #2 — the DB CHECK also enforces this
    )
    session.add(account)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        # Unique (user_id, label): two accounts with the same label would make
        # routing a signal by label a coin flip.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"an account labelled {body.label!r} already exists",
        ) from exc
    await session.commit()
    await session.refresh(account)
    return _account_out(account)


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    account_id: str, user: CurrentUser, session: DbSession
) -> None:
    """Deactivate an account (soft delete). Its decisions and orders remain."""
    try:
        account_uuid = uuid.UUID(account_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        ) from exc

    result = await session.execute(
        select(BrokerAccount).where(
            BrokerAccount.id == account_uuid, BrokerAccount.user_id == user.id
        )
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        )
    account.is_active = False
    await session.commit()
