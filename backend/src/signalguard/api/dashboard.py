"""Read feeds for the dashboard: decisions, orders, positions, equity curve.

Every query is scoped to the calling user's own broker accounts — one user can
never read another's decisions or positions. All money is serialised as a string
(see `schemas`), never a float.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from signalguard.api.deps import CurrentUser, DbSession
from signalguard.api.schemas import (
    DecisionOut,
    EquityPointOut,
    OrderOut,
    PositionOut,
)
from signalguard.db.models import (
    BrokerAccount,
    Decision,
    EquitySnapshot,
    Order,
    Position,
)
from signalguard.enums import ReasonCode

router = APIRouter(prefix="/api", tags=["dashboard"])

_MAX_PAGE = 200


def _dec(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


async def _account_ids(session: DbSession, user_id: uuid.UUID) -> list[uuid.UUID]:
    result = await session.execute(
        select(BrokerAccount.id).where(BrokerAccount.user_id == user_id)
    )
    return list(result.scalars().all())


@router.get("/decisions", response_model=list[DecisionOut])
async def list_decisions(
    user: CurrentUser,
    session: DbSession,
    reason_code: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(gt=0, le=_MAX_PAGE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DecisionOut]:
    """Newest first, optionally filtered by reason code (the rejection breakdown)."""
    ids = await _account_ids(session, user.id)
    if not ids:
        return []

    stmt = select(Decision).where(Decision.broker_account_id.in_(ids))
    if reason_code is not None:
        if reason_code not in ReasonCode.__members__.values():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown reason_code: {reason_code}",
            )
        stmt = stmt.where(Decision.reason_code == reason_code)
    stmt = stmt.order_by(Decision.evaluated_at.desc()).limit(limit).offset(offset)

    result = await session.execute(stmt)
    return [_decision_out(d) for d in result.scalars().all()]


def _decision_out(d: Decision) -> DecisionOut:
    return DecisionOut(
        id=str(d.id),
        broker_account_id=str(d.broker_account_id) if d.broker_account_id else None,
        verdict=d.verdict,
        reason_code=d.reason_code,
        reason_detail=d.reason_detail,
        computed_qty=_dec(d.computed_qty),
        entry_reference_price=_dec(d.entry_reference_price),
        stop_price=_dec(d.stop_price),
        evaluated_at=d.evaluated_at,
        latency_ms=d.latency_ms,
        is_test=d.is_test,
    )


@router.get("/orders", response_model=list[OrderOut])
async def list_orders(
    user: CurrentUser,
    session: DbSession,
    limit: Annotated[int, Query(gt=0, le=_MAX_PAGE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[OrderOut]:
    ids = await _account_ids(session, user.id)
    if not ids:
        return []
    result = await session.execute(
        select(Order)
        .where(Order.broker_account_id.in_(ids))
        .order_by(Order.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [_order_out(o) for o in result.scalars().all()]


def _order_out(o: Order) -> OrderOut:
    return OrderOut(
        id=str(o.id),
        broker_account_id=str(o.broker_account_id),
        symbol=o.symbol,
        side=o.side,
        type=o.type,
        role=o.role,
        status=o.status,
        qty=str(o.qty),
        price=_dec(o.price),
        stop_price=_dec(o.stop_price),
        filled_qty=str(o.filled_qty),
        avg_fill_price=_dec(o.avg_fill_price),
        fees=str(o.fees),
        submitted_at=o.submitted_at,
        filled_at=o.filled_at,
        updated_at=o.updated_at,
    )


@router.get("/positions", response_model=list[PositionOut])
async def list_positions(user: CurrentUser, session: DbSession) -> list[PositionOut]:
    """Open positions across the user's accounts. A cache of broker truth (§10)."""
    ids = await _account_ids(session, user.id)
    if not ids:
        return []
    result = await session.execute(
        select(Position)
        .where(Position.broker_account_id.in_(ids), Position.qty != 0)
        .order_by(Position.symbol)
    )
    return [
        PositionOut(
            id=str(p.id),
            broker_account_id=str(p.broker_account_id),
            symbol=p.symbol,
            qty=str(p.qty),
            avg_entry=str(p.avg_entry),
            mark_price=_dec(p.mark_price),
            unrealized_pnl=_dec(p.unrealized_pnl),
            updated_at=p.updated_at,
        )
        for p in result.scalars().all()
    ]


@router.get("/equity", response_model=list[EquityPointOut])
async def equity_curve(
    user: CurrentUser,
    session: DbSession,
    account_id: Annotated[str, Query()],
    since: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(gt=0, le=1000)] = _MAX_PAGE,
) -> list[EquityPointOut]:
    """Equity snapshots for one account, oldest first — the curve the chart draws."""
    account_uuid = await _require_owned_account(session, user.id, account_id)

    stmt = select(EquitySnapshot).where(
        EquitySnapshot.broker_account_id == account_uuid
    )
    if since is not None:
        stmt = stmt.where(EquitySnapshot.taken_at >= since)
    stmt = stmt.order_by(EquitySnapshot.taken_at).limit(limit)

    result = await session.execute(stmt)
    return [
        EquityPointOut(
            equity=str(s.equity),
            free_balance=_dec(s.free_balance),
            taken_at=s.taken_at,
            is_session_baseline=s.is_session_baseline,
        )
        for s in result.scalars().all()
    ]


async def _require_owned_account(
    session: DbSession, user_id: uuid.UUID, account_id: str
) -> uuid.UUID:
    try:
        account_uuid = uuid.UUID(account_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        ) from exc
    result = await session.execute(
        select(BrokerAccount.id).where(
            BrokerAccount.id == account_uuid, BrokerAccount.user_id == user_id
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="account not found"
        )
    return account_uuid
