"""Risk-profile read, update, and the live sizing preview (CLAUDE.md §7, §11)."""

from __future__ import annotations

import uuid
from datetime import time
from decimal import Decimal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from signalguard.api.deps import CurrentUser, DbSession
from signalguard.api.schemas import (
    RiskProfileOut,
    RiskProfileUpdate,
    SizingPreviewOut,
    SizingPreviewRequest,
)
from signalguard.db.models import BrokerAccount, EquitySnapshot, Instrument, RiskProfile
from signalguard.ingress.pipeline import profile_to_config
from signalguard.risk.sizing import compute_position_size
from signalguard.risk.types import InstrumentSpec

router = APIRouter(prefix="/api/risk-profile", tags=["risk-profile"])


def _profile_out(p: RiskProfile) -> RiskProfileOut:
    return RiskProfileOut(
        version=p.version,
        max_alert_age_sec=p.max_alert_age_sec,
        future_tolerance_sec=p.future_tolerance_sec,
        dedupe_window_sec=p.dedupe_window_sec,
        allowed_symbols=list(p.allowed_symbols or []),
        min_stop_distance_pct=str(p.min_stop_distance_pct),
        consecutive_loss_threshold=p.consecutive_loss_threshold,
        circuit_breaker_cooldown_minutes=p.circuit_breaker_cooldown_minutes,
        circuit_breaker_manual_reset=p.circuit_breaker_manual_reset,
        max_daily_dd_pct=str(p.max_daily_dd_pct),
        risk_per_trade_pct=str(p.risk_per_trade_pct),
        fee_slippage_buffer_bps=p.fee_slippage_buffer_bps,
        max_notional_per_trade=str(p.max_notional_per_trade),
        max_open_positions=p.max_open_positions,
        max_total_notional=str(p.max_total_notional),
        allow_pyramiding=p.allow_pyramiding,
        daily_reset_time=p.daily_reset_time.isoformat(),
        timezone=p.timezone,
        updated_at=p.updated_at,
    )


async def _load_profile(session: DbSession, user_id: uuid.UUID) -> RiskProfile:
    result = await session.execute(
        select(RiskProfile).where(RiskProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        # Every account is created with a profile; its absence is a real fault,
        # not a normal 404, and we refuse to guess defaults into the risk engine.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="risk profile not found"
        )
    return profile


@router.get("", response_model=RiskProfileOut)
async def get_profile(user: CurrentUser, session: DbSession) -> RiskProfileOut:
    return _profile_out(await _load_profile(session, user.id))


@router.put("", response_model=RiskProfileOut)
async def update_profile(
    body: RiskProfileUpdate, user: CurrentUser, session: DbSession
) -> RiskProfileOut:
    """Apply the provided fields and bump the version.

    The version bump is what keeps a profile edit from rewriting history: it is
    copied into every future decision's snapshot, so past decisions keep the
    rules that actually produced them (plan §3, Q4). Tightening a limit takes
    effect from the next decision, never retroactively to an open position.
    """
    profile = await _load_profile(session, user.id)

    fields = body.model_dump(exclude_unset=True)

    # These two need parsing/validation beyond what pydantic did on the wire type.
    if "daily_reset_time" in fields:
        fields["daily_reset_time"] = _parse_reset_time(fields["daily_reset_time"])
    if "timezone" in fields:
        _validate_timezone(fields["timezone"])

    if not fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="no fields to update"
        )

    for name, value in fields.items():
        setattr(profile, name, value)
    profile.version += 1

    await session.commit()
    await session.refresh(profile)
    return _profile_out(profile)


def _parse_reset_time(raw: str) -> time:
    try:
        return time.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="daily_reset_time must be HH:MM or HH:MM:SS",
        ) from exc


def _validate_timezone(name: str) -> None:
    # An invalid timezone would silently break the DST-correct daily reset
    # (constraint #7), so reject it at the edge rather than storing it.
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown timezone: {name}",
        ) from exc


@router.post("/preview", response_model=SizingPreviewOut)
async def preview_size(
    body: SizingPreviewRequest, user: CurrentUser, session: DbSession
) -> SizingPreviewOut:
    """Show the exact quantity the engine would size, for a hypothetical trade.

    This runs the real sizing transform — not an approximation — so the number on
    the form is the number that would actually be sent. It reads the account's
    cached exchange filters but does **not** apply the staleness cutoff the live
    pipeline does: a preview is a what-if, not an order.
    """
    account = await _load_owned_account(session, user.id, body.account_id)
    profile = await _load_profile(session, user.id)

    instrument = await _load_instrument(session, account.broker, body.symbol)
    equity, free_balance = await _resolve_preview_balances(session, account.id, body)

    result = compute_position_size(
        entry_price=body.entry_price,
        stop_price=body.stop_price,
        total_equity=equity,
        free_balance=free_balance,
        config=profile_to_config(profile),
        instrument=instrument,
    )
    return SizingPreviewOut(
        tradeable=result.is_tradeable,
        qty=str(result.qty),
        notional=str(result.notional),
        risk_amount=str(result.risk_amount),
        stop_distance=str(result.stop_distance),
        reason=result.detail,
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


async def _load_instrument(
    session: DbSession, broker: str, symbol: str
) -> InstrumentSpec:
    result = await session.execute(
        select(Instrument).where(
            Instrument.broker == broker, Instrument.symbol == symbol
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"no cached exchange filters for {symbol}. Connect the account so "
                "SignalGuard can fetch its lot size and minimums."
            ),
        )
    return InstrumentSpec(
        symbol=row.symbol,
        tick_size=row.tick_size,
        lot_step=row.lot_step,
        min_qty=row.min_qty,
        min_notional=row.min_notional,
    )


async def _resolve_preview_balances(
    session: DbSession, account_id: uuid.UUID, body: SizingPreviewRequest
) -> tuple[Decimal, Decimal]:
    """Pick the equity/free-balance to size against: override, else last snapshot.

    `free_balance` defaults to `equity` (treat all equity as spendable) for a
    hypothetical, so the preview isn't capped by a stale cash figure the user is
    only exploring around.
    """
    equity = body.equity
    free_balance = body.free_balance

    if equity is None or free_balance is None:
        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.broker_account_id == account_id)
            .order_by(EquitySnapshot.taken_at.desc())
            .limit(1)
        )
        snapshot = result.scalar_one_or_none()
        if equity is None:
            if snapshot is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "no equity snapshot for this account yet — pass an equity "
                        "value to preview a hypothetical size."
                    ),
                )
            equity = snapshot.equity
        if free_balance is None:
            free_balance = (
                snapshot.free_balance
                if snapshot is not None and snapshot.free_balance is not None
                else equity
            )

    return equity, free_balance
