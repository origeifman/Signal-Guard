"""Request and response models for the dashboard REST API.

**Money and quantities cross the wire as strings, never JSON numbers.** FastAPI's
default encoder turns a `Decimal` into a `float`, and a float is exactly what
constraint #3 forbids for money: `0.1 + 0.2` is not `0.3`, and a dashboard that
silently rounds a quantity in transit is misreporting what was traded. So every
money field on a response model is a `str`, filled with `str(value)` when the
response is built — the Decimal never touches the JSON encoder.

Request models mirror the database CHECK constraints (risk_profiles) so a bad
value fails fast with a clear 422 instead of surfacing as a raw IntegrityError.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- auth ---------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    # 12 is a floor, not a policy — long enough that Argon2id is doing real work.
    password: str = Field(min_length=12, max_length=1024)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        value = value.strip().lower()
        local, _, domain = value.partition("@")
        if not local or not domain or "." not in domain:
            raise ValueError("not a valid email address")
        return value


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime


# --- risk profile -------------------------------------------------------------


class RiskProfileOut(BaseModel):
    version: int
    max_alert_age_sec: int
    future_tolerance_sec: int
    dedupe_window_sec: int
    allowed_symbols: list[str]
    min_stop_distance_pct: str
    consecutive_loss_threshold: int
    circuit_breaker_cooldown_minutes: int
    circuit_breaker_manual_reset: bool
    max_daily_dd_pct: str
    risk_per_trade_pct: str
    fee_slippage_buffer_bps: int
    max_notional_per_trade: str
    max_open_positions: int
    max_total_notional: str
    allow_pyramiding: bool
    daily_reset_time: str
    timezone: str
    updated_at: datetime


class RiskProfileUpdate(BaseModel):
    """A partial update. Only the fields present are changed; the rest are left.

    Every provided field is validated here against the same bounds the database
    enforces, so the API rejects a nonsensical profile (e.g. `risk_per_trade_pct`
    of 50 meaning "50" when the column means a fraction) with a readable error
    rather than a 500 from a constraint violation. Percentages are fractions:
    0.01 is 1%.
    """

    model_config = ConfigDict(extra="forbid")

    max_alert_age_sec: int | None = Field(default=None, gt=0)
    future_tolerance_sec: int | None = Field(default=None, ge=0)
    dedupe_window_sec: int | None = Field(default=None, gt=0)
    allowed_symbols: list[str] | None = None
    min_stop_distance_pct: Decimal | None = Field(default=None, gt=0, lt=1)
    consecutive_loss_threshold: int | None = Field(default=None, gt=0)
    circuit_breaker_cooldown_minutes: int | None = Field(default=None, ge=0)
    circuit_breaker_manual_reset: bool | None = None
    max_daily_dd_pct: Decimal | None = Field(default=None, gt=0, lt=1)
    risk_per_trade_pct: Decimal | None = Field(default=None, gt=0, lt=1)
    fee_slippage_buffer_bps: int | None = Field(default=None, ge=0)
    max_notional_per_trade: Decimal | None = Field(default=None, gt=0)
    max_open_positions: int | None = Field(default=None, gt=0)
    max_total_notional: Decimal | None = Field(default=None, gt=0)
    allow_pyramiding: bool | None = None
    daily_reset_time: str | None = None
    timezone: str | None = None

    @field_validator("allowed_symbols")
    @classmethod
    def _upper_symbols(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        # Symbols are matched exactly against the payload; normalise so
        # "btcusdt" and "BTCUSDT" cannot disagree about what is allowed.
        return [s.strip().upper() for s in value if s.strip()]


class SizingPreviewRequest(BaseModel):
    """Inputs for the live sizing preview on the risk-profile page (CLAUDE.md §11).

    `equity`/`free_balance` are optional overrides so the user can explore a
    scenario ("what would this size at $10,000?") without a funded account. When
    omitted, the endpoint fills them from the account's latest equity snapshot.
    """

    account_id: str
    symbol: str
    entry_price: Decimal = Field(gt=0)
    stop_price: Decimal = Field(gt=0)
    equity: Decimal | None = Field(default=None, gt=0)
    free_balance: Decimal | None = Field(default=None, ge=0)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


class SizingPreviewOut(BaseModel):
    """The number the engine would actually use, plus why if it would be rejected."""

    tradeable: bool
    qty: str
    notional: str
    risk_amount: str
    stop_distance: str
    reason: str | None = None


# --- broker accounts ----------------------------------------------------------


class BrokerAccountCreate(BaseModel):
    broker: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=64)
    api_key: str = Field(min_length=1)
    api_secret: str = Field(min_length=1)
    # Present but must be true: constraint #2. The database also refuses a
    # non-testnet row, but rejecting here gives a readable error.
    is_testnet: bool = True

    @field_validator("is_testnet")
    @classmethod
    def _testnet_only(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "Live trading is not authorised in this phase — is_testnet must be true."
            )
        return value


class BrokerAccountOut(BaseModel):
    """Never carries credentials — not the key, not the secret, not the ciphertext."""

    id: str
    broker: str
    label: str
    is_testnet: bool
    is_active: bool
    trading_state: str
    locked_at: datetime | None
    locked_reason: str | None
    created_at: datetime


# --- decisions / orders / positions / equity ----------------------------------


class DecisionOut(BaseModel):
    id: str
    broker_account_id: str | None
    verdict: str
    reason_code: str
    reason_detail: str | None
    computed_qty: str | None
    entry_reference_price: str | None
    stop_price: str | None
    evaluated_at: datetime
    latency_ms: int
    is_test: bool


class OrderOut(BaseModel):
    id: str
    broker_account_id: str
    symbol: str
    side: str
    type: str
    role: str
    status: str
    qty: str
    price: str | None
    stop_price: str | None
    filled_qty: str
    avg_fill_price: str | None
    fees: str
    submitted_at: datetime | None
    filled_at: datetime | None
    updated_at: datetime


class PositionOut(BaseModel):
    id: str
    broker_account_id: str
    symbol: str
    qty: str
    avg_entry: str
    mark_price: str | None
    unrealized_pnl: str | None
    updated_at: datetime


class EquityPointOut(BaseModel):
    equity: str
    free_balance: str | None
    taken_at: datetime
    is_session_baseline: bool


# --- kill switch --------------------------------------------------------------


class KillSwitchRequest(BaseModel):
    """The confirmation string guards against a fat-fingered POST flattening a book.

    The frontend's big red button asks the user to type CONFIRM; the API refuses
    the request without it. Flattening an account is destructive and instant.
    """

    account_id: str
    confirm: str

    @field_validator("confirm")
    @classmethod
    def _must_confirm(cls, value: str) -> str:
        if value.strip().upper() != "CONFIRM":
            raise ValueError('confirmation must be the exact string "CONFIRM"')
        return value


class KillSwitchOut(BaseModel):
    account_id: str
    locked: bool
    orders_cancelled: int
    positions_closed: int
    clean: bool
    errors: list[str]
