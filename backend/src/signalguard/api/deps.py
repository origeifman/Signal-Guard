"""Shared FastAPI dependencies for the dashboard API.

The important one is `current_user`: it turns a session cookie into a `User`, and
it **fails closed**. A missing cookie, an unknown token, a revoked session, or an
expired one all resolve to 401 — never to "assume logged in". The token is looked
up by its hash, so a stolen database yields no usable sessions (§12).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalguard.crypto import hash_session_token
from signalguard.db.models import Session, User
from signalguard.db.session import get_session

# Name kept short and non-descriptive on purpose — a cookie called
# "auth_token" advertises what it is worth stealing.
SESSION_COOKIE = "sg_session"

# How long a login lasts before it must be re-established. Sessions are also
# revocable server-side (that is the whole reason the token is stored), so this
# is a ceiling, not the only control.
SESSION_TTL_DAYS = 7


DbSession = Annotated[AsyncSession, Depends(get_session)]


async def current_user(
    session: DbSession,
    sg_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User:
    """Resolve the logged-in user, or raise 401. Never returns None."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="not authenticated",
    )
    if not sg_session:
        raise unauthorized

    token_hash = hash_session_token(sg_session)
    result = await session.execute(
        select(Session, User)
        .join(User, User.id == Session.user_id)
        .where(Session.token_hash == token_hash)
    )
    row = result.first()
    if row is None:
        raise unauthorized

    login: Session = row[0]
    user: User = row[1]

    now = datetime.now(UTC)
    if login.revoked_at is not None:
        raise unauthorized
    if login.expires_at <= now:
        raise unauthorized
    if not user.is_active:
        raise unauthorized

    return user


CurrentUser = Annotated[User, Depends(current_user)]
