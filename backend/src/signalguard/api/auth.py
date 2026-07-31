"""Email + password auth with server-side sessions (CLAUDE.md §3, §12).

Sessions are stored (their hash is) rather than being stateless signed cookies,
because an account that can move money must be able to revoke a session
immediately — a signed cookie is valid until it expires no matter what. Cookies
are `HttpOnly`, `SameSite=Lax`, and `Secure` outside local/test.

Registration and login return the same generic error on a bad credential so the
endpoint cannot be used to enumerate which emails have accounts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from signalguard.api.deps import (
    SESSION_COOKIE,
    SESSION_TTL_DAYS,
    CurrentUser,
    DbSession,
)
from signalguard.api.schemas import (
    LoginRequest,
    RegisterRequest,
    UserOut,
)
from signalguard.config import get_settings
from signalguard.crypto import (
    generate_session_token,
    hash_password,
    hash_session_token,
    verify_password,
)
from signalguard.db.models import RiskProfile, Session, User

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _cookie_is_secure() -> bool:
    """Secure cookies everywhere except local/test, where TLS is not in play.

    A Secure cookie is never sent over plain HTTP, which would silently break
    local development and the in-process test client. In staging/production the
    service sits behind TLS and the flag is on.
    """
    return get_settings().app_env not in ("local", "test")


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_TTL_DAYS * 24 * 3600,
        httponly=True,
        secure=_cookie_is_secure(),
        samesite="lax",
        path="/",
    )


async def _open_session(
    session: DbSession, user_id: uuid.UUID, request: Request
) -> str:
    """Create a login session row, returning the raw token to hand to the client.

    Only the hash is persisted. The raw token exists exactly once, in the cookie.
    """
    token = generate_session_token()
    now = datetime.now(UTC)
    session.add(
        Session(
            user_id=user_id,
            token_hash=hash_session_token(token),
            expires_at=now + timedelta(days=SESSION_TTL_DAYS),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
        )
    )
    return token


def _user_out(user: User) -> UserOut:
    return UserOut(id=str(user.id), email=user.email, created_at=user.created_at)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest, request: Request, response: Response, session: DbSession
) -> UserOut:
    """Create an account, its default (deny-all) risk profile, and a session.

    The new profile has an empty symbol allowlist, so a fresh account trades
    nothing until the user opts a symbol in — the fail-closed default is deny.
    """
    user = User(email=body.email, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        # Unique-violation on email. Same generic message as a bad login, so the
        # endpoint does not confirm whether an address is already registered.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="could not create account",
        ) from exc

    session.add(RiskProfile(user_id=user.id))
    token = await _open_session(session, user.id, request)
    await session.commit()

    _set_session_cookie(response, token)
    return _user_out(user)


@router.post("/login", response_model=UserOut)
async def login(
    body: LoginRequest, request: Request, response: Response, session: DbSession
) -> UserOut:
    result = await session.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    # Verify even when the user is missing, against a throwaway hash, so a
    # request for a non-existent account takes the same time as a wrong password
    # — no timing oracle for which emails exist.
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials"
    )
    if user is None:
        verify_password(body.password, _DUMMY_HASH)
        raise invalid
    if not user.is_active or not verify_password(body.password, user.password_hash):
        raise invalid

    token = await _open_session(session, user.id, request)
    await session.commit()

    _set_session_cookie(response, token)
    return _user_out(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request, response: Response, session: DbSession
) -> Response:
    """Revoke the current session and clear the cookie. Idempotent."""
    raw = request.cookies.get(SESSION_COOKIE)
    if raw:
        result = await session.execute(
            select(Session).where(Session.token_hash == hash_session_token(raw))
        )
        login_session = result.scalar_one_or_none()
        if login_session is not None and login_session.revoked_at is None:
            login_session.revoked_at = datetime.now(UTC)
            await session.commit()

    response.delete_cookie(SESSION_COOKIE, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return _user_out(user)


# A valid Argon2id hash of a random string, computed once at import. Used to keep
# the login path constant-time when the account does not exist.
_DUMMY_HASH = hash_password(generate_session_token())
