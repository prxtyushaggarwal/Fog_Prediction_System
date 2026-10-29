"""Environment-configured single-account authentication for the fog prediction app."""

import os
from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

bearer_scheme = HTTPBearer(auto_error=False)
ACCESS_TOKEN_MINUTES = 30


def _settings():
    username = os.getenv("AUTH_USERNAME")
    password = os.getenv("AUTH_PASSWORD")
    secret = os.getenv("JWT_SECRET_KEY")
    if not username or not password or not secret or len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured. Set AUTH_USERNAME, "
                "AUTH_PASSWORD, and JWT_SECRET_KEY (at least 32 characters)."
            ),
        )
    return username, password, secret


def authenticate_user(username: str, password: str) -> str:
    configured_username, configured_password, secret = _settings()
    if not (
        compare_digest(username, configured_username)
        and compare_digest(password, configured_password)
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return configured_username


def create_access_token(subject: str) -> str:
    _, _, secret = _settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    _, _, secret = _settings()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "iat", "exp"]},
        )
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject:
            raise InvalidTokenError("Missing subject")
        return subject
    except InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
