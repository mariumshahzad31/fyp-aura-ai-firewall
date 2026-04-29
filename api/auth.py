"""JWT authentication with admin / user roles."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import get_settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

_USER_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _user_table() -> Dict[str, Dict[str, Any]]:
    global _USER_CACHE
    if _USER_CACHE is not None:
        return _USER_CACHE
    s = get_settings()
    _USER_CACHE = {
        s.admin_username: {
            "username": s.admin_username,
            "hashed": pwd_context.hash(s.admin_password),
            "role": "admin",
        },
        s.user_username: {
            "username": s.user_username,
            "hashed": pwd_context.hash(s.user_password),
            "role": "user",
        },
    }
    return _USER_CACHE


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    u = _user_table().get(username)
    if not u:
        return None
    if not pwd_context.verify(password, u["hashed"]):
        return None
    return {"username": username, "role": u["role"]}


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    s = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=s.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        if username is None:
            raise cred_exc
        return {"username": username, "role": role}
    except JWTError as exc:
        raise cred_exc from exc


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
