from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from typing import Callable, Deque

from starlette.requests import Request
from starlette.responses import Response

from aura_platform.config import platform_settings
from aura_platform.context import tenant_id_cv, zt_verified_cv


_NONCES: Deque[tuple[float, str]] = deque(maxlen=5000)
_NONCE_TTL_SEC = 300


def _prune(ts: float) -> None:
    while _NONCES and ts - _NONCES[0][0] > _NONCE_TTL_SEC:
        _NONCES.popleft()


def _replay_ok(nonce: str, stamp: float) -> bool:
    needle = hashlib.sha256((nonce + str(int(stamp))).encode("utf-8")).hexdigest()[:32]
    _prune(stamp)
    for _, n in _NONCES:
        if n == needle:
            return False
    _NONCES.append((stamp, needle))
    return True


def _signature_ok(secret: bytes, tenant: str, path: str, nonce: str, stamp: float, sig_b64like: str) -> bool:
    msg = "|".join([tenant or "_default", path, nonce, str(int(stamp))]).encode("utf-8")
    expect = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, sig_b64like.strip())


class ZeroTrustMiddleware:
    TIMESTAMP_HEADER = "X-AURA-Timestamp"
    NONCE_HEADER = "X-AURA-Nonce"
    SIG_HEADER = "X-AURA-Signature"

    def __init__(self, app: Callable):
        self.app = app

    @staticmethod
    def _skipped(path: str) -> bool:
        if path in ("/health", "/openapi.json", "/docs", "/redoc"):
            return True
        if path.startswith("/static"):
            return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = platform_settings()
        if not settings["zero_trust"]:
            zt_verified_cv.set(True)
            await self.app(scope, receive, send)
            return

        req = Request(scope, receive)
        path = req.url.path
        method = scope.get("method", "GET")
        if method == "OPTIONS" or self._skipped(path):
            zt_verified_cv.set(True)
            await self.app(scope, receive, send)
            return

        secret_b = settings.get("zt_service_secret", "").strip().encode("utf-8")
        hdr_token = settings.get("zt_header_token") or "X-Service-Token"
        tok = (req.headers.get(hdr_token) or "").encode("utf-8")

        if path.startswith("/api/v1/auth/token"):
            zt_verified_cv.set(True)
            await self.app(scope, receive, send)
            return

        if not path.startswith("/api/v1"):
            zt_verified_cv.set(True)
            await self.app(scope, receive, send)
            return

        verified = False
        if secret_b and tok == secret_b:
            verified = True
        elif secret_b:
            try:
                ts_raw = req.headers.get(self.TIMESTAMP_HEADER) or "0"
                stamp = float(ts_raw)
            except ValueError:
                stamp = 0.0
            now = time.time()
            if abs(now - stamp) <= 180.0:
                nonce = (req.headers.get(self.NONCE_HEADER) or "").strip()
                sig = (req.headers.get(self.SIG_HEADER) or "").strip()
                tenant = tenant_id_cv.get("_default")
                if nonce and sig and _replay_ok(nonce, now) and _signature_ok(secret_b, tenant, path, nonce, stamp, sig):
                    verified = True

        if not verified:
            zt_verified_cv.set(False)
            await Response('{"detail":"zero trust verification failed"}', status_code=403, media_type="application/json")(
                scope, receive, send
            )
            return

        zt_verified_cv.set(True)
        await self.app(scope, receive, send)
