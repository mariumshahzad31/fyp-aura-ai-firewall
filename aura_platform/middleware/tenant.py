from __future__ import annotations

import re
from typing import Callable

from starlette.requests import Request
from starlette.responses import Response

from aura_platform.config import platform_settings
from aura_platform.context import tenant_id_cv


_TENANT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{1,126}$")


class TenantMiddleware:
    HEADER = "X-Tenant-ID"

    def __init__(self, app: Callable):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        req = Request(scope, receive)
        settings = platform_settings()
        if not settings["multi_tenant"]:
            tenant_id_cv.set("_default")
            await self.app(scope, receive, send)
            return
        raw = (req.headers.get(self.HEADER) or "").strip()
        if not raw and settings["require_tenant"]:
            resp = Response('{"detail":"missing tenant"}', status_code=400, media_type="application/json")
            await resp(scope, receive, send)
            return
        tid = raw or "_default"
        if tid != "_default" and not _TENANT_RE.match(tid):
            resp = Response('{"detail":"invalid tenant"}', status_code=400, media_type="application/json")
            await resp(scope, receive, send)
            return
        tenant_id_cv.set(tid)
        await self.app(scope, receive, send)
