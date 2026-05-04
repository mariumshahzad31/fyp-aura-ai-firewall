from __future__ import annotations

from contextvars import ContextVar

tenant_id_cv: ContextVar[str] = ContextVar("aura_tenant_id", default="_default")
zt_verified_cv: ContextVar[bool] = ContextVar("aura_zt_verified", default=False)


def reset_request_context(*, tenant: str = "_default", zt_verified: bool = False) -> None:
    tenant_id_cv.set(tenant)
    zt_verified_cv.set(zt_verified)
