from __future__ import annotations

from fastapi import FastAPI

from aura_platform.api_addon import router as platform_router
from aura_platform.config import platform_settings
from aura_platform.middleware import TenantMiddleware, ZeroTrustMiddleware


def install_platform_extensions(app: FastAPI) -> None:
    """Attach optional middleware + additive routes without editing existing route modules."""
    _ = platform_settings()
    app.add_middleware(ZeroTrustMiddleware)
    app.add_middleware(TenantMiddleware)
    app.include_router(platform_router)


def build_extended_app() -> FastAPI:
    from api.main import app as aura_app

    install_platform_extensions(aura_app)
    return aura_app
