"""Optional ASGI entry: `uvicorn platform.asgi_extended:app`.

Core `uvicorn api.main:app` remains unchanged."""

from aura_platform.bootstrap import build_extended_app


app = build_extended_app()
