"""
Central configuration for AURA (env + defaults). Secrets via environment only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _b(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@lru_cache
def get_settings() -> "AuraSettings":
    return AuraSettings()


class AuraSettings:
    """Production-oriented settings loaded once."""

    def __init__(self) -> None:
        self.project_root = ROOT
        self.jwt_secret = os.environ.get("AURA_JWT_SECRET", "change-me-in-production-use-long-random-secret")
        self.jwt_algorithm = os.environ.get("AURA_JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.environ.get("AURA_ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
        self.admin_username = os.environ.get("AURA_ADMIN_USER", "admin")
        self.admin_password = os.environ.get("AURA_ADMIN_PASSWORD", "admin")
        self.user_username = os.environ.get("AURA_USER_NAME", "analyst")
        self.user_password = os.environ.get("AURA_USER_PASSWORD", "analyst")
        self.openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or os.environ.get("AURA_LLM_API_KEY")
        self.openai_base_url: str = os.environ.get("AURA_LLM_BASE_URL", "https://api.openai.com/v1")
        self.openai_model: str = os.environ.get("AURA_LLM_MODEL", "gpt-4o-mini")
        self.llm_enabled: bool = _b("AURA_LLM_ENABLED", "true")
        self.firewall_dry_run: bool = _b("AURA_FIREWALL_DRY_RUN", "true")
        self.firewall_auto_block: bool = _b("AURA_FIREWALL_AUTO_BLOCK", "false")
        self.packet_capture_enabled: bool = _b("AURA_PACKET_CAPTURE", "false")
        # Evolutionary Cybersecurity Enhancement Layer (GA+PSO) policy overlay (non-destructive).
        self.evo_enabled: bool = _b("AURA_EVO_ENABLED", "false")
        self.nvd_api_key: Optional[str] = os.environ.get("NVD_API_KEY")
        self.cors_origins: list[str] = [
            x.strip() for x in os.environ.get("AURA_CORS_ORIGINS", "*").split(",") if x.strip()
        ]
        # Durable event storage (additive; JSONL alert bus remains primary ring buffer).
        self.sqlite_enabled: bool = _b("AURA_SQLITE_ENABLED", "true")
        self.sqlite_path: str = os.environ.get("AURA_SQLITE_PATH", str(ROOT / "logs" / "aura_events.db"))
        # Near real-time scoring worker for packet capture (optional).
        self.realtime_scoring_enabled: bool = _b("AURA_REALTIME_SCORING", "true")
