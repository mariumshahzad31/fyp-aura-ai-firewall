"""Feature flags + integration settings for optional platform layer."""

from __future__ import annotations

import os
from functools import lru_cache


def _b(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@lru_cache
def platform_settings() -> dict:
    return {
        "multi_tenant": _b("AURA_PLATFORM_MULTITENANT", "false"),
        "require_tenant": _b("AURA_PLATFORM_REQUIRE_TENANT_HEADER", "false"),
        "zero_trust": _b("AURA_PLATFORM_ZERO_TRUST", "false"),
        "zt_service_secret": os.environ.get("AURA_SERVICE_TOKEN_SECRET", ""),
        "zt_header_token": os.environ.get("AURA_SERVICE_TOKEN_HEADER", "X-Service-Token"),
        "zt_reauth_paths": "/api/v1",
        "kafka_enabled": _b("AURA_STREAMING_ENABLED", "false"),
        "kafka_bootstrap": os.environ.get("AURA_KAFKA_BOOTSTRAP", "localhost:9092"),
        "kafka_topic_in": os.environ.get("AURA_KAFKA_TOPIC_IN", "aura.events"),
        "kafka_topic_out": os.environ.get("AURA_KAFKA_TOPIC_OUT", "aura.threats"),
        "kafka_consumer_group": os.environ.get("AURA_KAFKA_GROUP", "aura-consumer"),
        "transformer_enabled": _b("AURA_TRANSFORMER_PARALLEL", "false"),
        "transformer_model_path": os.environ.get("AURA_TRANSFORMER_MODEL_PATH", "models/transformer_model.keras"),
        "graph_detection": _b("AURA_GRAPH_DETECTION", "false"),
    }
