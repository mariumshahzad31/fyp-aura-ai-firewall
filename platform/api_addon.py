"""Additive routes mounted under `/platform/v1`; core `/api/v1/*` unchanged + untouched in source."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.auth import get_current_user
from platform.config import platform_settings
from platform.context import tenant_id_cv
from platform.ml.transformer_infer import TransformerThreatHead, augment_prediction_row
from platform.state import GLOBAL_THREAT_GRAPH
from platform.streaming.kafka_io import AuraKafkaProducer, kafka_emit_threat

router = APIRouter(prefix="/platform/v1", tags=["platform"])


def _inject_predictor():
    from api.main import get_predictor

    return get_predictor()


class PlatformPredictBody(BaseModel):
    records: List[Dict[str, Any]]
    include_explanation: bool = False
    use_llm: bool = False
    tenant_id_override: Optional[str] = Field(default=None)


@router.post("/predict/parallel-seq")
async def platform_parallel_predict(
    body: PlatformPredictBody,
    predictor=Depends(_inject_predictor),
    user: Dict[str, Any] = Depends(get_current_user),
    x_live_ms: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if body.tenant_id_override and platform_settings()["multi_tenant"]:
        tenant_id_cv.set(str(body.tenant_id_override))

    def feat_block_sync() -> Optional[np.ndarray]:
        try:
            import pandas as pd

            from utils.preprocessing import build_feature_frame

            df = pd.DataFrame(body.records)
            feat = build_feature_frame(df)
            risk_obj = predictor
            Xt = np.asarray(risk_obj._transform(feat), dtype=np.float32)  # pylint: disable=protected-access
            return Xt
        except Exception:
            return None

    rows = await run_in_threadpool(
        predictor.predict_records,
        body.records,
        body.include_explanation,
        body.use_llm,
    )
    Xt_for_seq = await run_in_threadpool(feat_block_sync)

    enriched: List[Dict[str, Any]] = []
    head = TransformerThreatHead()
    for row in rows:
        enriched.append(augment_prediction_row(row, Xt_for_seq))

    tenant = tenant_id_cv.get("_default")
    lm = enriched[0].get("live_meta") if enriched else {}
    src_ip = (lm or {}).get("observed_source_ip")

    GLOBAL_THREAT_GRAPH.ingest({"tenant_id": tenant, "source_ip": src_ip, **(enriched[0] if enriched else {})})

    kafka_payload = {
        "tenant_id": tenant,
        "risk_class": (enriched[0] if enriched else {}).get("risk_class"),
        "cve_id": (enriched[0] if enriched else {}).get("cve_id"),
        "source_ip": src_ip,
        "user": user.get("username"),
    }
    AuraKafkaProducer().send_event(kafka_payload)

    if platform_settings()["graph_detection"]:
        graph_view = GLOBAL_THREAT_GRAPH.multi_stage_signals()
    else:
        graph_view = {"graph_detection": False}

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if platform_settings()["kafka_enabled"] and kafka_payload.get("risk_class") in ("Malicious", "Critical"):
        try:
            kafka_emit_threat({**kafka_payload, "graph": graph_view})
        except Exception:
            pass

    out = {
        "predictions": enriched,
        "latency_ms_predict_path": elapsed_ms,
        "transformer_status": getattr(head, "enabled", False),
        "graph_signals": graph_view,
    }
    if x_live_ms is not None:
        out["live_probe_header"] = x_live_ms
    return out


@router.get("/zero-trust/health")
async def zt_ping() -> Dict[str, str]:
    cfg = platform_settings()
    if not cfg["zero_trust"]:
        raise HTTPException(404)
    return {"status": "zero_trust_armed"}


@router.post("/graph/clear")
async def graph_clear() -> Dict[str, str]:
    if not platform_settings()["graph_detection"]:
        raise HTTPException(404)
    GLOBAL_THREAT_GRAPH.clear()
    return {"status": "cleared"}
