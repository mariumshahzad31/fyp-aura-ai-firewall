"""
High-level orchestration: dataset row → ML → optional intel → firewall → alert bus.
Use from workers or tests; keeps Streamlit/API logic thin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.alert_bus import append_alert
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.firewall import get_firewall_manager
from utils.helpers import LOGS_DIR, setup_logging
from utils.prediction import AuraPredictor
from utils.sqlite_store import get_event_store
from config.settings import get_settings

try:
    from evo_integration import load_policy
except Exception:  # noqa: BLE001
    load_policy = None  # type: ignore[assignment]

logger = setup_logging("aura.orchestration", log_file=LOGS_DIR / "orchestration.log")

def score_and_respond(
    predictor: AuraPredictor,
    records: List[Dict[str, Any]],
    *,
    use_llm: bool = False,
    auto_firewall: bool = False,
    subject_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = predictor.predict_records(records, include_explanation=True, use_llm=use_llm)
    now = datetime.now(timezone.utc).isoformat()
    settings = get_settings()
    event_store = get_event_store()
    evo_policy = load_policy() if (settings.evo_enabled and load_policy is not None) else None
    if evo_policy is not None:
        logger.info("[EVO] evolutionary policy loaded and applied")
    else:
        logger.info("[EVO] evolutionary policy not active or not loaded")
    for row in out:
        key = subject_key or str(
            (row.get("live_meta") or {}).get("observed_source_ip") or row.get("cve_id") or "unknown"
        )
        behavior_store = get_behavior_store()
        dev = behavior_store.deviation_score(key, str(row.get("risk_class", "Safe")))
        behavior_store.record_event(key, str(row.get("risk_class", "Safe")), now)
        row["behavioral"] = combine_scores(int(row.get("risk_code", 0)), dev)
        if evo_policy is not None:
            try:
                row["evolutionary"] = evo_policy.decide(row)
            except Exception:
                row["evolutionary"] = {"error": "evo_policy_failed"}
        append_alert({"ts": now, "cve_id": row.get("cve_id"), "risk_class": row.get("risk_class"), "source": "orchestration"})
        if event_store is not None:
            try:
                row["timestamp"] = row.get("timestamp") or now
                event_store.insert_scored_event(row, source="orchestration", subject_key=key, ts=now)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sqlite insert failed: %s", str(exc)[:240])
        if auto_firewall and row.get("live_meta"):
            ip = row["live_meta"].get("observed_source_ip")
            if ip:
                risk_class = str(row.get("risk_class"))
                evo = row.get("evolutionary") if isinstance(row.get("evolutionary"), dict) else None
                evo_action = evo.get("evo_action") if isinstance(evo, dict) else None
                should_block = risk_class in ("Malicious", "Critical")
                if isinstance(evo_action, str):
                    should_block = evo_action.upper().startswith("DROP+BLOCK")
                if should_block:
                    get_firewall_manager().block_observed_ip(str(ip), risk_class, "orchestration")
    return out
