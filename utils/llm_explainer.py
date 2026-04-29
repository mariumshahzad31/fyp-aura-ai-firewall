"""
LLM-based threat explanation layer (OpenAI-compatible API) with deterministic fallback
built from existing `explain_decision` outputs. Keeps production behavior when API keys absent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import get_settings
from utils.prediction import explain_decision, interpret_risk_band

logger = logging.getLogger("aura.llm")


def _fallback_structured(
    risk_name: str,
    proba_row: np.ndarray,
    anomaly_flag: int,
    cvss_pred: float,
    cwe_name: str,
    summary: str,
    live_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = explain_decision(
        risk_name=risk_name,
        proba_row=proba_row,
        anomaly_flag=anomaly_flag,
        cvss_pred=cvss_pred,
        cwe_name=cwe_name,
        summary=summary,
    )
    steps = [
        "1. Feature vector derived from Dataset-Attacks-Firewall.csv row (CVE, CWE, CVSS, firewall metrics).",
        "2. RandomForest risk probabilities compared to historical class boundaries.",
        "3. Isolation Forest tests for geometric outlier vs training manifold.",
        "4. CVSS regressor refines severity estimate on 0–10 scale.",
    ]
    if live_meta:
        steps.append(
            f"5. Live context: source {live_meta.get('observed_source_ip')} "
            f"{live_meta.get('protocol', '')} "
            f"ports {live_meta.get('src_port')}->{live_meta.get('dst_port')} "
            "(audit only; scoring row is dataset-backed)."
        )
    return {
        "engine": "deterministic_template",
        "threat_summary": base.get("threat_summary", ""),
        "anomaly_context": base.get("anomaly_context", ""),
        "weakness_context": base.get("weakness_context", ""),
        "evidence": base.get("evidence", ""),
        "recommended_actions": base.get("recommended_actions", ""),
        "risk_severity_interpretation": base.get("risk_severity_interpretation", ""),
        "reasoning_steps": steps,
        "attack_interpretation": interpret_risk_band(risk_name, cvss_pred),
    }


async def explain_with_llm_async(
    risk_name: str,
    proba_row: np.ndarray,
    anomaly_flag: int,
    cvss_pred: float,
    cwe_name: str,
    summary: str,
    live_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    s = get_settings()
    fb = _fallback_structured(
        risk_name, proba_row, anomaly_flag, cvss_pred, cwe_name, summary, live_meta
    )
    if not s.llm_enabled or not s.openai_api_key:
        return fb
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; LLM explanations disabled.")
        return fb

    system = (
        "You are a senior SOC analyst. Given structured ML outputs about a firewall/CVE record, "
        "produce concise JSON with keys: threat_narrative (string), reasoning_steps (array of strings), "
        "attack_interpretation (string), recommended_actions (array of strings). "
        "Be factual; do not invent CVE details not in the input."
    )
    user_payload = {
        "risk_class": risk_name,
        "probabilities": {str(i): float(proba_row[i]) for i in range(len(proba_row))},
        "anomaly_flag": anomaly_flag,
        "cvss_predicted": cvss_pred,
        "cwe_name": cwe_name,
        "summary_excerpt": summary[:1200],
        "live_meta": live_meta or {},
        "deterministic_fallback": fb,
    }
    url = f"{s.openai_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"}
    body = {
        "model": s.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text)
    out = {
        "engine": "openai_compatible",
        "threat_summary": parsed.get("threat_narrative", fb["threat_summary"]),
        "reasoning_steps": parsed.get("reasoning_steps", fb["reasoning_steps"]),
        "attack_interpretation": parsed.get("attack_interpretation", fb["attack_interpretation"]),
        "recommended_actions": " ".join(parsed.get("recommended_actions", []))
        if isinstance(parsed.get("recommended_actions"), list)
        else str(parsed.get("recommended_actions", fb["recommended_actions"])),
        "anomaly_context": fb["anomaly_context"],
        "weakness_context": fb["weakness_context"],
        "evidence": fb["evidence"],
        "risk_severity_interpretation": fb["risk_severity_interpretation"],
    }
    return out


def explain_with_llm_sync(
    risk_name: str,
    proba_row: np.ndarray,
    anomaly_flag: int,
    cvss_pred: float,
    cwe_name: str,
    summary: str,
    live_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronous LLM call (httpx) for Streamlit and CLI; avoids nested asyncio issues."""
    s = get_settings()
    fb = _fallback_structured(
        risk_name, proba_row, anomaly_flag, cvss_pred, cwe_name, summary, live_meta
    )
    if not s.llm_enabled or not s.openai_api_key:
        return fb
    try:
        import httpx
    except ImportError:
        return fb
    system = (
        "You are a senior SOC analyst. Given structured ML outputs about a firewall/CVE record, "
        "produce concise JSON with keys: threat_narrative (string), reasoning_steps (array of strings), "
        "attack_interpretation (string), recommended_actions (array of strings). "
        "Be factual; do not invent CVE details not in the input."
    )
    user_payload = {
        "risk_class": risk_name,
        "probabilities": {str(i): float(proba_row[i]) for i in range(len(proba_row))},
        "anomaly_flag": anomaly_flag,
        "cvss_predicted": cvss_pred,
        "cwe_name": cwe_name,
        "summary_excerpt": summary[:1200],
        "live_meta": live_meta or {},
        "deterministic_fallback": fb,
    }
    url = f"{s.openai_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"}
    body = {
        "model": s.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        return {
            "engine": "openai_compatible",
            "threat_summary": parsed.get("threat_narrative", fb["threat_summary"]),
            "reasoning_steps": parsed.get("reasoning_steps", fb["reasoning_steps"]),
            "attack_interpretation": parsed.get("attack_interpretation", fb["attack_interpretation"]),
            "recommended_actions": " ".join(parsed.get("recommended_actions", []))
            if isinstance(parsed.get("recommended_actions"), list)
            else str(parsed.get("recommended_actions", fb["recommended_actions"])),
            "anomaly_context": fb["anomaly_context"],
            "weakness_context": fb["weakness_context"],
            "evidence": fb["evidence"],
            "risk_severity_interpretation": fb["risk_severity_interpretation"],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM sync failed: %s", exc)
        return fb


def enhance_explanation_dict(
    base_explanation: Dict[str, str],
    use_llm: bool,
    risk_name: str,
    proba_row: np.ndarray,
    anomaly_flag: int,
    cvss_pred: float,
    cwe_name: str,
    summary: str,
    live_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not use_llm:
        fb = _fallback_structured(
            risk_name, proba_row, anomaly_flag, cvss_pred, cwe_name, summary, live_meta
        )
        return {**fb, **{k: v for k, v in base_explanation.items() if v}}
    merged = explain_with_llm_sync(
        risk_name, proba_row, anomaly_flag, cvss_pred, cwe_name, summary, live_meta
    )
    merged["deterministic_explanation"] = base_explanation
    return merged
