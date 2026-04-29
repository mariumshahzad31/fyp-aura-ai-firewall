"""
External CVE / threat-intelligence correlation (NVD API 2.0) with disk cache.
Augments dataset-backed CVE records with live CVE metadata when available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("aura.threat_intel")

CACHE_DIR = Path(__file__).resolve().parents[1] / "logs" / "intel_cache"
CACHE_TTL_SEC = 86_400


def _cache_key(cve_id: str) -> str:
    return hashlib.sha256(cve_id.upper().encode("utf-8")).hexdigest()[:24]


def _read_cache(cve_id: str) -> Optional[Dict[str, Any]]:
    p = CACHE_DIR / f"{_cache_key(cve_id)}.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - float(raw.get("_ts", 0)) > CACHE_TTL_SEC:
            return None
        return raw.get("data")
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(cve_id: str, data: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_cache_key(cve_id)}.json"
    payload = {"_ts": time.time(), "data": data}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_nvd_cve(cve_id: str) -> Optional[Dict[str, Any]]:
    """
    Query NVD CVE 2.0 API (requires API key for higher rate limits).
    Returns normalized dict or None on failure.
    """
    cached = _read_cache(cve_id)
    if cached is not None:
        return cached
    s = get_settings()
    try:
        import httpx
    except ImportError:
        logger.warning("httpx missing; threat intel fetch skipped.")
        return None
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {"cveId": cve_id.strip().upper()}
    headers = {}
    if s.nvd_api_key:
        headers["apiKey"] = s.nvd_api_key
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("NVD fetch failed for %s: %s", cve_id, exc)
        return None
    vulns = data.get("vulnerabilities") or []
    if not vulns:
        return None
    cve = vulns[0].get("cve") or {}
    metrics = cve.get("metrics", {})
    cvss = None
    for block in metrics.values():
        if isinstance(block, list) and block:
            cvss = block[0].get("cvssData", {}).get("baseScore")
            break
    norm = {
        "cve_id": cve_id.upper(),
        "description": (cve.get("descriptions") or [{}])[0].get("value", ""),
        "cvss_score": cvss,
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(cve_id, norm)
    return norm


def correlate_dataset_cve(
    dataset_cve: str,
    dataset_cvss: float,
) -> Dict[str, Any]:
    """
    Merge dataset row with optional NVD enrichment; always includes dataset baseline.
    """
    out: Dict[str, Any] = {
        "dataset_cve": dataset_cve,
        "dataset_cvss": float(dataset_cvss),
        "intel": None,
    }
    nvd = fetch_nvd_cve(dataset_cve)
    if nvd:
        out["intel"] = nvd
        delta = None
        if nvd.get("cvss_score") is not None:
            delta = float(nvd["cvss_score"]) - float(dataset_cvss)
        out["cvss_delta_vs_dataset"] = delta
    return out
