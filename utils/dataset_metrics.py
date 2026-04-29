"""
Aggregated statistics and time-series derived exclusively from Dataset-Attacks-Firewall.
Used by the dashboard for cards, charts, analytics, and log previews when live history is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.preprocessing import build_feature_frame, cvss_to_risk_class, load_raw_dataset


@dataclass
class DatasetSummary:
    row_count: int
    unique_cve: int
    unique_sources: int
    date_min: Optional[datetime]
    date_max: Optional[datetime]
    cvss_mean: float
    cvss_std: float
    risk_counts: Dict[str, int]
    normal_ratio: float
    anomaly_ratio_proxy: float
    top_cwe: List[Tuple[str, int]]


def _risk_name_from_cvss(cvss: float) -> str:
    from utils.preprocessing import RISK_NAMES

    return RISK_NAMES[cvss_to_risk_class(float(cvss))]


def compute_dataset_summary(df: Optional[pd.DataFrame] = None) -> DatasetSummary:
    raw = df if df is not None else load_raw_dataset()
    rc = raw["cvss"].astype(float)
    risk_idx = rc.apply(cvss_to_risk_class)
    from utils.preprocessing import RISK_NAMES

    counts: Dict[str, int] = {name: 0 for name in RISK_NAMES}
    for i in risk_idx:
        counts[RISK_NAMES[int(i)]] += 1
    total = len(raw)
    safe_n = counts.get("Safe", 0)
    normal_ratio = float(safe_n / total) if total else 0.0
    high_n = counts.get("Malicious", 0) + counts.get("Critical", 0)
    anomaly_ratio_proxy = float(high_n / total) if total else 0.0

    pub = pd.to_datetime(raw["pub_date"], dayfirst=True, errors="coerce")
    cwe_counts = raw["cwe_name"].fillna("Unknown").astype(str).str.strip().value_counts().head(8)
    top_cwe = list(zip(cwe_counts.index.tolist(), cwe_counts.values.tolist()))

    return DatasetSummary(
        row_count=int(total),
        unique_cve=int(raw["Data"].nunique()),
        unique_sources=int(raw["Firewall Traffics"].astype(str).nunique()),
        date_min=pub.min(),
        date_max=pub.max(),
        cvss_mean=float(rc.mean()),
        cvss_std=float(rc.std()) if len(rc) > 1 else 0.0,
        risk_counts=counts,
        normal_ratio=normal_ratio,
        anomaly_ratio_proxy=anomaly_ratio_proxy,
        top_cwe=top_cwe,
    )


def traffic_timeseries(df: Optional[pd.DataFrame] = None, freq: str = "D") -> pd.DataFrame:
    """Event counts over publication time (dataset-driven traffic volume)."""
    raw = df if df is not None else load_raw_dataset()
    t = pd.to_datetime(raw["pub_date"], dayfirst=True, errors="coerce")
    s = pd.Series(np.ones(len(raw)), index=t)
    s = s.sort_index()
    agg = s.resample(freq).sum().fillna(0)
    out = agg.reset_index()
    out.columns = ["timestamp", "events"]
    return out


def hourly_traffic_profile(df: Optional[pd.DataFrame] = None, max_points: int = 168) -> pd.DataFrame:
    """Last N hourly buckets of event counts for monitoring charts."""
    ts = traffic_timeseries(df, freq="h")
    if len(ts) > max_points:
        ts = ts.tail(max_points)
    return ts.reset_index(drop=True)


def unique_network_prefixes(df: Optional[pd.DataFrame] = None, top_n: int = 20) -> pd.DataFrame:
    raw = df if df is not None else load_raw_dataset()
    fw = raw["Firewall Traffics"].astype(str)
    pref = fw.apply(lambda x: ".".join(x.split(".")[:2]) if x and x != "nan" else "unknown")
    vc = pref.value_counts().head(top_n).reset_index()
    vc.columns = ["prefix", "events"]
    return vc


def threat_watchlist(df: Optional[pd.DataFrame] = None, top_n: int = 25) -> pd.DataFrame:
    """Source IPs (firewall field) with highest concentration of CVSS >= 6 rows."""
    raw = df if df is not None else load_raw_dataset()
    sub = raw[raw["cvss"].astype(float) >= 6.0].copy()
    if sub.empty:
        sub = raw.copy()
    ips = sub["Firewall Traffics"].astype(str).value_counts().head(top_n).reset_index()
    ips.columns = ["source_ip", "high_risk_events"]
    return ips


def analytics_risk_timeline(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    raw = df if df is not None else load_raw_dataset()
    t = pd.to_datetime(raw["pub_date"], dayfirst=True, errors="coerce")
    raw = raw.copy()
    raw["_ts"] = t
    raw["_risk"] = raw["cvss"].astype(float).apply(cvss_to_risk_class)
    from utils.preprocessing import RISK_NAMES

    raw["_label"] = raw["_risk"].apply(lambda i: RISK_NAMES[int(i)])
    daily = raw.dropna(subset=["_ts"]).groupby([pd.Grouper(key="_ts", freq="D"), "_label"]).size().unstack(fill_value=0)
    daily = daily.sort_index()
    return daily.reset_index()


def cwe_distribution(df: Optional[pd.DataFrame] = None, top_n: int = 15) -> pd.DataFrame:
    raw = df if df is not None else load_raw_dataset()
    vc = raw["cwe_name"].fillna("Unknown").astype(str).str.strip().value_counts().head(top_n)
    out = vc.reset_index()
    out.columns = ["cwe_name", "events"]
    return out


def build_dataset_log_rows(df: Optional[pd.DataFrame] = None, n: int = 200) -> List[Dict[str, Any]]:
    """Structured log-like rows from raw CVE records (for empty simulation history)."""
    raw = df if df is not None else load_raw_dataset()
    tail = raw.tail(min(n, len(raw))).copy()
    rows: List[Dict[str, Any]] = []
    for _, r in tail.iterrows():
        cvss = float(r["cvss"])
        label = _risk_name_from_cvss(cvss)
        fw = str(r.get("Firewall Traffics", ""))
        action = "Monitor"
        if label in ("Malicious", "Critical"):
            action = "Block / Quarantine"
        elif label == "Suspicious":
            action = "Rate limit / Inspect"
        rows.append(
            {
                "timestamp": pd.to_datetime(r["pub_date"], dayfirst=True, errors="coerce"),
                "event_type": str(r.get("Data", "")),
                "source": fw,
                "status": label,
                "action": action,
                "cvss": cvss,
            }
        )
    return rows


def security_state_label(summary: DatasetSummary) -> Tuple[str, str]:
    """Human-readable security posture from dataset risk mix."""
    crit = summary.risk_counts.get("Critical", 0)
    mal = summary.risk_counts.get("Malicious", 0)
    total = summary.row_count or 1
    crit_ratio = crit / total
    if crit_ratio > 0.05 or (crit + mal) / total > 0.35:
        return "ELEVATED", "#f59e0b"
    if (crit + mal) / total > 0.15:
        return "ATTENTION", "#eab308"
    return "STABLE", "#22c55e"


def match_search(row: Dict[str, Any], query: str) -> bool:
    if not query or not query.strip():
        return True
    q = query.strip().lower()
    blob = " ".join(str(v) for v in row.values() if v is not None).lower()
    return q in blob
