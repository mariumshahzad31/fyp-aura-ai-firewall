"""
Map live network observations to CSV-schema records so ML features remain sourced from
Dataset-Attacks-Firewall.csv (matched or deterministically sampled rows).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.preprocessing import load_raw_dataset


def _normalize_fw(s: str) -> str:
    return str(s).strip()


def _prefix_key(fw: str, parts: int = 2) -> str:
    p = fw.replace("..", ".").split(".")
    return ".".join([x for x in p if x][:parts]) if p else "unknown"


def build_dataset_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Map firewall-traffic prefix -> row indices for fast lookup."""
    idx: Dict[str, List[int]] = {}
    for i, v in enumerate(df["Firewall Traffics"].astype(str)):
        k = _prefix_key(v, 2)
        idx.setdefault(k, []).append(i)
    return idx


def nearest_dataset_row_for_ip(
    df: pd.DataFrame,
    index: Dict[str, List[int]],
    source_ip: str,
) -> pd.Series:
    """
    Pick a dataset row for a live IP: prefer same 2-octet prefix, else 1-octet, else hash-stable sample.
    """
    ip = _normalize_fw(source_ip)
    for parts in (4, 3, 2, 1):
        key = _prefix_key(ip, parts=min(parts, 4))
        if key in index and index[key]:
            rows = index[key]
            h = int(hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12], 16)
            pick = rows[h % len(rows)]
            return df.iloc[pick]
    h = int(hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12], 16)
    return df.iloc[h % len(df)]


def packet_context_to_record(
    df: pd.DataFrame,
    index: Optional[Dict[str, List[int]]],
    source_ip: str,
    dest_ip: str = "",
    src_port: int = 0,
    dst_port: int = 0,
    proto: str = "ip",
) -> Dict[str, Any]:
    """
    Build a raw CSV-schema dict using a **full** matched dataset row so every engineered
    numeric feature remains sourced from Dataset-Attacks-Firewall.csv. Live IPs/ports are
    attached only in `_live_meta` for audit, firewall actions, and explanations.
    """
    if index is None:
        index = build_dataset_index(df)
    row = nearest_dataset_row_for_ip(df, index, source_ip)
    rec = row.to_dict()
    rec["_live_meta"] = {
        "observed_source_ip": source_ip,
        "dest_ip": dest_ip,
        "src_port": int(src_port),
        "dst_port": int(dst_port),
        "protocol": proto,
        "dataset_proxy_row": int(row.name) if hasattr(row, "name") else None,
    }
    return rec


def strip_live_meta(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove keys not in dataset schema before DataFrame construction."""
    out: List[Dict[str, Any]] = []
    for r in records:
        d = {k: v for k, v in r.items() if not str(k).startswith("_")}
        out.append(d)
    return out
