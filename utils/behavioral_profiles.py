"""
Per-subject behavioral profiles (source prefix or user id) with exponential moving statistics
and incremental SGD head trained **only** on rows sourced from Dataset-Attacks-Firewall.csv
(auxiliary signal; does not replace persisted RF/LSTM).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from utils.helpers import LOGS_DIR
from utils.preprocessing import RISK_NAMES, build_feature_frame, load_raw_dataset, transform_with_preprocessor

PROFILE_PATH = LOGS_DIR / "behavioral_profiles.json"


@dataclass
class EmaStats:
    rate: float = 0.0
    mal_frac: float = 0.0
    last_ts: Optional[str] = None


class BehavioralProfileStore:
    """Thread-safe EWMA stats per key + optional incremental classifier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema: Dict[str, EmaStats] = {}
        self._alpha = 0.15
        self._sgd: Optional[SGDClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._prep = None
        self._load_disk()

    def _load_disk(self) -> None:
        if not PROFILE_PATH.exists():
            return
        try:
            data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            for k, v in data.get("ema", {}).items():
                self._ema[k] = EmaStats(
                    rate=float(v.get("rate", 0)),
                    mal_frac=float(v.get("mal_frac", 0)),
                    last_ts=v.get("last_ts"),
                )
        except (OSError, json.JSONDecodeError):
            pass

    def _save_disk(self) -> None:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ema": {k: {"rate": s.rate, "mal_frac": s.mal_frac, "last_ts": s.last_ts} for k, s in self._ema.items()},
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_event(
        self,
        key: str,
        risk_class: str,
        timestamp_iso: Optional[str] = None,
    ) -> None:
        mal = 1.0 if risk_class in ("Malicious", "Critical") else 0.0
        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        with self._lock:
            s = self._ema.setdefault(key, EmaStats())
            s.rate = (1 - self._alpha) * s.rate + self._alpha * 1.0
            s.mal_frac = (1 - self._alpha) * s.mal_frac + self._alpha * mal
            s.last_ts = ts
            self._save_disk()

    def deviation_score(self, key: str, risk_class: str) -> float:
        """Higher when current risk conflicts with historical benign pattern."""
        mal = 1.0 if risk_class in ("Malicious", "Critical") else 0.0
        with self._lock:
            s = self._ema.get(key)
            if s is None:
                return 0.0
            # spike if historical mal_frac low but current event malicious
            if mal > 0.5 and s.mal_frac < 0.15:
                return min(1.0, 0.4 + (0.15 - s.mal_frac) * 3.0)
            return 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: {"rate": s.rate, "mal_frac": s.mal_frac, "last_ts": s.last_ts} for k, s in self._ema.items()}


_store: Optional[BehavioralProfileStore] = None
_store_lock = threading.Lock()


def get_behavior_store() -> BehavioralProfileStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = BehavioralProfileStore()
        return _store


def incremental_train_batch(
    df_chunk: pd.DataFrame,
    preprocessor: Any,
) -> Tuple[Optional[SGDClassifier], Dict[str, float]]:
    """
    Fit/update auxiliary SGD on dataset rows only (risk_class from CVSS).
    Returns model and simple accuracy on the chunk.
    """
    if len(df_chunk) < 5:
        return None, {}
    feat = build_feature_frame(df_chunk)
    y = feat["risk_class"].to_numpy(dtype=np.int32)
    X = transform_with_preprocessor(preprocessor, feat)
    X = np.asarray(X, dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SGDClassifier(loss="log_loss", average=True, random_state=42)
    clf.fit(Xs, y)
    pred = clf.predict(Xs)
    acc = float((pred == y).mean())
    return clf, {"auxiliary_sgd_accuracy_batch": acc, "n": float(len(y))}


def combine_scores(ml_risk_code: int, behavioral_deviation: float) -> Dict[str, Any]:
    """Non-destructive fusion for API/UI display."""
    boost = 0
    if behavioral_deviation > 0.35 and ml_risk_code < 2:
        boost = 1
    adj = min(3, ml_risk_code + boost)
    return {
        "ml_risk_code": ml_risk_code,
        "ml_risk_name": RISK_NAMES[ml_risk_code],
        "adaptive_adjustment": boost,
        "fused_risk_name": RISK_NAMES[adj],
        "behavioral_deviation": behavioral_deviation,
    }
