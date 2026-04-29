"""Model artifacts for dashboard: feature importance, metrics blobs (read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from utils.helpers import LOGS_DIR, MODELS_DIR


def load_classifier_feature_importance(top_n: int = 18) -> pd.DataFrame:
    """RandomForest feature importances with preprocessor feature names."""
    path = MODELS_DIR / "risk_model.pkl"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    bundle = joblib.load(path)
    clf = bundle.get("classifier")
    prep = bundle.get("preprocessor")
    if clf is None or prep is None or not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(clf.feature_importances_))]
    imp = clf.feature_importances_
    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def load_risk_metrics_blob() -> Dict[str, Any]:
    path = MODELS_DIR / "risk_model.pkl"
    if not path.exists():
        return {}
    bundle = joblib.load(path)
    return dict(bundle.get("metrics") or {})


def load_cvss_metrics_blob() -> Dict[str, Any]:
    path = MODELS_DIR / "cvss_model.pkl"
    if not path.exists():
        return {}
    bundle = joblib.load(path)
    return dict(bundle.get("metrics") or {})


def read_last_training_metrics_file() -> Optional[Dict[str, Any]]:
    p = LOGS_DIR / "last_training_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def model_confidence_from_probabilities(prob_row: Optional[Dict[str, float]]) -> float:
    if not prob_row:
        return 0.0
    vals = list(prob_row.values())
    if not vals:
        return 0.0
    return float(max(vals))
