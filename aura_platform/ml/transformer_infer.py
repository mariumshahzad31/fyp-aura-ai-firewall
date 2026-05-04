"""Parallel Transformer sequence head (additive to LSTM; does not touch AuraPredictor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from aura_platform.config import platform_settings


@lru_cache(maxsize=1)
def _load_keras_optional(path_str: str):
    try:
        from tensorflow import keras  # noqa: PLC0415
    except ImportError:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    return keras.models.load_model(str(p))


class TransformerThreatHead:
    seq_len_default = 12

    def __init__(self, model_path: Optional[str] = None) -> None:
        cfg = platform_settings()
        self.enabled = cfg["transformer_enabled"]
        self.path = model_path or cfg["transformer_model_path"]

    def _pad(self, feats: np.ndarray) -> Tuple[np.ndarray, int]:
        n = feats.shape[0]
        target = max(2, min(self.seq_len_default, n))
        if n >= target:
            block = feats[-target:]
        else:
            pad_row = feats[-1:] if len(feats) else np.zeros((1, feats.shape[1] if feats.ndim > 1 else 1))
            repeats = np.repeat(pad_row, target - n, axis=0)
            block = np.vstack([repeats, feats])[-target:]
        seq = np.asarray(block, dtype=np.float32)[np.newaxis, ...]
        return seq, target

    def predict_probs(self, feature_matrix_rows: np.ndarray) -> Dict[str, Any]:
        if not self.enabled:
            return {"engine": "disabled"}
        model = _load_keras_optional(self.path)
        if model is None:
            return {"engine": "missing_or_tf", "detail": self.path}

        feats = np.asarray(feature_matrix_rows, dtype=np.float32)
        seq, seq_len_used = self._pad(feats)
        probs = np.asarray(model.predict(seq, verbose=0), dtype=np.float32).ravel()
        top_idx = int(np.argmax(probs)) if probs.size else 0
        return {"engine": "transformer_parallel", "seq_len_used": seq_len_used, "probs": probs.tolist(), "top_idx": top_idx}


def augment_prediction_row(row: Dict[str, Any], X_rows: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if not platform_settings()["transformer_enabled"]:
        return row
    if X_rows is None or X_rows.size == 0:
        return row
    head = TransformerThreatHead()
    extra = head.predict_probs(X_rows)
    out = dict(row)
    seq = out.get("parallel_sequence_models") if isinstance(out.get("parallel_sequence_models"), dict) else {}
    merged = dict(seq)
    merged["transformer_head"] = extra
    out["parallel_sequence_models"] = merged
    return out
