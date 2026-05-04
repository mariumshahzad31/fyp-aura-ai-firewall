"""
Multi-objective fitness for AURA's Evolutionary Cybersecurity Enhancement Layer.

This module is intentionally model-agnostic: it evaluates *decision policies* that sit on top of
existing AURA model outputs (RF, IsolationForest, GradientBoosting, optional LSTM), without
changing or retraining those models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FitnessWeights:
    """
    Fitness = w1*Accuracy − w2*FPR − w3*FNR − w4*Latency + w5*Stability
    """

    w_accuracy: float = 1.0
    w_fpr: float = 1.0
    w_fnr: float = 1.25
    w_latency: float = 0.15
    w_stability: float = 0.5


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    fpr: float
    fnr: float
    latency_ms: float
    stability: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "fpr": float(self.fpr),
            "fnr": float(self.fnr),
            "latency_ms": float(self.latency_ms),
            "stability": float(self.stability),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "tn": int(self.tn),
            "fn": int(self.fn),
        }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return (tp, fp, tn, fn) for boolean arrays where positive means "threat".
    """

    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    return tp, fp, tn, fn


def compute_metrics(
    y_true_threat: np.ndarray,
    y_pred_threat: np.ndarray,
    *,
    latency_ms: float,
    stability: float,
) -> MetricBundle:
    tp, fp, tn, fn = confusion_from_binary(y_true_threat, y_pred_threat)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    return MetricBundle(
        accuracy=float(acc),
        fpr=float(fpr),
        fnr=float(fnr),
        latency_ms=float(latency_ms),
        stability=float(stability),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def stability_from_bootstrap(accuracies: Iterable[float]) -> float:
    """
    Convert a list of bootstrap accuracies into a [0,1] stability score.
    Higher is better (lower variance). Uses a soft normalization that avoids hard thresholds.
    """

    vals = np.asarray(list(accuracies), dtype=float)
    if vals.size < 2:
        return 1.0
    std = float(np.std(vals))
    # Map std -> stability. std≈0 => 1.0; std>=0.15 => ~0.4
    return float(1.0 / (1.0 + (std / 0.05) ** 2))


def multi_objective_fitness(
    metrics: MetricBundle,
    weights: Optional[FitnessWeights] = None,
) -> float:
    """
    Fitness = w1*Accuracy − w2*FPR − w3*FNR − w4*Latency + w5*Stability
    Latency is normalized in a smooth way so ms-scale differences do not dominate.
    """

    w = weights or FitnessWeights()
    # Normalize latency: 0ms -> 0 penalty; ~100ms -> ~0.5; ~300ms -> ~0.75
    lat_pen = float(metrics.latency_ms / (metrics.latency_ms + 100.0))
    return float(
        w.w_accuracy * metrics.accuracy
        - w.w_fpr * metrics.fpr
        - w.w_fnr * metrics.fnr
        - w.w_latency * lat_pen
        + w.w_stability * metrics.stability
    )


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0, 1], dtype=int)
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1], dtype=int)
    metrics = compute_metrics(y_true, y_pred, latency_ms=120.0, stability=0.85)
    print("Metric bundle:", metrics.to_dict())
    print("Fitness:", multi_objective_fitness(metrics))

