"""
Evolutionary Cybersecurity Enhancement Layer
===========================================

Integration module that adds GA + PSO optimization on top of AURA's existing ML outputs.

STRICT GUARANTEE:
- Does NOT modify/replace RandomForest, IsolationForest, or GradientBoosting models.
- Only learns/tunes a *decision policy layer* (thresholds + weights + rule config).

Primary integration points:
- Behavioral analysis module: uses `behavioral_deviation` already computed by AURA.
- Risk fusion engine: implemented here as a tunable score combiner.
- Firewall decision system: policy returns action recommendation and "threat" flag.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fitness_function import FitnessWeights, MetricBundle, compute_metrics, multi_objective_fitness, stability_from_bootstrap
from ga_optimizer import GAConfig, GAResult, ga_optimize
from pso_optimizer import PSOConfig, PSOResult, pso_optimize


@dataclass(frozen=True)
class EvoPolicy:
    """
    Tunable policy layer parameters.

    - include_* are GA-controlled "feature selection" toggles over AURA signals.
    - weights_* and thresholds_* are PSO-tuned continuous parameters.
    """

    # GA "feature selection" toggles over available signals
    include_rf_proba: bool = True
    include_iforest: bool = True
    include_anomaly_score: bool = True
    include_cvss_pred: bool = True
    include_behavioral_dev: bool = True

    # PSO continuous fusion weights (normalized at runtime)
    w_rf: float = 0.35
    w_iforest: float = 0.15
    w_anom: float = 0.2
    w_cvss: float = 0.2
    w_beh: float = 0.1

    # Thresholds / sensitivity
    score_threshold: float = 0.58  # decision score in [0,1]
    anomaly_score_threshold: float = 0.55  # anomaly_score in [0,1] (generic uses [0,1]; IF sample scores vary)
    behavioral_threshold: float = 0.35  # behavioral deviation in [0,1]
    min_risk_index: int = 2  # 0..3 (0 Safe .. 3 Critical) used as base threat floor

    # Firewall rule configuration (optimized by GA/PSO jointly)
    block_on_threat: bool = True
    block_requires_anomaly: bool = False
    block_requires_behavioral_spike: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvoPolicy":
        # defensive casting
        def b(x: Any, default: bool) -> bool:
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                return x.strip().lower() in ("1", "true", "yes", "on")
            return default

        def f(x: Any, default: float) -> float:
            try:
                return float(x)
            except Exception:
                return default

        def i(x: Any, default: int) -> int:
            try:
                return int(x)
            except Exception:
                return default

        base = EvoPolicy()
        return EvoPolicy(
            include_rf_proba=b(d.get("include_rf_proba"), base.include_rf_proba),
            include_iforest=b(d.get("include_iforest"), base.include_iforest),
            include_anomaly_score=b(d.get("include_anomaly_score"), base.include_anomaly_score),
            include_cvss_pred=b(d.get("include_cvss_pred"), base.include_cvss_pred),
            include_behavioral_dev=b(d.get("include_behavioral_dev"), base.include_behavioral_dev),
            w_rf=f(d.get("w_rf"), base.w_rf),
            w_iforest=f(d.get("w_iforest"), base.w_iforest),
            w_anom=f(d.get("w_anom"), base.w_anom),
            w_cvss=f(d.get("w_cvss"), base.w_cvss),
            w_beh=f(d.get("w_beh"), base.w_beh),
            score_threshold=f(d.get("score_threshold"), base.score_threshold),
            anomaly_score_threshold=f(d.get("anomaly_score_threshold"), base.anomaly_score_threshold),
            behavioral_threshold=f(d.get("behavioral_threshold"), base.behavioral_threshold),
            min_risk_index=max(0, min(3, i(d.get("min_risk_index"), base.min_risk_index))),
            block_on_threat=b(d.get("block_on_threat"), base.block_on_threat),
            block_requires_anomaly=b(d.get("block_requires_anomaly"), base.block_requires_anomaly),
            block_requires_behavioral_spike=b(
                d.get("block_requires_behavioral_spike"), base.block_requires_behavioral_spike
            ),
        )

    def fused_score(self, row: Dict[str, Any]) -> float:
        """
        Fuse AURA outputs into a normalized score in [0,1] (higher => more threat-like).
        Inputs are taken from `utils.prediction.AuraPredictor.predict_records` output + behavioral fusion.
        """

        parts: List[Tuple[float, float]] = []  # (weight, value)

        if self.include_rf_proba:
            probs = row.get("risk_probabilities") or {}
            try:
                mal = float(probs.get("Malicious", 0.0))
                crit = float(probs.get("Critical", 0.0))
                susp = float(probs.get("Suspicious", 0.0))
                rf_val = float(np.clip(0.55 * mal + 0.85 * crit + 0.25 * susp, 0.0, 1.0))
            except Exception:
                rf_val = 0.0
            parts.append((float(self.w_rf), rf_val))

        if self.include_iforest:
            # IF outputs flag (-1 outlier, 1 inlier). Convert to {0,1}.
            try:
                flag = int(row.get("anomaly_score_flag", 1))
                if_val = 1.0 if flag < 0 else 0.0
            except Exception:
                if_val = 0.0
            parts.append((float(self.w_iforest), if_val))

        if self.include_anomaly_score:
            try:
                # Note: prediction.py returns raw IF score_samples, while generic path uses [0,1].
                # Normalize robustly by logistic map after sign flip heuristics.
                s = float(row.get("anomaly_score", 0.0))
                # If values look like [0,1], use directly; else squish.
                if 0.0 <= s <= 1.0:
                    an = s
                else:
                    an = float(1.0 / (1.0 + np.exp(-np.clip(s, -10.0, 10.0))))
            except Exception:
                an = 0.0
            parts.append((float(self.w_anom), float(np.clip(an, 0.0, 1.0))))

        if self.include_cvss_pred:
            try:
                cv = float(row.get("cvss_predicted", 0.0)) / 10.0
            except Exception:
                cv = 0.0
            parts.append((float(self.w_cvss), float(np.clip(cv, 0.0, 1.0))))

        if self.include_behavioral_dev:
            beh = row.get("behavioral") or {}
            try:
                dev = float(beh.get("behavioral_deviation", 0.0))
            except Exception:
                dev = 0.0
            parts.append((float(self.w_beh), float(np.clip(dev, 0.0, 1.0))))

        if not parts:
            return 0.0
        wsum = float(sum(max(0.0, w) for w, _ in parts))
        if wsum <= 0:
            return 0.0
        val = float(sum(max(0.0, w) * float(v) for w, v in parts) / wsum)
        return float(np.clip(val, 0.0, 1.0))

    def decide(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return an "optimized decision" without mutating base AURA outputs.
        """

        score = self.fused_score(row)

        # Base threat floor from model risk_code
        try:
            base_code = int(row.get("risk_code", 0))
        except Exception:
            base_code = 0
        base_threat = base_code >= int(self.min_risk_index)

        # Optional gating signals
        anom_ok = True
        if self.block_requires_anomaly:
            try:
                anom_ok = int(row.get("anomaly_score_flag", 1)) < 0
            except Exception:
                anom_ok = False
        beh_ok = True
        if self.block_requires_behavioral_spike:
            beh = row.get("behavioral") or {}
            try:
                beh_ok = float(beh.get("behavioral_deviation", 0.0)) >= float(self.behavioral_threshold)
            except Exception:
                beh_ok = False

        threat = bool((score >= float(self.score_threshold)) or base_threat)
        action = "ALLOW+MONITOR"
        if threat:
            action = "RATE_LIMIT+LOG"
            if self.block_on_threat and anom_ok and beh_ok:
                action = "DROP+BLOCK+ALERT"

        return {
            "evo_score": float(score),
            "evo_threat": bool(threat),
            "evo_action": str(action),
            "policy": {
                "score_threshold": float(self.score_threshold),
                "min_risk_index": int(self.min_risk_index),
                "block_on_threat": bool(self.block_on_threat),
                "block_requires_anomaly": bool(self.block_requires_anomaly),
                "block_requires_behavioral_spike": bool(self.block_requires_behavioral_spike),
            },
        }


def default_policy_path(project_root: Optional[Path] = None) -> Path:
    here = Path(__file__).resolve().parent
    root = project_root or here
    return root / "saved_models" / "evo_policy.json"


def load_policy(path: Optional[Path] = None) -> Optional[EvoPolicy]:
    p = path or default_policy_path()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            return EvoPolicy.from_dict(d)
    except Exception:
        return None
    return None


def save_policy(policy: EvoPolicy, path: Optional[Path] = None) -> Path:
    p = path or default_policy_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(policy.to_json(), encoding="utf-8")
    return p


def _risk_name_to_threat(risk_name: str, *, min_risk_index: int) -> bool:
    order = ["Safe", "Suspicious", "Malicious", "Critical"]
    try:
        idx = order.index(str(risk_name))
    except ValueError:
        idx = 0
    return idx >= int(min_risk_index)


def evaluate_policy_on_rows(
    rows: List[Dict[str, Any]],
    y_true_risk_name: List[str],
    policy: EvoPolicy,
    *,
    weights: Optional[FitnessWeights] = None,
    bootstrap_k: int = 10,
    rng_seed: int = 42,
    assumed_latency_ms: float = 0.0,
) -> Tuple[float, MetricBundle, Dict[str, Any]]:
    """
    Compute multi-objective fitness for a policy over already-scored rows.
    - `rows` must contain AURA prediction outputs with optional `behavioral`.
    - `y_true_risk_name` are dataset-derived labels ("Safe"/"Suspicious"/"Malicious"/"Critical").
    """

    t0 = time.perf_counter()
    preds: List[bool] = []
    for r in rows:
        d = policy.decide(r)
        preds.append(bool(d["evo_threat"]))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_true_threat = np.array(
        [_risk_name_to_threat(lbl, min_risk_index=policy.min_risk_index) for lbl in y_true_risk_name], dtype=bool
    )
    y_pred_threat = np.array(preds, dtype=bool)

    rng = np.random.default_rng(rng_seed)
    accs: List[float] = []
    n = len(y_true_threat)
    if n > 0:
        for _ in range(max(1, int(bootstrap_k))):
            idx = rng.integers(0, n, size=n)
            accs.append(float(np.mean(y_true_threat[idx] == y_pred_threat[idx])))
    st = stability_from_bootstrap(accs)

    # Latency: combine measured evaluation latency with an external assumption (e.g., model inference).
    latency_ms = float(elapsed_ms + float(assumed_latency_ms))
    metrics = compute_metrics(y_true_threat, y_pred_threat, latency_ms=latency_ms, stability=st)
    fit = multi_objective_fitness(metrics, weights=weights)
    detail = {
        "fitness": float(fit),
        "metrics": metrics.to_dict(),
        "policy": asdict(policy),
        "elapsed_eval_ms": float(elapsed_ms),
    }
    return float(fit), metrics, detail


def _ga_genome_to_policy(base: EvoPolicy, genome: np.ndarray) -> EvoPolicy:
    """
    Map bit genome to EvoPolicy toggles / discrete rules.
    Genome layout (8 bits):
      0 include_rf_proba
      1 include_iforest
      2 include_anomaly_score
      3 include_cvss_pred
      4 include_behavioral_dev
      5 block_on_threat
      6 block_requires_anomaly
      7 block_requires_behavioral_spike
    """

    g = np.asarray(genome, dtype=int).reshape(-1)
    if g.size < 8:
        raise ValueError("genome must have at least 8 bits")
    return EvoPolicy(
        include_rf_proba=bool(g[0]),
        include_iforest=bool(g[1]),
        include_anomaly_score=bool(g[2]),
        include_cvss_pred=bool(g[3]),
        include_behavioral_dev=bool(g[4]),
        w_rf=base.w_rf,
        w_iforest=base.w_iforest,
        w_anom=base.w_anom,
        w_cvss=base.w_cvss,
        w_beh=base.w_beh,
        score_threshold=base.score_threshold,
        anomaly_score_threshold=base.anomaly_score_threshold,
        behavioral_threshold=base.behavioral_threshold,
        min_risk_index=base.min_risk_index,
        block_on_threat=bool(g[5]),
        block_requires_anomaly=bool(g[6]),
        block_requires_behavioral_spike=bool(g[7]),
    )


def _pso_position_to_policy(base: EvoPolicy, pos: np.ndarray) -> EvoPolicy:
    """
    Map PSO position vector -> continuous policy parameters.
    Vector layout (9 dims), all are already within bounds:
      0..4 weights (rf, iforest, anom, cvss, beh)
      5 score_threshold
      6 behavioral_threshold
      7 min_risk_index (continuous mapped -> int 0..3)
      8 block_requires_anomaly_gate (continuous -> bool)
    """

    x = np.asarray(pos, dtype=float).reshape(-1)
    if x.size < 9:
        raise ValueError("PSO position must have 9 dims")

    # weights: keep small floor to avoid zeroing everything
    w_rf, w_if, w_an, w_cv, w_be = [float(max(0.0, v)) for v in x[:5]]
    score_th = float(np.clip(x[5], 0.05, 0.95))
    beh_th = float(np.clip(x[6], 0.0, 1.0))
    min_idx = int(np.clip(np.round(x[7]), 0, 3))
    req_anom = bool(x[8] >= 0.5)

    return EvoPolicy(
        include_rf_proba=base.include_rf_proba,
        include_iforest=base.include_iforest,
        include_anomaly_score=base.include_anomaly_score,
        include_cvss_pred=base.include_cvss_pred,
        include_behavioral_dev=base.include_behavioral_dev,
        w_rf=w_rf,
        w_iforest=w_if,
        w_anom=w_an,
        w_cvss=w_cv,
        w_beh=w_be,
        score_threshold=score_th,
        anomaly_score_threshold=base.anomaly_score_threshold,
        behavioral_threshold=beh_th,
        min_risk_index=min_idx,
        block_on_threat=base.block_on_threat,
        block_requires_anomaly=req_anom,
        block_requires_behavioral_spike=base.block_requires_behavioral_spike,
    )


def tune_evolutionary_layer(
    scored_rows: List[Dict[str, Any]],
    y_true_risk_name: List[str],
    *,
    fitness_weights: Optional[FitnessWeights] = None,
    ga_cfg: Optional[GAConfig] = None,
    pso_cfg: Optional[PSOConfig] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run GA (discrete feature/rule selection) then PSO (continuous threshold/weights) using the
    multi-objective fitness defined in `fitness_function.py`.
    """

    w = fitness_weights or FitnessWeights()
    base = EvoPolicy()

    # --- Baseline
    base_fit, base_metrics, _ = evaluate_policy_on_rows(scored_rows, y_true_risk_name, base, weights=w)
    if logger:
        logger(
            f"[EVO] baseline fitness={base_fit:.6f} acc={base_metrics.accuracy:.4f} "
            f"fpr={base_metrics.fpr:.4f} fnr={base_metrics.fnr:.4f} lat_ms={base_metrics.latency_ms:.2f} "
            f"stability={base_metrics.stability:.4f}"
        )

    # --- GA optimize toggles + discrete rule config
    def ga_objective(genome: np.ndarray) -> Tuple[float, Dict[str, float]]:
        pol = _ga_genome_to_policy(base, genome)
        fit, m, _detail = evaluate_policy_on_rows(scored_rows, y_true_risk_name, pol, weights=w, bootstrap_k=6)
        return float(fit), {
            "acc": float(m.accuracy),
            "fpr": float(m.fpr),
            "fnr": float(m.fnr),
            "lat_ms": float(m.latency_ms),
            "stability": float(m.stability),
        }

    ga_res: GAResult = ga_optimize(
        n_bits=8,
        objective=ga_objective,
        config=ga_cfg or GAConfig(),
        logger=logger,
    )
    ga_policy = _ga_genome_to_policy(base, ga_res.best_genome)

    # --- PSO optimize continuous weights + thresholds
    bounds = [
        (0.0, 1.0),  # w_rf
        (0.0, 1.0),  # w_iforest
        (0.0, 1.0),  # w_anom
        (0.0, 1.0),  # w_cvss
        (0.0, 1.0),  # w_beh
        (0.05, 0.95),  # score_threshold
        (0.0, 1.0),  # behavioral_threshold
        (0.0, 3.0),  # min_risk_index (continuous -> int)
        (0.0, 1.0),  # block_requires_anomaly gate
    ]

    def pso_objective(pos: np.ndarray) -> Tuple[float, Dict[str, float]]:
        pol = _pso_position_to_policy(ga_policy, pos)
        fit, m, _detail = evaluate_policy_on_rows(scored_rows, y_true_risk_name, pol, weights=w, bootstrap_k=8)
        return float(fit), {
            "acc": float(m.accuracy),
            "fpr": float(m.fpr),
            "fnr": float(m.fnr),
            "lat_ms": float(m.latency_ms),
            "stability": float(m.stability),
        }

    pso_res: PSOResult = pso_optimize(
        dim=9,
        bounds=bounds,
        objective=pso_objective,
        config=pso_cfg or PSOConfig(),
        logger=logger,
    )
    tuned = _pso_position_to_policy(ga_policy, pso_res.best_position)

    tuned_fit, tuned_metrics, tuned_detail = evaluate_policy_on_rows(
        scored_rows, y_true_risk_name, tuned, weights=w, bootstrap_k=12
    )

    if logger:
        logger(
            f"[EVO] tuned fitness={tuned_fit:.6f} acc={tuned_metrics.accuracy:.4f} "
            f"fpr={tuned_metrics.fpr:.4f} fnr={tuned_metrics.fnr:.4f} lat_ms={tuned_metrics.latency_ms:.2f} "
            f"stability={tuned_metrics.stability:.4f}"
        )

    return {
        "baseline": {"fitness": float(base_fit), "metrics": base_metrics.to_dict(), "policy": asdict(base)},
        "ga": {
            "best_fitness": float(ga_res.best_fitness),
            "best_genome": ga_res.best_genome.astype(int).tolist(),
            "history_best": ga_res.history_best,
            "history_mean": ga_res.history_mean,
            "diagnostics": ga_res.diagnostics,
            "policy_after_ga": asdict(ga_policy),
        },
        "pso": {
            "best_fitness": float(pso_res.best_fitness),
            "best_position": [float(x) for x in pso_res.best_position.tolist()],
            "history_best": pso_res.history_best,
            "history_mean": pso_res.history_mean,
            "diagnostics": pso_res.diagnostics,
        },
        "tuned": {"fitness": float(tuned_fit), "metrics": tuned_metrics.to_dict(), "policy": asdict(tuned)},
        "tuned_detail": tuned_detail,
    }


def build_scored_dataset(
    predictor: Any,
    df_raw: pd.DataFrame,
    *,
    max_rows: int = 900,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Create a scored dataset for tuning:
    - Uses the existing predictor to get AURA outputs.
    - Injects AURA behavioral fusion fields deterministically.
    - Produces y_true risk labels from dataset CVSS->risk mapping already in preprocessing.
    """

    from utils.preprocessing import RISK_NAMES, build_feature_frame
    from utils.behavioral_profiles import combine_scores, get_behavior_store

    if df_raw is None or df_raw.empty:
        raise ValueError("df_raw is empty")

    feat = build_feature_frame(df_raw)
    n = min(int(max_rows), len(feat))
    if n < 10:
        raise ValueError("Not enough rows to tune evolutionary layer")

    # Sample stratified-ish by time (keeps chronology variety but avoids huge inference cost)
    ordered = feat.sort_values("pub_ts").reset_index(drop=True)
    pick = np.linspace(0, len(ordered) - 1, num=n, dtype=int)
    idx = ordered.iloc[pick].index.to_numpy()

    raw_sel = df_raw.iloc[idx].reset_index(drop=True)
    y_true = [RISK_NAMES[int(x)] for x in build_feature_frame(raw_sel)["risk_class"].to_numpy(dtype=int)]

    records = [raw_sel.iloc[i].to_dict() for i in range(len(raw_sel))]

    t0 = time.perf_counter()
    scored = predictor.predict_records(records, include_explanation=False, use_llm=False)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Add behavioral fusion fields (uses existing store; does not alter models)
    store = get_behavior_store()
    now = "offline_tuning"
    for row in scored:
        key = str(row.get("cve_id") or (row.get("live_meta") or {}).get("observed_source_ip") or "offline")
        risk_class = str(row.get("risk_class", "Safe"))
        dev = store.deviation_score(key, risk_class)
        store.record_event(key, risk_class, now)
        row["behavioral"] = combine_scores(int(row.get("risk_code", 0)), dev)

    if logger:
        logger(f"[EVO] built scored dataset n={len(scored)} inference_ms={infer_ms:.2f}")
    return scored, y_true


def run_validation_and_tune(
    predictor: Any,
    df_raw: pd.DataFrame,
    *,
    max_rows: int = 900,
    output_path: Optional[Path] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    scored, y_true = build_scored_dataset(predictor, df_raw, max_rows=max_rows, logger=logger)
    report = tune_evolutionary_layer(scored, y_true, logger=logger)
    tuned_policy = EvoPolicy.from_dict(report["tuned"]["policy"])
    p = save_policy(tuned_policy, path=output_path)
    report["saved_policy_path"] = str(p)
    return report


def _stdout_logger() -> Callable[[str], None]:
    def _log(msg: str) -> None:
        print(msg, flush=True)

    return _log


def main() -> None:
    """
    CLI entry:
      python evo_integration.py --validate
    """

    import argparse

    parser = argparse.ArgumentParser(description="AURA evolutionary optimization layer (GA + PSO).")
    parser.add_argument("--validate", action="store_true", help="Run before/after validation and save policy")
    parser.add_argument("--max-rows", type=int, default=900, help="Max rows to use for tuning")
    parser.add_argument("--out", type=str, default="", help="Optional output policy path")
    args = parser.parse_args()

    if not args.validate:
        print("Nothing to do. Use --validate.", flush=True)
        return

    from utils.prediction import AuraPredictor
    from utils.preprocessing import load_raw_dataset

    predictor = AuraPredictor()
    df = load_raw_dataset()
    out_path = Path(args.out).resolve() if args.out else None
    rep = run_validation_and_tune(
        predictor, df, max_rows=int(args.max_rows), output_path=out_path, logger=_stdout_logger()
    )
    print("\n=== EVOLUTIONARY VALIDATION REPORT ===")
    print(json.dumps({k: rep[k] for k in ("baseline", "ga", "pso", "tuned", "saved_policy_path")}, indent=2))


if __name__ == "__main__":
    main()

