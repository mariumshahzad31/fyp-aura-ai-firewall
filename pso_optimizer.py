"""
Particle Swarm Optimization (PSO) for AURA decision-threshold tuning.

Purpose (per requirements):
- Optimize risk thresholds
- Optimize anomaly sensitivity
- Optimize CVSS scoring weights (within the fusion policy)

This PSO operates on a continuous vector and calls an external objective function that returns
fitness and optional diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PSOConfig:
    n_particles: int = 28
    n_iters: int = 45
    inertia: float = 0.72
    cognitive: float = 1.45
    social: float = 1.45
    v_max: float = 0.25
    seed: int = 42
    patience: int = 10
    min_improve: float = 1e-4


@dataclass(frozen=True)
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history_best: List[float]
    history_mean: List[float]
    diagnostics: Dict[str, float]


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def pso_optimize(
    dim: int,
    bounds: Optional[List[Tuple[float, float]]],
    objective: Callable[[np.ndarray], Tuple[float, Dict[str, float]]],
    *,
    config: Optional[PSOConfig] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> PSOResult:
    """
    Optimize `objective(position)` where objective returns (fitness, diagnostics).
    """

    cfg = config or PSOConfig()
    rng = np.random.default_rng(cfg.seed)

    if bounds is None:
        bounds = [(0.0, 1.0)] * dim
    if len(bounds) != dim:
        raise ValueError("bounds length must equal dim")

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = np.maximum(1e-9, hi - lo)

    # Init in bounds
    x = lo + rng.random((cfg.n_particles, dim)) * span
    v = (rng.random((cfg.n_particles, dim)) - 0.5) * 2.0 * min(cfg.v_max, 1.0) * span

    pbest = x.copy()
    pbest_fit = np.full(cfg.n_particles, -np.inf, dtype=float)
    pbest_diag: List[Dict[str, float]] = [dict() for _ in range(cfg.n_particles)]

    gbest = x[0].copy()
    gbest_fit = -np.inf
    gbest_diag: Dict[str, float] = {}

    history_best: List[float] = []
    history_mean: List[float] = []

    no_improve = 0

    for it in range(cfg.n_iters):
        fits = np.zeros(cfg.n_particles, dtype=float)
        diags: List[Dict[str, float]] = []

        for i in range(cfg.n_particles):
            fit, d = objective(x[i].copy())
            fits[i] = float(fit)
            diags.append(d)

            if fits[i] > pbest_fit[i]:
                pbest_fit[i] = fits[i]
                pbest[i] = x[i].copy()
                pbest_diag[i] = dict(d)

        best_i = int(np.argmax(fits))
        iter_best = float(fits[best_i])

        if iter_best > gbest_fit + cfg.min_improve:
            gbest_fit = iter_best
            gbest = x[best_i].copy()
            gbest_diag = dict(diags[best_i])
            no_improve = 0
        else:
            no_improve += 1

        history_best.append(float(gbest_fit))
        history_mean.append(float(np.mean(fits)))

        if logger:
            logger(
                f"[PSO] iter={it+1}/{cfg.n_iters} best={gbest_fit:.6f} mean={np.mean(fits):.6f} "
                f"no_improve={no_improve}/{cfg.patience}"
            )

        if no_improve >= cfg.patience:
            if logger:
                logger("[PSO] early-stop: convergence/patience reached")
            break

        # Velocity / position update
        r1 = rng.random((cfg.n_particles, dim))
        r2 = rng.random((cfg.n_particles, dim))
        v = (
            cfg.inertia * v
            + cfg.cognitive * r1 * (pbest - x)
            + cfg.social * r2 * (gbest[np.newaxis, :] - x)
        )
        # Clamp velocity per-dimension
        v = np.clip(v, -cfg.v_max * span, cfg.v_max * span)
        x = x + v
        x = np.minimum(np.maximum(x, lo), hi)

    return PSOResult(
        best_position=gbest,
        best_fitness=float(gbest_fit),
        history_best=history_best,
        history_mean=history_mean,
        diagnostics=gbest_diag,
    )


def _demo_pso_objective(position: np.ndarray) -> tuple[float, dict[str, float]]:
    threshold = float(np.clip(0.3 + 0.4 * np.mean(position), 0.05, 0.95))
    scores = np.random.default_rng(42).random(100)
    y_pred = (scores > threshold).astype(int)
    y_true = np.random.default_rng(42).integers(0, 2, size=100)
    acc = float(np.mean(y_true == y_pred))
    fitness = float(acc - 0.1 * np.mean(y_pred))
    return fitness, {"accuracy": acc, "threshold": threshold}


if __name__ == "__main__":
    result = pso_optimize(
        dim=5,
        bounds=[(0.0, 1.0)] * 5,
        objective=_demo_pso_objective,
        config=PSOConfig(n_particles=20, n_iters=20, patience=5),
        logger=print,
    )
    print("\nPSO demo complete")
    print("Best position:", [float(x) for x in result.best_position.tolist()])
    print("Best fitness:", result.best_fitness)

