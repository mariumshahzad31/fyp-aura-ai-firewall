"""
Genetic Algorithm (GA) optimizer for AURA policy feature-selection and rule optimization.

Purpose (per requirements):
- Feature selection (select which AURA signals participate in fusion / decision)
- Firewall rule optimization (optimize rule configuration parameters)

This GA is designed to operate without modifying AURA's core ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 42
    n_gens: int = 35
    tournament_k: int = 4
    crossover_rate: float = 0.9
    mutation_rate: float = 0.08
    adaptive_mutation: bool = True
    elitism: int = 3
    seed: int = 42
    patience: int = 10
    min_improve: float = 1e-4


@dataclass(frozen=True)
class GAResult:
    best_genome: np.ndarray
    best_fitness: float
    history_best: List[float]
    history_mean: List[float]
    diagnostics: Dict[str, float]


def _tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int) -> int:
    n = fitness.size
    idx = rng.integers(0, n, size=k)
    best = int(idx[np.argmax(fitness[idx])])
    return best


def _uniform_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(a.size) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2


def _mutate_bits(rng: np.random.Generator, g: np.ndarray, rate: float) -> np.ndarray:
    out = g.copy()
    flip = rng.random(out.size) < rate
    out[flip] = 1 - out[flip]
    return out


def ga_optimize(
    n_bits: int,
    objective: Callable[[np.ndarray], Tuple[float, Dict[str, float]]],
    *,
    config: Optional[GAConfig] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> GAResult:
    """
    Optimize a bit-vector genome of length n_bits.
    objective(genome)->(fitness, diagnostics)
    """

    cfg = config or GAConfig()
    rng = np.random.default_rng(cfg.seed)

    pop = (rng.random((cfg.pop_size, n_bits)) < 0.5).astype(np.int8)

    best_g = pop[0].copy()
    best_fit = -np.inf
    best_diag: Dict[str, float] = {}

    history_best: List[float] = []
    history_mean: List[float] = []

    no_improve = 0

    for gen in range(cfg.n_gens):
        fits = np.zeros(cfg.pop_size, dtype=float)
        diags: List[Dict[str, float]] = []

        for i in range(cfg.pop_size):
            fit, d = objective(pop[i].copy())
            fits[i] = float(fit)
            diags.append(d)

        gen_best_i = int(np.argmax(fits))
        gen_best = float(fits[gen_best_i])

        if gen_best > best_fit + cfg.min_improve:
            best_fit = gen_best
            best_g = pop[gen_best_i].copy()
            best_diag = dict(diags[gen_best_i])
            no_improve = 0
        else:
            no_improve += 1

        history_best.append(float(best_fit))
        history_mean.append(float(np.mean(fits)))

        # Adaptive mutation: if stagnating, raise mutation a bit.
        mut = cfg.mutation_rate
        if cfg.adaptive_mutation and no_improve > max(2, cfg.patience // 3):
            mut = min(0.25, mut * (1.0 + 0.25 * float(no_improve)))

        if logger:
            logger(
                f"[GA] gen={gen+1}/{cfg.n_gens} best={best_fit:.6f} mean={np.mean(fits):.6f} "
                f"mutation={mut:.3f} no_improve={no_improve}/{cfg.patience}"
            )

        if no_improve >= cfg.patience:
            if logger:
                logger("[GA] early-stop: convergence/patience reached")
            break

        # Elitism: keep top N
        elite_n = max(0, min(int(cfg.elitism), cfg.pop_size))
        elite_idx = np.argsort(-fits)[:elite_n] if elite_n else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if elite_n else np.zeros((0, n_bits), dtype=np.int8)

        # Produce offspring
        next_pop: List[np.ndarray] = [e for e in elites]
        while len(next_pop) < cfg.pop_size:
            p1 = pop[_tournament_select(rng, fits, cfg.tournament_k)]
            p2 = pop[_tournament_select(rng, fits, cfg.tournament_k)]
            if rng.random() < cfg.crossover_rate:
                c1, c2 = _uniform_crossover(rng, p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = _mutate_bits(rng, c1, mut)
            c2 = _mutate_bits(rng, c2, mut)
            next_pop.append(c1.astype(np.int8))
            if len(next_pop) < cfg.pop_size:
                next_pop.append(c2.astype(np.int8))

        pop = np.stack(next_pop[: cfg.pop_size], axis=0).astype(np.int8)

    return GAResult(
        best_genome=best_g,
        best_fitness=float(best_fit),
        history_best=history_best,
        history_mean=history_mean,
        diagnostics=best_diag,
    )

