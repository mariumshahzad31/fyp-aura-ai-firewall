import numpy as np
from typing import Callable

from ga_optimizer import ga_optimize
from pso_optimizer import pso_optimize
from fitness_function import compute_metrics, multi_objective_fitness


def _stdout_logger() -> Callable[[str], None]:
    def _log(msg: str) -> None:
        print(msg, flush=True)

    return _log


def _build_demo_dataset(seed: int = 42, n_samples: int = 100):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n_samples)
    y_pred_base = rng.integers(0, 2, size=n_samples)
    return y_true, y_pred_base


def _compute_demo_fitness(y_true: np.ndarray, threshold: float):
    y_pred = (np.random.rand(len(y_true)) > threshold).astype(int)
    metrics = compute_metrics(
        y_true,
        y_pred,
        latency_ms=110,
        stability=0.8,
    )
    return multi_objective_fitness(metrics), metrics.to_dict()


def _ga_demo_objective(y_true: np.ndarray):
    def objective(genome: np.ndarray) -> tuple[float, dict[str, float]]:
        threshold = float(np.clip(0.3 + 0.4 * np.mean(genome), 0.05, 0.95))
        return _compute_demo_fitness(y_true, threshold)

    return objective


def _pso_demo_objective(y_true: np.ndarray):
    def objective(position: np.ndarray) -> tuple[float, dict[str, float]]:
        threshold = float(np.clip(0.3 + 0.4 * np.mean(position), 0.05, 0.95))
        return _compute_demo_fitness(y_true, threshold)

    return objective


def run_demo() -> None:
    np.random.seed(42)
    y_true, y_pred_base = _build_demo_dataset()

    metrics_base = compute_metrics(
        y_true,
        y_pred_base,
        latency_ms=100,
        stability=0.7,
    )
    fitness_base = multi_objective_fitness(metrics_base)

    print("\n=== BASELINE ===")
    print(metrics_base.to_dict())
    print("Fitness:", fitness_base)

    print("\n=== RUNNING GA ===")
    ga_result = ga_optimize(
        n_bits=10,
        objective=_ga_demo_objective(y_true),
        logger=_stdout_logger(),
    )
    print("Best GA Fitness:", ga_result.best_fitness)

    print("\n=== RUNNING PSO ===")
    pso_result = pso_optimize(
        dim=5,
        bounds=None,
        objective=_pso_demo_objective(y_true),
        logger=_stdout_logger(),
    )
    print("Best PSO Fitness:", pso_result.best_fitness)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
