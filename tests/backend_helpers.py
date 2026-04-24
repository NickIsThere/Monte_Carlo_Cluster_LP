from __future__ import annotations

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.config import BackendConfig, ParallelScoreConfig, ParallelSolverConfig, SamplerConfig


def make_parallel_test_problem() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 2.0], [4.0, 2.0]], dtype=float),
        b=np.array([4.0, 12.0], dtype=float),
        c=np.array([1.0, 1.0], dtype=float),
        sense="max",
    )


def make_optimal_pair() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([8.0 / 3.0, 2.0 / 3.0], dtype=float)
    y = np.array([1.0 / 3.0, 1.0 / 6.0], dtype=float)
    return x, y


def make_candidate_batches() -> tuple[np.ndarray, np.ndarray]:
    optimal_x, optimal_y = make_optimal_pair()
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            optimal_x,
            [3.0, 1.0],
            [2.0, 0.5],
        ],
        dtype=float,
    )
    Y = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.25],
            optimal_y,
            [0.5, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    return X, Y


def make_parallel_solver_config(
    *,
    backend: str = "numpy_cpu",
    dtype: str = "float64",
    samples: int = 4000,
    chunk_size: int = 1000,
    iterations: int = 12,
    seed: int = 42,
) -> ParallelSolverConfig:
    return ParallelSolverConfig(
        samples_per_iteration=samples,
        chunk_size=chunk_size,
        elite_fraction=0.05,
        iterations=iterations,
        backend=BackendConfig(backend=backend, dtype=dtype, active_epsilon=1e-5, dual_positive_epsilon=1e-6),
        scoring=ParallelScoreConfig(),
        sampler=SamplerConfig(
            seed=seed,
            alpha=0.4,
            sigma_init=1.5,
            sigma_min=1e-3,
            sigma_max=4.0,
            primal_init_mean=0.5,
            dual_init_mean=0.5,
        ),
    )
