from __future__ import annotations

import numpy as np

from lpas.backends import concatenate_metric_dicts, create_backend, slice_metric_dict
from lpas.gpu.memory import estimate_batch_memory_bytes
from lpas.solver.parallel_solver import ParallelLPSolver
from tests.backend_helpers import make_parallel_solver_config, make_parallel_test_problem


def test_chunked_evaluation_returns_same_top_elites_as_full_evaluation() -> None:
    backend = create_backend("numpy_cpu", dtype="float64")
    problem = make_parallel_test_problem()
    problem_data = backend.prepare_problem(problem)
    rng = np.random.default_rng(123)
    X = np.maximum(rng.normal(loc=0.8, scale=0.9, size=(24, problem.n)), 0.0)
    Y = np.maximum(rng.normal(loc=0.4, scale=0.6, size=(24, problem.m)), 0.0)
    metrics = backend.evaluate_batch(problem_data, X, Y, active_epsilon=1e-5)
    scores = backend.score_batch(
        problem_data,
        X,
        Y,
        metrics,
        make_parallel_solver_config().scoring,
        active_frequency=None,
        active_epsilon=1e-5,
        dual_positive_epsilon=1e-6,
    )
    full_X, full_Y, full_scores, _ = backend.select_elites(X, Y, scores, 4)

    local_X = []
    local_Y = []
    local_scores = []
    local_metrics = []
    for start in range(0, X.shape[0], 6):
        X_chunk = X[start : start + 6]
        Y_chunk = Y[start : start + 6]
        metrics_chunk = backend.evaluate_batch(problem_data, X_chunk, Y_chunk, active_epsilon=1e-5)
        scores_chunk = backend.score_batch(
            problem_data,
            X_chunk,
            Y_chunk,
            metrics_chunk,
            make_parallel_solver_config().scoring,
            active_frequency=None,
            active_epsilon=1e-5,
            dual_positive_epsilon=1e-6,
        )
        elite_X, elite_Y, elite_scores, elite_indices = backend.select_elites(X_chunk, Y_chunk, scores_chunk, 4)
        local_X.append(elite_X)
        local_Y.append(elite_Y)
        local_scores.append(elite_scores)
        local_metrics.append(slice_metric_dict(backend, metrics_chunk, elite_indices))
    merged_X = backend.concatenate(local_X, axis=0)
    merged_Y = backend.concatenate(local_Y, axis=0)
    merged_scores = backend.concatenate(local_scores, axis=0)
    _ = concatenate_metric_dicts(backend, local_metrics)
    chunk_X, chunk_Y, chunk_scores, _ = backend.select_elites(merged_X, merged_Y, merged_scores, 4)

    np.testing.assert_allclose(full_scores, chunk_scores)
    np.testing.assert_allclose(full_X, chunk_X)
    np.testing.assert_allclose(full_Y, chunk_Y)


def test_memory_estimator_increases_with_k_n_and_m() -> None:
    base = estimate_batch_memory_bytes(10, 2, 3)
    assert estimate_batch_memory_bytes(20, 2, 3) > base
    assert estimate_batch_memory_bytes(10, 3, 3) > base
    assert estimate_batch_memory_bytes(10, 2, 4) > base


def test_solver_runs_with_chunk_size_smaller_than_samples_per_iteration() -> None:
    problem = make_parallel_test_problem()
    config = make_parallel_solver_config(
        backend="numpy_cpu",
        dtype="float64",
        samples=1200,
        chunk_size=300,
        iterations=4,
    )
    result = ParallelLPSolver(config).solve(problem)
    assert len(result.history) == 4
    assert np.isfinite(result.best_score)
    assert np.isfinite(result.best_primal_objective)
