from __future__ import annotations

import numpy as np
import pytest

from lpas.backends import create_backend
from lpas.cpu_accel.numba_kernels import evaluate_primal_dual_batch_numba
from lpas.cpu_accel.numba_scoring import score_primal_dual_batch_numba
from lpas.utils.config import ParallelScoreConfig
from tests.backend_helpers import make_candidate_batches, make_parallel_test_problem

numba = pytest.importorskip("numba")


def test_numba_evaluation_matches_numpy_on_fixed_data() -> None:
    problem = make_parallel_test_problem()
    X, Y = make_candidate_batches()
    numpy_backend = create_backend("numpy_cpu", dtype="float64")
    metrics_np = numpy_backend.evaluate_batch(
        numpy_backend.prepare_problem(problem),
        X,
        Y,
        active_epsilon=1e-5,
    )
    metrics_numba = evaluate_primal_dual_batch_numba(problem.A, problem.b, problem.c, X, Y, active_epsilon=1e-5)
    for key in ("primal_objective", "dual_objective", "gap", "primal_violation", "dual_violation", "complementarity", "active_count"):
        np.testing.assert_allclose(metrics_np[key], metrics_numba[key], atol=1e-10, rtol=0.0)


def test_numba_score_matches_numpy_score() -> None:
    problem = make_parallel_test_problem()
    X, Y = make_candidate_batches()
    numpy_backend = create_backend("numpy_cpu", dtype="float64")
    metrics_np = numpy_backend.evaluate_batch(
        numpy_backend.prepare_problem(problem),
        X,
        Y,
        active_epsilon=1e-5,
    )
    weights = ParallelScoreConfig()
    numpy_scores = numpy_backend.score_batch(
        numpy_backend.prepare_problem(problem),
        X,
        Y,
        metrics_np,
        weights,
        active_frequency=None,
        active_epsilon=1e-5,
        dual_positive_epsilon=1e-6,
    )
    numba_scores = score_primal_dual_batch_numba(evaluate_primal_dual_batch_numba(problem.A, problem.b, problem.c, X, Y, active_epsilon=1e-5), weights)
    np.testing.assert_allclose(numpy_scores, numba_scores, atol=1e-10, rtol=0.0)


def test_numba_backend_only_used_when_explicitly_requested() -> None:
    assert create_backend("numpy_cpu", dtype="float64").resolved_backend == "numpy_cpu"
    assert create_backend("numba_cpu", dtype="float64").resolved_backend == "numba_cpu"


def test_no_hidden_fallback_to_numba_occurs() -> None:
    backend = create_backend("numpy_cpu", dtype="float64")
    assert backend.__class__.__name__ != "NumbaBackend"
