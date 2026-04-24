from __future__ import annotations

import numpy as np
import pytest

from lpas.backends import create_backend
from lpas.utils.config import ParallelScoreConfig
from tests.backend_helpers import make_candidate_batches, make_parallel_test_problem


def _evaluate_backend(backend_name: str, *, dtype: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    backend = create_backend(backend_name, dtype=dtype)
    problem = make_parallel_test_problem()
    X, Y = make_candidate_batches()
    problem_data = backend.prepare_problem(problem)
    X_backend = backend.to_backend_array(X)
    Y_backend = backend.to_backend_array(Y)
    metrics = backend.evaluate_batch(problem_data, X_backend, Y_backend, active_epsilon=1e-5)
    scores = backend.score_batch(
        problem_data,
        X_backend,
        Y_backend,
        metrics,
        ParallelScoreConfig(),
        active_frequency=None,
        active_epsilon=1e-5,
        dual_positive_epsilon=1e-6,
    )
    converted = {
        key: None if value is None else backend.to_numpy(value)
        for key, value in metrics.items()
    }
    return converted, backend.to_numpy(scores)


def _assert_metrics_close(left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, atol: float) -> None:
    for key in (
        "primal_objective",
        "dual_objective",
        "gap",
        "primal_violation",
        "dual_violation",
        "complementarity",
        "active_count",
    ):
        np.testing.assert_allclose(left[key], right[key], atol=atol, rtol=0.0)


def test_numpy_cpu_matches_torch_cpu() -> None:
    pytest.importorskip("torch")
    expected_metrics, expected_scores = _evaluate_backend("numpy_cpu", dtype="float64")
    actual_metrics, actual_scores = _evaluate_backend("torch_cpu", dtype="float64")
    _assert_metrics_close(expected_metrics, actual_metrics, atol=1e-10)
    np.testing.assert_allclose(expected_scores, actual_scores, atol=1e-10, rtol=0.0)


def test_numpy_cpu_matches_numba_cpu() -> None:
    pytest.importorskip("numba")
    expected_metrics, expected_scores = _evaluate_backend("numpy_cpu", dtype="float64")
    actual_metrics, actual_scores = _evaluate_backend("numba_cpu", dtype="float64")
    _assert_metrics_close(expected_metrics, actual_metrics, atol=1e-10)
    np.testing.assert_allclose(expected_scores, actual_scores, atol=1e-10, rtol=0.0)


def test_numpy_cpu_matches_torch_cuda_if_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    expected_metrics, expected_scores = _evaluate_backend("numpy_cpu", dtype="float32")
    actual_metrics, actual_scores = _evaluate_backend("torch_cuda", dtype="float32")
    _assert_metrics_close(expected_metrics, actual_metrics, atol=1e-4)
    np.testing.assert_allclose(expected_scores, actual_scores, atol=1e-4, rtol=0.0)


def test_numpy_cpu_matches_torch_mps_if_available() -> None:
    torch = pytest.importorskip("torch")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        pytest.skip("MPS is unavailable")
    expected_metrics, expected_scores = _evaluate_backend("numpy_cpu", dtype="float32")
    actual_metrics, actual_scores = _evaluate_backend("torch_mps", dtype="float32")
    _assert_metrics_close(expected_metrics, actual_metrics, atol=1e-4)
    np.testing.assert_allclose(expected_scores, actual_scores, atol=1e-4, rtol=0.0)
