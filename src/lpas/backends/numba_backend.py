from __future__ import annotations

from typing import Any

import numpy as np

from lpas.backends.numpy_backend import NumpyBackend, active_dual_agreement_numpy, active_support_numpy
from lpas.cpu_accel.numba_kernels import evaluate_primal_dual_batch_numba
from lpas.cpu_accel.numba_scoring import score_primal_dual_batch_numba
from lpas.utils.config import ParallelScoreConfig

try:
    import numba  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime/backend tests
    numba = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


class NumbaBackend(NumpyBackend):
    def __init__(self, requested_backend: str = "numba_cpu", *, dtype: str = "float32") -> None:
        if numba is None:
            raise RuntimeError("The numba_cpu backend was requested, but Numba is not installed.") from _NUMBA_IMPORT_ERROR
        super().__init__(requested_backend=requested_backend, dtype=dtype)
        self.info = self.info.__class__(
            requested_backend=requested_backend,
            resolved_backend="numba_cpu",
            device_name="cpu",
            dtype=dtype,
        )

    def evaluate_batch(
        self,
        problem_data: dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        *,
        active_epsilon: float,
    ) -> dict[str, np.ndarray]:
        return evaluate_primal_dual_batch_numba(problem_data["A"], problem_data["b"], problem_data["c"], X, Y, active_epsilon=active_epsilon)

    def score_batch(
        self,
        problem_data: dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        metrics: dict[str, np.ndarray],
        weights: ParallelScoreConfig,
        *,
        active_frequency: np.ndarray | None,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> np.ndarray:
        active_support = None
        active_agreement = None
        active_conflict = None
        if weights.active_support != 0.0 or weights.active_agreement != 0.0 or weights.active_conflict != 0.0:
            primal_slack = problem_data["b"][None, :] - X @ problem_data["A"].T
            active_mask = primal_slack <= active_epsilon
            if weights.active_support != 0.0 and active_frequency is not None:
                active_support = active_support_numpy(active_mask, active_frequency)
            if weights.active_agreement != 0.0 or weights.active_conflict != 0.0:
                active_agreement, active_conflict = active_dual_agreement_numpy(
                    Y,
                    active_mask,
                    dual_positive_epsilon=dual_positive_epsilon,
                )
        return score_primal_dual_batch_numba(
            metrics,
            weights,
            active_support=active_support,
            active_agreement=active_agreement,
            active_conflict=active_conflict,
        )
