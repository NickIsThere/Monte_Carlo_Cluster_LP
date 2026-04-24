from __future__ import annotations

from typing import Any

import numpy as np

from lpas.cpu_accel.numba_kernels import evaluate_primal_dual_batch_numba
from lpas.utils.config import ParallelScoreConfig

try:
    import numba as nb
except ModuleNotFoundError:  # pragma: no cover - exercised via backend availability tests
    nb = None


if nb is not None:

    @nb.njit(parallel=True, fastmath=False)
    def _score_primal_dual_batch_numba(
        primal_objective: np.ndarray,
        gap: np.ndarray,
        primal_violation: np.ndarray,
        dual_violation: np.ndarray,
        complementarity: np.ndarray,
        active_support: np.ndarray,
        active_agreement: np.ndarray,
        active_conflict: np.ndarray,
        objective_weight: float,
        gap_weight: float,
        primal_violation_weight: float,
        dual_violation_weight: float,
        complementarity_weight: float,
        active_support_weight: float,
        active_agreement_weight: float,
        active_conflict_weight: float,
    ) -> np.ndarray:
        K = primal_objective.shape[0]
        scores = np.empty(K, dtype=primal_objective.dtype)
        for k in nb.prange(K):
            value = objective_weight * primal_objective[k]
            value -= gap_weight * max(gap[k], 0.0)
            value -= primal_violation_weight * primal_violation[k]
            value -= dual_violation_weight * dual_violation[k]
            value -= complementarity_weight * complementarity[k]
            value += active_support_weight * active_support[k]
            value += active_agreement_weight * active_agreement[k]
            value -= active_conflict_weight * active_conflict[k]
            scores[k] = value
        return scores


def score_primal_dual_batch_numba(
    metrics: dict[str, Any],
    weights: ParallelScoreConfig,
    *,
    active_support: np.ndarray | None = None,
    active_agreement: np.ndarray | None = None,
    active_conflict: np.ndarray | None = None,
) -> np.ndarray:
    if nb is None:
        raise RuntimeError("The numba_cpu backend was requested, but Numba is not installed.")
    primal_objective = np.asarray(metrics["primal_objective"])
    zeros = np.zeros_like(primal_objective)
    scores = _score_primal_dual_batch_numba(
        primal_objective,
        np.asarray(metrics["gap"]),
        np.asarray(metrics["primal_violation"]),
        np.asarray(metrics["dual_violation"]),
        np.asarray(metrics["complementarity"]),
        zeros if active_support is None else np.asarray(active_support),
        zeros if active_agreement is None else np.asarray(active_agreement),
        zeros if active_conflict is None else np.asarray(active_conflict),
        weights.objective,
        weights.gap,
        weights.primal_violation,
        weights.dual_violation,
        weights.complementarity,
        weights.active_support,
        weights.active_agreement,
        weights.active_conflict,
    )
    return np.nan_to_num(scores, nan=-1e12, posinf=1e12, neginf=-1e12)


def evaluate_and_score_batch_numba(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    weights: ParallelScoreConfig,
    *,
    active_epsilon: float = 1e-5,
    active_support: np.ndarray | None = None,
    active_agreement: np.ndarray | None = None,
    active_conflict: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = evaluate_primal_dual_batch_numba(A, b, c, X, Y, active_epsilon=active_epsilon)
    scores = score_primal_dual_batch_numba(
        metrics,
        weights,
        active_support=active_support,
        active_agreement=active_agreement,
        active_conflict=active_conflict,
    )
    return metrics, scores
