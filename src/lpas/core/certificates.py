from __future__ import annotations

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.core.primal_dual import PrimalDualMetrics, evaluate_primal_dual_pair, evaluate_primal_dual_pairs


def is_certified_pair(
    problem: LPProblem,
    x: np.ndarray,
    y: np.ndarray,
    *,
    feasibility_tol: float = 1e-7,
    gap_tol: float = 1e-6,
) -> bool:
    metrics = evaluate_primal_dual_pair(problem, x, y, feasibility_tol=feasibility_tol)
    return bool(
        metrics.primal_feasible
        and metrics.dual_feasible
        and metrics.raw_gap >= -gap_tol
        and metrics.raw_gap <= gap_tol
    )


def select_best_certificate(
    metrics: PrimalDualMetrics,
    *,
    gap_tol: float = 1e-6,
) -> int | None:
    feasible = np.atleast_1d(np.asarray(metrics.primal_feasible)) & np.atleast_1d(np.asarray(metrics.dual_feasible))
    raw_gap = np.atleast_1d(np.asarray(metrics.raw_gap, dtype=float))
    certified = feasible & (raw_gap >= -gap_tol) & (raw_gap <= gap_tol)
    if np.any(certified):
        indices = np.flatnonzero(certified)
        positive_gap = np.maximum(raw_gap[indices], 0.0)
        return int(indices[np.argmin(positive_gap)])
    return None


def select_best_certificate_from_pairs(
    problem: LPProblem,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feasibility_tol: float = 1e-7,
    gap_tol: float = 1e-6,
) -> int | None:
    metrics = evaluate_primal_dual_pairs(problem, X, Y, feasibility_tol=feasibility_tol)
    return select_best_certificate(metrics, gap_tol=gap_tol)
