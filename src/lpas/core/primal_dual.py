from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lpas.core.feasibility import is_dual_feasible, is_primal_feasible
from lpas.core.lp_problem import LPProblem
from lpas.utils.validation import ensure_batch


@dataclass(frozen=True)
class PrimalDualMetrics:
    primal_objective: Any
    dual_objective: Any
    raw_gap: Any
    primal_slack: np.ndarray
    dual_slack: np.ndarray
    primal_violation: np.ndarray
    dual_violation: np.ndarray
    nonneg_x_violation: np.ndarray
    nonneg_y_violation: np.ndarray
    primal_violation_norm: Any
    dual_violation_norm: Any
    complementarity_vector: np.ndarray
    complementarity_error: Any
    primal_feasible: Any
    dual_feasible: Any


def _batch_metrics(problem: LPProblem, X: np.ndarray, Y: np.ndarray, tol: float) -> PrimalDualMetrics:
    AX = X @ problem.A.T
    primal_slack = problem.b - AX
    primal_violation = np.maximum(AX - problem.b, 0.0)
    nonneg_x_violation = np.maximum(-X, 0.0)
    primal_violation_norm = np.sum(primal_violation, axis=1) + np.sum(nonneg_x_violation, axis=1)

    YAT = Y @ problem.A
    dual_slack = YAT - problem.c
    dual_violation = np.maximum(problem.c - YAT, 0.0)
    nonneg_y_violation = np.maximum(-Y, 0.0)
    dual_violation_norm = np.sum(dual_violation, axis=1) + np.sum(nonneg_y_violation, axis=1)

    primal_objective = X @ problem.c
    dual_objective = Y @ problem.b
    raw_gap = dual_objective - primal_objective

    complementarity_vector = Y * primal_slack
    complementarity_error = np.sum(np.abs(complementarity_vector), axis=1)

    primal_feasible = (np.max(primal_violation, axis=1, initial=0.0) <= tol) & (
        np.max(nonneg_x_violation, axis=1, initial=0.0) <= tol
    )
    dual_feasible = (np.max(dual_violation, axis=1, initial=0.0) <= tol) & (
        np.max(nonneg_y_violation, axis=1, initial=0.0) <= tol
    )

    return PrimalDualMetrics(
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        raw_gap=raw_gap,
        primal_slack=primal_slack,
        dual_slack=dual_slack,
        primal_violation=primal_violation,
        dual_violation=dual_violation,
        nonneg_x_violation=nonneg_x_violation,
        nonneg_y_violation=nonneg_y_violation,
        primal_violation_norm=primal_violation_norm,
        dual_violation_norm=dual_violation_norm,
        complementarity_vector=complementarity_vector,
        complementarity_error=complementarity_error,
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
    )


def evaluate_primal_dual_pairs(
    problem: LPProblem,
    X: np.ndarray,
    Y: np.ndarray,
    feasibility_tol: float = 1e-7,
) -> PrimalDualMetrics:
    X_batch = ensure_batch(X, expected_dim=problem.n, name="X")
    Y_batch = ensure_batch(Y, expected_dim=problem.m, name="Y")
    if X_batch.shape[0] != Y_batch.shape[0]:
        raise ValueError("X and Y must have the same batch size")
    return _batch_metrics(problem, X_batch, Y_batch, feasibility_tol)


def evaluate_primal_dual_pair(
    problem: LPProblem,
    x: np.ndarray,
    y: np.ndarray,
    feasibility_tol: float = 1e-7,
) -> PrimalDualMetrics:
    metrics = evaluate_primal_dual_pairs(problem, np.asarray(x, dtype=float), np.asarray(y, dtype=float), feasibility_tol)
    return PrimalDualMetrics(
        primal_objective=float(metrics.primal_objective[0]),
        dual_objective=float(metrics.dual_objective[0]),
        raw_gap=float(metrics.raw_gap[0]),
        primal_slack=metrics.primal_slack[0],
        dual_slack=metrics.dual_slack[0],
        primal_violation=metrics.primal_violation[0],
        dual_violation=metrics.dual_violation[0],
        nonneg_x_violation=metrics.nonneg_x_violation[0],
        nonneg_y_violation=metrics.nonneg_y_violation[0],
        primal_violation_norm=float(metrics.primal_violation_norm[0]),
        dual_violation_norm=float(metrics.dual_violation_norm[0]),
        complementarity_vector=metrics.complementarity_vector[0],
        complementarity_error=float(metrics.complementarity_error[0]),
        primal_feasible=bool(is_primal_feasible(problem, x, feasibility_tol)),
        dual_feasible=bool(is_dual_feasible(problem, y, feasibility_tol)),
    )
