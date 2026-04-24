from __future__ import annotations

import numpy as np

from lpas.core.lp_problem import LPProblem


def project_nonnegative(z: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(z, dtype=float), 0.0)


def primal_violation_norm(problem: LPProblem, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    constraint_violation = np.maximum(problem.A @ x - problem.b, 0.0)
    bound_violation = np.maximum(-x, 0.0)
    return float(np.sum(constraint_violation) + np.sum(bound_violation))


def dual_violation_norm(problem: LPProblem, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    dual_slack = problem.A.T @ y - problem.c
    constraint_violation = np.maximum(-dual_slack, 0.0)
    bound_violation = np.maximum(-y, 0.0)
    return float(np.sum(constraint_violation) + np.sum(bound_violation))


def is_primal_feasible(problem: LPProblem, x: np.ndarray, tol: float = 1e-7) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(problem.A @ x <= problem.b + tol) and np.all(x >= -tol))


def is_dual_feasible(problem: LPProblem, y: np.ndarray, tol: float = 1e-7) -> bool:
    y = np.asarray(y, dtype=float)
    return bool(np.all(problem.A.T @ y >= problem.c - tol) and np.all(y >= -tol))
