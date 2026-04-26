from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.config import FeasibilityConfig


@dataclass(frozen=True)
class PrimalFeasibilityReport:
    is_feasible: bool
    max_constraint_violation: float
    sum_constraint_violation: float
    max_nonnegativity_violation: float
    sum_nonnegativity_violation: float
    total_violation: float


def _resolve_feasibility_config(
    *,
    tol: float | None,
    config: FeasibilityConfig | None,
) -> FeasibilityConfig:
    if config is not None:
        return config
    if tol is not None:
        scalar_tol = float(tol)
        return FeasibilityConfig(
            absolute_tolerance=scalar_tol,
            relative_tolerance=0.0,
            nonnegativity_tolerance=scalar_tol,
        )
    return FeasibilityConfig()


def primal_constraint_tolerances(
    problem: LPProblem,
    *,
    tol: float | None = None,
    config: FeasibilityConfig | None = None,
) -> np.ndarray:
    resolved = _resolve_feasibility_config(tol=tol, config=config)
    return resolved.absolute_tolerance + resolved.relative_tolerance * np.abs(np.asarray(problem.b, dtype=float))


def primal_feasibility_report(
    problem: LPProblem,
    x: np.ndarray,
    *,
    tol: float | None = None,
    config: FeasibilityConfig | None = None,
) -> PrimalFeasibilityReport:
    x_array = np.asarray(x, dtype=float)
    resolved = _resolve_feasibility_config(tol=tol, config=config)
    constraint_tolerance = primal_constraint_tolerances(problem, config=resolved)
    constraint_violation = np.maximum(problem.A @ x_array - problem.b - constraint_tolerance, 0.0)
    nonnegativity_violation = np.maximum(-x_array - resolved.nonnegativity_tolerance, 0.0)
    max_constraint_violation = float(np.max(constraint_violation, initial=0.0))
    sum_constraint_violation = float(np.sum(constraint_violation))
    max_nonnegativity_violation = float(np.max(nonnegativity_violation, initial=0.0))
    sum_nonnegativity_violation = float(np.sum(nonnegativity_violation))
    total_violation = float(sum_constraint_violation + sum_nonnegativity_violation)
    return PrimalFeasibilityReport(
        is_feasible=bool(max_constraint_violation <= 0.0 and max_nonnegativity_violation <= 0.0),
        max_constraint_violation=max_constraint_violation,
        sum_constraint_violation=sum_constraint_violation,
        max_nonnegativity_violation=max_nonnegativity_violation,
        sum_nonnegativity_violation=sum_nonnegativity_violation,
        total_violation=total_violation,
    )


def project_nonnegative(z: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(z, dtype=float), 0.0)


def primal_violation_norm(
    problem: LPProblem,
    x: np.ndarray,
    tol: float | None = None,
    config: FeasibilityConfig | None = None,
) -> float:
    return primal_feasibility_report(problem, x, tol=tol, config=config).total_violation


def dual_violation_norm(problem: LPProblem, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    dual_slack = problem.A.T @ y - problem.c
    constraint_violation = np.maximum(-dual_slack, 0.0)
    bound_violation = np.maximum(-y, 0.0)
    return float(np.sum(constraint_violation) + np.sum(bound_violation))


def is_primal_feasible(
    problem: LPProblem,
    x: np.ndarray,
    tol: float | None = 1e-7,
    config: FeasibilityConfig | None = None,
) -> bool:
    return primal_feasibility_report(problem, x, tol=tol, config=config).is_feasible


def is_dual_feasible(problem: LPProblem, y: np.ndarray, tol: float = 1e-7) -> bool:
    y = np.asarray(y, dtype=float)
    return bool(np.all(problem.A.T @ y >= problem.c - tol) and np.all(y >= -tol))
