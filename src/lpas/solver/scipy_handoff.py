from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from lpas.core.active_set import primal_active_mask
from lpas.core.lp_problem import LPProblem
from lpas.geometry.active_set_similarity import jaccard_similarity
from lpas.solver.result import ScipySolveResult, SolverResult, SolverStatus


@dataclass(frozen=True)
class AdaptiveComparison:
    scipy_objective: float | None
    adaptive_objective: float | None
    relative_objective_error: float | None
    suggested_status: SolverStatus | None


def solve_with_scipy(problem: LPProblem, active_tol: float = 1e-6) -> ScipySolveResult:
    linprog_kwargs = problem.to_scipy_linprog()
    objective_multiplier = float(linprog_kwargs.pop("objective_multiplier"))
    result = linprog(**linprog_kwargs)
    x = None if result.x is None else np.asarray(result.x, dtype=float)
    objective = None if result.fun is None else float(objective_multiplier * result.fun)
    active_mask = None
    nonneg_active_mask = None
    augmented_active_mask = None
    if x is not None:
        active_mask = primal_active_mask(problem, x, epsilon=active_tol)
        nonneg_active_mask = np.asarray(x <= active_tol, dtype=bool)
        augmented_active_mask = np.concatenate([active_mask, nonneg_active_mask])
    return ScipySolveResult(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        x=x,
        objective=objective,
        primal_active_mask=active_mask,
        nonneg_active_mask=nonneg_active_mask,
        augmented_primal_active_mask=augmented_active_mask,
        raw_status=result,
    )


def populate_reference_diagnostics(solver_result: SolverResult, scipy_result: ScipySolveResult) -> SolverResult:
    solver_result.scipy_result = scipy_result
    reference_mask = scipy_result.augmented_primal_active_mask
    if reference_mask is None:
        return solver_result
    if solver_result.raw_best_active_mask is not None and solver_result.raw_best_nonneg_active_mask is not None:
        raw_mask = np.concatenate([solver_result.raw_best_active_mask, solver_result.raw_best_nonneg_active_mask])
        solver_result.raw_vs_scipy_active_set_jaccard = jaccard_similarity(raw_mask, reference_mask)
    if solver_result.polished_best_active_mask is not None and solver_result.polished_best_nonneg_active_mask is not None:
        polished_mask = np.concatenate([solver_result.polished_best_active_mask, solver_result.polished_best_nonneg_active_mask])
        solver_result.polished_vs_scipy_active_set_jaccard = jaccard_similarity(polished_mask, reference_mask)
    return solver_result


def compare_adaptive_to_scipy(solver_result: SolverResult, scipy_result: ScipySolveResult) -> AdaptiveComparison:
    populate_reference_diagnostics(solver_result, scipy_result)
    adaptive_objective = solver_result.best_primal_objective
    scipy_objective = scipy_result.objective
    relative_error = None
    if adaptive_objective is not None and scipy_objective is not None:
        denom = max(abs(scipy_objective), 1e-12)
        relative_error = abs(adaptive_objective - scipy_objective) / denom
    suggested_status = None
    if not scipy_result.success and scipy_result.status == 2:
        suggested_status = SolverStatus.INFEASIBLE_SUSPECTED
    elif not scipy_result.success and scipy_result.status == 3:
        suggested_status = SolverStatus.UNBOUNDED_SUSPECTED
    return AdaptiveComparison(
        scipy_objective=scipy_objective,
        adaptive_objective=adaptive_objective,
        relative_objective_error=relative_error,
        suggested_status=suggested_status,
    )
