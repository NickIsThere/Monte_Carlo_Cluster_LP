from __future__ import annotations

import numpy as np

from lpas.core.feasibility import is_primal_feasible
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.scipy_handoff import solve_with_scipy


def test_solver_finds_good_solution_on_small_lp(small_lp, fast_solver_config) -> None:
    result = AdaptiveLPSolver(fast_solver_config).solve(small_lp)
    scipy_result = solve_with_scipy(small_lp)
    assert result.best_x is not None
    assert result.best_primal_objective is not None
    assert scipy_result.success
    assert is_primal_feasible(small_lp, result.best_x, tol=1e-5)
    assert abs(result.best_primal_objective - 10.0) <= 0.5
    assert abs(result.best_primal_objective - scipy_result.objective) <= 0.5
    assert result.best_active_set is not None
    assert result.best_active_set[0][0]
    assert result.best_active_set[0][1]


def test_thin_feasible_region_improves_violation_over_time(thin_problem, fast_solver_config) -> None:
    result = AdaptiveLPSolver(fast_solver_config).solve(thin_problem)
    assert result.history
    first = result.history[0].mean_primal_violation
    last = result.history[-1].mean_primal_violation
    assert last <= first
