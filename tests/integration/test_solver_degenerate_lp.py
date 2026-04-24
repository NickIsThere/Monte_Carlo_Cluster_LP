from __future__ import annotations

from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.result import SolverStatus
from lpas.solver.scipy_handoff import solve_with_scipy


def test_solver_handles_degenerate_lp(degenerate_problem, fast_solver_config) -> None:
    result = AdaptiveLPSolver(fast_solver_config).solve(degenerate_problem)
    scipy_result = solve_with_scipy(degenerate_problem)
    assert scipy_result.success
    assert result.best_primal_objective is not None
    assert abs(result.best_primal_objective - 1.0) <= 0.1
    assert result.status != SolverStatus.NUMERICAL_FAILURE
