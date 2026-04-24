from __future__ import annotations

from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.result import SolverStatus
from lpas.solver.scipy_handoff import compare_adaptive_to_scipy, solve_with_scipy


def test_unbounded_lp_is_not_certified(unbounded_problem, fast_solver_config) -> None:
    result = AdaptiveLPSolver(fast_solver_config).solve(unbounded_problem)
    scipy_result = solve_with_scipy(unbounded_problem)
    comparison = compare_adaptive_to_scipy(result, scipy_result)
    assert scipy_result.status == 3
    assert result.status != SolverStatus.OPTIMAL_CERTIFIED
    assert comparison.suggested_status == SolverStatus.UNBOUNDED_SUSPECTED
