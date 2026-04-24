from __future__ import annotations

from dataclasses import dataclass

from lpas.experiments.metrics import relative_objective_error
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.result import SolverResult
from lpas.solver.scipy_handoff import ScipySolveResult, populate_reference_diagnostics, solve_with_scipy
from lpas.utils.config import SolverConfig


@dataclass(frozen=True)
class BenchmarkRecord:
    problem_name: str
    adaptive_result: SolverResult
    scipy_result: ScipySolveResult
    relative_error: float | None


def run_benchmark_problem(problem_name: str, problem, config: SolverConfig | None = None) -> BenchmarkRecord:
    solver = AdaptiveLPSolver(config)
    adaptive_result = solver.solve(problem)
    scipy_result = solve_with_scipy(problem)
    relative_error = relative_objective_error(adaptive_result.best_primal_objective, scipy_result.objective)
    populate_reference_diagnostics(adaptive_result, scipy_result)
    return BenchmarkRecord(
        problem_name=problem_name,
        adaptive_result=adaptive_result,
        scipy_result=scipy_result,
        relative_error=relative_error,
    )


def run_benchmark_suite(named_problems: list[tuple[str, object]], config: SolverConfig | None = None) -> list[BenchmarkRecord]:
    return [run_benchmark_problem(name, problem, config=config) for name, problem in named_problems]
