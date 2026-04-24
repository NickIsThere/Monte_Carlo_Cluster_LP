from __future__ import annotations

from pprint import pprint

from lpas.experiments.generators import random_feasible_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.scipy_handoff import compare_adaptive_to_scipy, solve_with_scipy
from lpas.utils.config import SamplerConfig, SolverConfig


def main() -> None:
    problem = random_feasible_lp(n=3, m=5, seed=7)
    config = SolverConfig(
        batch_size=640,
        max_iter=90,
        patience=22,
        sampler=SamplerConfig(seed=7, sigma_init=2.5, primal_init_mean=0.75, dual_init_mean=0.75),
    )
    adaptive_result = AdaptiveLPSolver(config).solve(problem)
    scipy_result = solve_with_scipy(problem)
    comparison = compare_adaptive_to_scipy(adaptive_result, scipy_result)

    pprint(
        {
            "adaptive_status": adaptive_result.status.value,
            "adaptive_objective": adaptive_result.best_primal_objective,
            "scipy_success": scipy_result.success,
            "scipy_objective": scipy_result.objective,
            "relative_error": comparison.relative_objective_error,
        }
    )


if __name__ == "__main__":
    main()
