from __future__ import annotations

from pprint import pprint

from lpas.experiments.generators import tiny_known_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.scipy_handoff import solve_with_scipy
from lpas.utils.config import SamplerConfig, SolverConfig


def main() -> None:
    problem = tiny_known_lp()
    config = SolverConfig(
        batch_size=512,
        max_iter=80,
        patience=20,
        sampler=SamplerConfig(seed=42, sigma_init=2.5, primal_init_mean=0.75, dual_init_mean=0.75),
    )
    solver = AdaptiveLPSolver(config)
    adaptive_result = solver.solve(problem)
    scipy_result = solve_with_scipy(problem)

    print("Adaptive result")
    pprint(
        {
            "status": adaptive_result.status.value,
            "best_x": None if adaptive_result.best_x is None else adaptive_result.best_x.tolist(),
            "best_primal_objective": adaptive_result.best_primal_objective,
            "solution_source": adaptive_result.solution_source,
            "raw_best_x": None if adaptive_result.raw_best_x is None else adaptive_result.raw_best_x.tolist(),
            "raw_best_primal_objective": adaptive_result.raw_best_primal_objective,
            "polished_best_x": None if adaptive_result.polished_best_x is None else adaptive_result.polished_best_x.tolist(),
            "polished_best_primal_objective": adaptive_result.polished_best_primal_objective,
            "polishing_improved_solution": adaptive_result.polishing_improved_solution,
            "warm_start_message": None if adaptive_result.warm_start_hint is None else adaptive_result.warm_start_hint.message,
        }
    )
    print("\nSciPy baseline")
    pprint(
        {
            "success": scipy_result.success,
            "objective": scipy_result.objective,
            "x": None if scipy_result.x is None else scipy_result.x.tolist(),
        }
    )


if __name__ == "__main__":
    main()
