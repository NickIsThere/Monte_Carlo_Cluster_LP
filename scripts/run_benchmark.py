from __future__ import annotations

from lpas.experiments.generators import benchmark_lp_suite
from lpas.experiments.runners import run_default_benchmarks
from lpas.utils.config import SamplerConfig, SolverConfig


def main() -> None:
    config = SolverConfig(
        batch_size=640,
        max_iter=90,
        patience=22,
        sampler=SamplerConfig(seed=7, sigma_init=2.5, primal_init_mean=0.75, dual_init_mean=0.75),
    )
    _ = benchmark_lp_suite(seed=7, count=10)
    results = run_default_benchmarks(config=config)
    for record in results:
        print(
            record.problem_name,
            {
                "adaptive_status": record.adaptive_result.status.value,
                "adaptive_objective": record.adaptive_result.best_primal_objective,
                "scipy_objective": record.scipy_result.objective,
                "relative_error": record.relative_error,
            },
        )


if __name__ == "__main__":
    main()
