from __future__ import annotations

from lpas.experiments.benchmarks import run_benchmark_suite
from lpas.experiments.generators import benchmark_lp_suite, tiny_known_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.utils.config import SolverConfig


def run_single_demo(config: SolverConfig | None = None):
    solver = AdaptiveLPSolver(config)
    return solver.solve(tiny_known_lp())


def run_default_benchmarks(config: SolverConfig | None = None):
    problems = benchmark_lp_suite()
    named = [(f"problem_{index}", problem) for index, problem in enumerate(problems)]
    return run_benchmark_suite(named, config=config)


def run_ablation_study(configs: dict[str, SolverConfig]):
    problem = tiny_known_lp()
    results = {}
    for name, config in configs.items():
        results[name] = AdaptiveLPSolver(config).solve(problem)
    return results
