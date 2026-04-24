from __future__ import annotations

from lpas.experiments.benchmarks import run_benchmark_suite
from lpas.experiments.generators import benchmark_lp_suite


def test_small_benchmark_quality_against_scipy(benchmark_solver_config) -> None:
    problems = benchmark_lp_suite(seed=3, count=10)
    named = [(f"benchmark_{index}", problem) for index, problem in enumerate(problems)]
    results = run_benchmark_suite(named, config=benchmark_solver_config)
    failures = [record for record in results if record.scipy_result.success and (record.relative_error is None or record.relative_error > 1e-2)]
    assert not failures
