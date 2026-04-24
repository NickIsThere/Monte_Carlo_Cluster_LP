from __future__ import annotations

from dataclasses import replace

from lpas.experiments.benchmark_runner import GEOMETRY_AWARE_METHOD, run_problem_comparison, run_random_dense_suite
from lpas.experiments.random_dense import build_random_dense_suite
from lpas.experiments.reporting import benchmark_result_row
from lpas.experiments.toy_lps import triangle_unique_optimum


def test_fixed_seed_benchmark_produces_stable_rows(fast_solver_config) -> None:
    config = replace(fast_solver_config, batch_size=96, max_iter=10, patience=5)
    suite = build_random_dense_suite(n_variables=[2], n_constraints=[6], n_instances=2, seed=21)
    rows_a = [benchmark_result_row(result) for result in run_random_dense_suite(suite, config=config)]
    rows_b = [benchmark_result_row(result) for result in run_random_dense_suite(suite, config=config)]
    stable_keys = [
        "problem_name",
        "family",
        "method",
        "seed",
        "n_variables",
        "n_constraints",
        "reference_success",
        "reference_objective",
        "reference_active_set_size",
        "best_feasible_objective",
        "best_objective_any_candidate",
        "best_primal_violation",
        "best_dual_violation",
        "best_gap",
        "best_complementarity_error",
        "active_set_recovery_accuracy",
        "exact_active_set_match",
        "first_recovery_iteration",
        "n_samples_total",
        "objective_gap_to_highs",
    ]
    trimmed_a = [{key: row[key] for key in stable_keys} for row in rows_a]
    trimmed_b = [{key: row[key] for key in stable_keys} for row in rows_b]
    assert trimmed_a == trimmed_b


def test_known_2d_lp_recovers_high_active_set_similarity(fast_solver_config) -> None:
    case = triangle_unique_optimum()
    config = replace(fast_solver_config, batch_size=128, max_iter=12, patience=6)
    results = run_problem_comparison(
        problem_name=case.name,
        family="toy",
        problem=case.problem,
        config=config,
        capture_samples=False,
    )
    geometry_result = next(result for result in results if result.method == GEOMETRY_AWARE_METHOD)
    assert geometry_result.active_set_recovery_accuracy >= 2.0 / 3.0
