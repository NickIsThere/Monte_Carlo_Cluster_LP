from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lpas.core.feasibility import is_primal_feasible
from lpas.experiments.benchmark_runner import GEOMETRY_AWARE_METHOD, run_problem_comparison
from lpas.experiments.metrics import active_set_jaccard
from lpas.experiments.random_dense import (
    build_random_dense_suite,
    generate_bounded_dense_lp,
    generate_controlled_optimum_lp,
)
from lpas.experiments.reporting import _markdown_table, benchmark_result_row
from lpas.experiments.toy_lps import default_toy_cases, simple_3d_case
from lpas.experiments.visualization import plot_feasible_region
from lpas.solver.hints import evaluate_active_set_hint
from lpas.solver.warm_start import reconstruct_from_active_set


def test_toy_lp_generators_return_expected_shapes() -> None:
    cases = default_toy_cases(seed=5)
    dims = {case.name: case.dimension for case in cases}
    assert dims["triangle_unique"] == 2
    assert dims["polytope_3d"] == 3
    for case in cases:
        assert case.problem.A.shape[1] == case.dimension
        assert case.problem.b.shape[0] == case.problem.A.shape[0]
        assert case.problem.c.shape[0] == case.dimension


def test_random_dense_lp_generators_produce_feasible_instances() -> None:
    bounded = generate_bounded_dense_lp(n_variables=4, n_constraints=9, seed=7)
    controlled = generate_controlled_optimum_lp(n_variables=4, n_constraints=9, seed=11)
    assert is_primal_feasible(bounded.problem, bounded.feasible_point)
    assert is_primal_feasible(controlled.problem, controlled.feasible_point)
    assert controlled.planted_optimum is not None
    assert controlled.planted_active_mask is not None
    assert controlled.problem.A.shape == (9, 4)


def test_active_set_jaccard_and_empty_active_sets_are_safe(small_lp) -> None:
    assert active_set_jaccard(np.array([True, False]), np.array([True, False])) == 1.0
    assert active_set_jaccard(np.array([False, False]), np.array([False, False])) == 1.0
    evaluation = evaluate_active_set_hint(
        small_lp,
        np.empty((0, small_lp.m), dtype=bool),
        np.empty(0, dtype=float),
        np.zeros(small_lp.m, dtype=bool),
    )
    assert evaluation.hint_active_set_jaccard == 1.0
    assert evaluation.constraints_in_top_k_support == 1.0
    assert not evaluation.reconstruction.feasible


def test_benchmark_result_row_contains_required_fields(fast_solver_config) -> None:
    case = default_toy_cases(seed=3)[0]
    config = replace(fast_solver_config, batch_size=96, max_iter=8, patience=4)
    results = run_problem_comparison(
        problem_name=case.name,
        family="toy",
        problem=case.problem,
        config=config,
        capture_samples=False,
    )
    row = benchmark_result_row(results[0])
    required = {
        "problem_name",
        "family",
        "method",
        "best_feasible_objective",
        "best_objective_any_candidate",
        "best_primal_violation",
        "best_dual_violation",
        "best_gap",
        "best_complementarity_error",
        "active_set_recovery_accuracy",
        "time_to_identify_optimal_active_constraints",
        "wall_clock_seconds",
        "n_samples_total",
    }
    assert required.issubset(row.keys())


def test_markdown_table_escapes_pipe_characters() -> None:
    table = _markdown_table(
        [{"Metric": "status | summary", "Mean |raw gap|": "APPROXIMATE | 10\nstable"}],
        ["Metric", "Mean |raw gap|"],
    )
    assert "| Metric | Mean \\|raw gap\\| |" in table
    assert "| status \\| summary | APPROXIMATE \\| 10<br>stable |" in table


def test_known_active_set_corner_reconstruction_still_works(small_lp) -> None:
    hint = reconstruct_from_active_set(small_lp, np.array([True, True, False]))
    assert hint.feasible
    np.testing.assert_allclose(hint.candidate_x, np.array([2.0, 2.0]))


def test_plotting_function_saves_without_gui(tmp_path: Path, fast_solver_config) -> None:
    case = default_toy_cases(seed=2)[0]
    config = replace(fast_solver_config, batch_size=64, max_iter=4, patience=3)
    results = run_problem_comparison(
        problem_name=case.name,
        family="toy",
        problem=case.problem,
        config=config,
        capture_samples=True,
    )
    geometry_result = next(result for result in results if result.method == GEOMETRY_AWARE_METHOD)
    output_path = tmp_path / "toy_plot.png"
    plot_feasible_region(case, geometry_result, output_path)
    assert output_path.exists()


def test_simple_3d_case_shape() -> None:
    case = simple_3d_case(seed=17)
    assert case.problem.A.shape[1] == 3
    assert len(case.axis_limits) == 3


def test_random_dense_suite_respects_grid() -> None:
    suite = build_random_dense_suite(n_variables=[2, 3], n_constraints=[4, 6], n_instances=2, seed=19)
    assert suite
    assert all(instance.problem.n in {2, 3} for instance in suite)
    assert all(instance.problem.m in {4, 6} for instance in suite)
