from __future__ import annotations

from dataclasses import replace

from lpas.experiments.benchmark_runner import (
    GEOMETRY_AWARE_METHOD,
    GEOMETRY_AWARE_POLISHED_METHOD,
    run_problem_comparison,
)
from lpas.experiments.random_dense import build_random_dense_suite
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.utils.config import VertexPolishingConfig


def test_polishing_does_not_worsen_seeded_geometry_results(fast_solver_config) -> None:
    config = replace(
        fast_solver_config,
        batch_size=96,
        max_iter=10,
        patience=5,
        vertex_polishing=VertexPolishingConfig(
            elite_fraction=0.3,
            max_candidates_per_sample=24,
            max_total_candidates=96,
        ),
    )
    suite = build_random_dense_suite(n_variables=[2], n_constraints=[6], n_instances=3, seed=13)
    for instance in suite:
        results = {
            result.method: result
            for result in run_problem_comparison(
                problem_name=instance.name,
                family=instance.family,
                problem=instance.problem,
                config=config,
                capture_samples=False,
            )
        }
        raw = results[GEOMETRY_AWARE_METHOD]
        polished = results[GEOMETRY_AWARE_POLISHED_METHOD]
        if raw.best_feasible_objective is not None and polished.best_feasible_objective is not None:
            assert polished.best_feasible_objective >= raw.best_feasible_objective - 1e-8


def test_polishing_candidate_generation_respects_total_limit(fast_solver_config) -> None:
    config = replace(
        fast_solver_config,
        batch_size=96,
        max_iter=10,
        patience=5,
        vertex_polishing=VertexPolishingConfig(
            elite_fraction=0.5,
            max_candidates_per_sample=8,
            max_total_candidates=8,
        ),
    )
    instance = build_random_dense_suite(n_variables=[2], n_constraints=[6], n_instances=1, seed=23)[0]
    results = {
        result.method: result
        for result in run_problem_comparison(
            problem_name=instance.name,
            family=instance.family,
            problem=instance.problem,
            config=config,
            capture_samples=False,
        )
    }
    polished = results[GEOMETRY_AWARE_POLISHED_METHOD]
    assert polished.polishing_candidates_generated <= 8


def test_polishing_handles_degenerate_lp_without_crashing(degenerate_problem, fast_solver_config) -> None:
    config = replace(
        fast_solver_config,
        vertex_polishing=VertexPolishingConfig(elite_fraction=1.0, max_total_candidates=32),
    )
    result = AdaptiveLPSolver(config).solve(degenerate_problem)
    assert result.polishing_result is not None
