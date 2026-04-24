from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpas.solver.parallel_solver import ParallelLPSolver
from lpas.solver.scipy_handoff import solve_with_scipy
from tests.backend_helpers import make_parallel_solver_config, make_parallel_test_problem


def test_solver_improves_score_and_tracks_history() -> None:
    problem = make_parallel_test_problem()
    config = make_parallel_solver_config(
        backend="torch_cpu",
        dtype="float64",
        samples=2500,
        chunk_size=500,
        iterations=10,
        seed=7,
    )
    result = ParallelLPSolver(config).solve(problem)
    assert len(result.history) == 10
    assert result.history[-1].best_score >= result.history[0].best_score
    assert np.isfinite(result.best_score)
    assert np.isfinite(result.best_primal_objective)
    assert np.isfinite(result.best_gap)


def test_solver_approaches_scipy_optimum_with_loose_tolerance() -> None:
    problem = make_parallel_test_problem()
    scipy_result = solve_with_scipy(problem)
    assert scipy_result.objective is not None
    config = make_parallel_solver_config(
        backend="torch_cpu",
        dtype="float64",
        samples=3000,
        chunk_size=600,
        iterations=12,
        seed=11,
    )
    result = ParallelLPSolver(config).solve(problem)
    assert result.best_primal_objective >= scipy_result.objective - 0.75
    assert result.best_primal_violation <= 0.25


def test_solver_gap_improves_from_first_iteration() -> None:
    problem = make_parallel_test_problem()
    config = make_parallel_solver_config(
        backend="torch_cpu",
        dtype="float64",
        samples=2500,
        chunk_size=500,
        iterations=10,
        seed=21,
    )
    result = ParallelLPSolver(config).solve(problem)
    assert abs(result.history[-1].best_gap) <= abs(result.history[0].best_gap) + 1e-12


def test_solver_produces_no_nans() -> None:
    problem = make_parallel_test_problem()
    config = make_parallel_solver_config(
        backend="torch_cpu",
        dtype="float64",
        samples=2000,
        chunk_size=500,
        iterations=8,
        seed=5,
    )
    result = ParallelLPSolver(config).solve(problem)
    assert np.isfinite(result.best_score)
    assert np.isfinite(result.best_primal_objective)
    assert np.isfinite(result.best_dual_objective)
    assert np.isfinite(result.best_gap)
