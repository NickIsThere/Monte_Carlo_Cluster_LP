from __future__ import annotations

import numpy as np

from lpas.core.active_set import primal_active_mask
from lpas.core.feasibility import is_primal_feasible, primal_violation_norm
from lpas.experiments.random_dense import generate_controlled_optimum_lp
from lpas.solver.result import ArchiveEntry
from lpas.solver.vertex_polishing import polish_archive
from lpas.utils.config import VertexPolishingConfig


def _archive_entry(problem, x: np.ndarray, *, score: float) -> ArchiveEntry:
    x_array = np.asarray(x, dtype=float)
    return ArchiveEntry(
        x=x_array,
        y=np.zeros(problem.m, dtype=float),
        score=score,
        primal_objective=float(problem.c @ x_array),
        dual_objective=0.0,
        raw_gap=0.0,
        primal_violation=primal_violation_norm(problem, x_array),
        dual_violation=0.0,
        complementarity_error=0.0,
        primal_feasible=is_primal_feasible(problem, x_array, tol=1e-7),
        dual_feasible=False,
        primal_active_mask=primal_active_mask(problem, x_array, epsilon=1e-2),
        dual_active_mask=np.zeros(problem.n, dtype=bool),
    )


def test_polishing_recovers_known_optimum_from_near_corner_archive(small_lp) -> None:
    archive = [
        _archive_entry(small_lp, np.array([1.99, 2.0]), score=1.0),
        _archive_entry(small_lp, np.array([2.0, 1.99]), score=0.9),
        _archive_entry(small_lp, np.array([1.98, 2.01]), score=0.8),
    ]
    result = polish_archive(
        small_lp,
        archive,
        config=VertexPolishingConfig(elite_fraction=1.0, max_total_candidates=32),
    )
    assert result.best_vertex is not None
    assert result.best_vertex.feasible
    np.testing.assert_allclose(result.best_vertex.x, np.array([2.0, 2.0]), atol=1e-6)
    assert result.best_vertex.objective >= 10.0 - 1e-8
    assert result.improvement_over_raw is not None
    assert result.improvement_over_raw >= 0.0


def test_polishing_recovers_known_3d_planted_vertex() -> None:
    instance = generate_controlled_optimum_lp(n_variables=3, n_constraints=7, seed=17)
    x_star = np.asarray(instance.planted_optimum, dtype=float)
    archive = [
        _archive_entry(instance.problem, x_star + np.array([1e-3, -2e-3, 1e-3]), score=1.0),
        _archive_entry(instance.problem, x_star + np.array([-1.5e-3, 1e-3, -5e-4]), score=0.9),
        _archive_entry(instance.problem, x_star + np.array([7e-4, 8e-4, -1.2e-3]), score=0.8),
    ]
    result = polish_archive(
        instance.problem,
        archive,
        config=VertexPolishingConfig(elite_fraction=1.0, max_total_candidates=64),
    )
    assert result.best_vertex is not None
    assert result.best_vertex.feasible
    np.testing.assert_allclose(result.best_vertex.x, x_star, atol=1e-5)
