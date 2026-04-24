from __future__ import annotations

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.solver.vertex_polishing import reconstruct_vertex_from_active_set


def test_vertex_reconstruction_solves_known_2d_active_set(small_lp) -> None:
    vertex = reconstruct_vertex_from_active_set(small_lp, (0, 1))
    assert vertex is not None
    assert vertex.feasible
    np.testing.assert_allclose(vertex.x, np.array([2.0, 2.0]))
    assert vertex.objective == 10.0
    np.testing.assert_array_equal(vertex.original_active_mask, np.array([True, True, False]))


def test_singular_active_set_is_skipped(small_lp) -> None:
    assert reconstruct_vertex_from_active_set(small_lp, (1, 3)) is None


def test_infeasible_reconstructed_vertex_is_rejected(small_lp) -> None:
    vertex = reconstruct_vertex_from_active_set(small_lp, (0, 3))
    assert vertex is not None
    assert not vertex.feasible
    assert vertex.primal_violation > 0.0


def test_objective_uses_project_maximization_convention() -> None:
    min_problem = LPProblem(
        A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        b=np.array([4.0, 2.0, 3.0], dtype=float),
        c=np.array([3.0, 2.0], dtype=float),
        sense="min",
    )
    vertex = reconstruct_vertex_from_active_set(min_problem, (0, 1))
    assert vertex is not None
    assert vertex.objective == -10.0
