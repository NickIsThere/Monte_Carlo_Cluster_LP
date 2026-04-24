from __future__ import annotations

import numpy as np

from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.warm_start import reconstruct_from_active_set


def test_known_active_set_reconstructs_known_optimum(small_lp) -> None:
    hint = reconstruct_from_active_set(small_lp, np.array([True, True, False]))
    assert hint.feasible
    assert hint.candidate_x is not None
    np.testing.assert_allclose(hint.candidate_x, np.array([2.0, 2.0]))


def test_solver_returns_warm_start_hint(small_lp, fast_solver_config) -> None:
    result = AdaptiveLPSolver(fast_solver_config).solve(small_lp)
    assert result.warm_start_hint is not None
    assert result.warm_start_hint.message
