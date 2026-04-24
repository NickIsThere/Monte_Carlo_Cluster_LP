from __future__ import annotations

from dataclasses import replace

import numpy as np

from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.scipy_handoff import compare_adaptive_to_scipy, solve_with_scipy
from lpas.utils.config import VertexPolishingConfig


def test_polishing_recovers_known_optimum_and_matches_scipy(small_lp, fast_solver_config) -> None:
    config = replace(
        fast_solver_config,
        vertex_polishing=VertexPolishingConfig(
            elite_fraction=0.2,
            max_candidates_per_sample=64,
            max_total_candidates=256,
        ),
    )
    result = AdaptiveLPSolver(config).solve(small_lp)
    scipy_result = solve_with_scipy(small_lp)
    _ = compare_adaptive_to_scipy(result, scipy_result)

    assert result.raw_best_x is not None
    assert result.polished_best_x is not None
    assert result.raw_best_primal_objective is not None
    assert result.polished_best_primal_objective is not None
    np.testing.assert_allclose(result.polished_best_x, np.array([2.0, 2.0]), atol=1e-6)
    assert result.polished_best_primal_objective >= result.raw_best_primal_objective - 1e-9
    assert result.best_primal_objective >= result.raw_best_primal_objective - 1e-9
    assert result.polished_certified_feasible
    assert result.polished_vs_scipy_active_set_jaccard == 1.0
