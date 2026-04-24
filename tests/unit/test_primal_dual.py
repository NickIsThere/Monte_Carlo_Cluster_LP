from __future__ import annotations

import numpy as np

from lpas.core.primal_dual import evaluate_primal_dual_pair, evaluate_primal_dual_pairs


def test_primal_and_dual_objectives_computed_correctly(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    metrics = evaluate_primal_dual_pair(small_lp, x, y)
    assert metrics.primal_objective == 10.0
    assert metrics.dual_objective == 10.0


def test_gap_and_complementarity_hold_at_optimum(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    metrics = evaluate_primal_dual_pair(small_lp, x, y)
    assert abs(metrics.raw_gap) < 1e-12
    assert abs(metrics.complementarity_error) < 1e-12
    np.testing.assert_allclose(metrics.complementarity_vector, np.zeros_like(y))


def test_infeasible_samples_are_detected_correctly(small_lp) -> None:
    metrics = evaluate_primal_dual_pair(small_lp, np.array([5.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert not metrics.primal_feasible
    assert not metrics.dual_feasible
    assert metrics.primal_violation_norm > 0.0
    assert metrics.dual_violation_norm > 0.0


def test_residual_arrays_have_correct_shape(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    metrics = evaluate_primal_dual_pairs(small_lp, np.stack([x, x]), np.stack([y, y]))
    assert metrics.primal_slack.shape == (2, small_lp.m)
    assert metrics.dual_slack.shape == (2, small_lp.n)
