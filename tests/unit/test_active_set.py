from __future__ import annotations

import numpy as np
import pytest

from lpas.core.active_set import dual_active_mask, extract_active_sets, primal_active_mask, rank_active_constraints


def test_exact_and_inactive_constraints_detected(small_lp, optimal_pair) -> None:
    x, _ = optimal_pair
    mask = primal_active_mask(small_lp, x, epsilon=1e-8)
    np.testing.assert_array_equal(mask, np.array([True, True, False]))


def test_nearly_active_constraint_detected_with_tolerance(small_lp) -> None:
    x = np.array([2.0, 1.9999995])
    mask = primal_active_mask(small_lp, x, epsilon=1e-5)
    assert mask[0]


def test_batch_active_set_extraction_returns_correct_shapes(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    batch = extract_active_sets(small_lp, np.stack([x, x]), np.stack([y, y]), epsilon=1e-8)
    assert batch.primal_active_mask.shape == (2, small_lp.m)
    assert batch.dual_active_mask.shape == (2, small_lp.n)


def test_dual_active_constraints_detected_correctly(small_lp, optimal_pair) -> None:
    _, y = optimal_pair
    mask = dual_active_mask(small_lp, y, epsilon=1e-8)
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_empty_active_set_handled(small_lp) -> None:
    mask = primal_active_mask(small_lp, np.array([0.0, 0.0]), epsilon=1e-12)
    assert not np.any(mask)


def test_rank_active_constraints_unweighted_and_weighted() -> None:
    masks = np.array(
        [
            [True, False, True],
            [True, True, False],
            [False, True, False],
        ]
    )
    order = rank_active_constraints(masks)
    weighted_order = rank_active_constraints(masks, weights=np.array([0.9, 0.1, 0.1]))
    np.testing.assert_array_equal(order, np.array([0, 1, 2]))
    assert weighted_order[0] == 0


def test_rank_active_constraints_zero_total_and_mismatch() -> None:
    masks = np.array([[True, False], [False, True]])
    zero_weight_order = rank_active_constraints(masks, weights=np.array([0.0, 0.0]))
    np.testing.assert_array_equal(zero_weight_order, np.array([0, 1]))
    np.testing.assert_array_equal(rank_active_constraints(np.empty((0, 2), dtype=bool)), np.array([], dtype=int))
    with pytest.raises(ValueError):
        rank_active_constraints(masks, weights=np.array([1.0]))
