from __future__ import annotations

import numpy as np

from lpas.core.certificates import is_certified_pair, select_best_certificate
from lpas.core.primal_dual import evaluate_primal_dual_pairs


def test_feasible_primal_dual_pair_with_small_gap_is_certified(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    assert is_certified_pair(small_lp, x, y, gap_tol=1e-9)


def test_infeasible_primal_pair_is_not_certified(small_lp, optimal_pair) -> None:
    _, y = optimal_pair
    assert not is_certified_pair(small_lp, np.array([10.0, 10.0]), y)


def test_infeasible_dual_pair_is_not_certified(small_lp, optimal_pair) -> None:
    x, _ = optimal_pair
    assert not is_certified_pair(small_lp, x, np.array([0.0, 0.0, 0.0]))


def test_negative_gap_due_to_infeasibility_is_rejected(small_lp) -> None:
    x = np.array([100.0, 100.0])
    y = np.array([0.0, 0.0, 0.0])
    assert not is_certified_pair(small_lp, x, y, gap_tol=1e6)


def test_approximate_certificate_selection_works_with_tolerance(small_lp, optimal_pair) -> None:
    x, y = optimal_pair
    Y = np.stack([y, y + np.array([1e-7, 0.0, 0.0])])
    X = np.stack([x, x])
    metrics = evaluate_primal_dual_pairs(small_lp, X, Y)
    best = select_best_certificate(metrics, gap_tol=1e-5)
    assert best == 0
