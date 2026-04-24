from __future__ import annotations

import numpy as np

from lpas.core.primal_dual import PrimalDualMetrics, evaluate_primal_dual_pairs
from lpas.core.scoring import score_candidates
from lpas.utils.config import ScoringConfig


def _fake_metrics(scale: float = 1.0) -> PrimalDualMetrics:
    return PrimalDualMetrics(
        primal_objective=np.array([10.0, 4.0]) * scale,
        dual_objective=np.array([10.0, 15.0]) * scale,
        raw_gap=np.array([0.0, 11.0]) * scale,
        primal_slack=np.zeros((2, 2)),
        dual_slack=np.zeros((2, 2)),
        primal_violation=np.array([[0.0, 0.0], [1.0, 0.0]]),
        dual_violation=np.array([[0.0, 0.0], [0.0, 2.0]]),
        nonneg_x_violation=np.zeros((2, 2)),
        nonneg_y_violation=np.zeros((2, 2)),
        primal_violation_norm=np.array([0.0, 1.0]),
        dual_violation_norm=np.array([0.0, 2.0]),
        complementarity_vector=np.array([[0.0, 0.0], [0.0, 3.0]]),
        complementarity_error=np.array([0.0, 3.0]),
        primal_feasible=np.array([True, False]),
        dual_feasible=np.array([True, False]),
    )


def test_feasible_low_gap_pair_scores_better_than_infeasible_high_objective_pair(small_lp, optimal_pair) -> None:
    x_opt, y_opt = optimal_pair
    X = np.vstack([x_opt, np.array([6.0, 6.0])])
    Y = np.vstack([y_opt, np.array([0.0, 0.0, 0.0])])
    metrics = evaluate_primal_dual_pairs(small_lp, X, Y)
    scores = score_candidates(metrics)
    assert scores[0] > scores[1]


def test_increasing_penalties_reduce_score() -> None:
    metrics = _fake_metrics()
    scores = score_candidates(metrics)
    assert scores[0] > scores[1]


def test_geometry_reward_improves_score_only_when_enabled() -> None:
    metrics = _fake_metrics()
    no_geo = score_candidates(metrics, geometry_support=np.array([0.0, 1.0]), config=ScoringConfig(w_geo=0.0))
    with_geo = score_candidates(metrics, geometry_support=np.array([0.0, 1.0]), config=ScoringConfig(w_geo=1.0))
    assert no_geo[0] > no_geo[1]
    assert with_geo[1] > no_geo[1]


def test_rank_based_score_is_stable_under_objective_scaling() -> None:
    base_scores = score_candidates(_fake_metrics(scale=1.0))
    scaled_scores = score_candidates(_fake_metrics(scale=100.0))
    np.testing.assert_allclose(base_scores, scaled_scores)


def test_no_nan_scores_are_produced() -> None:
    metrics = _fake_metrics()
    metrics = PrimalDualMetrics(
        **{
            **metrics.__dict__,
            "primal_objective": np.array([np.nan, 1.0]),
        }
    )
    scores = score_candidates(metrics)
    assert np.all(np.isfinite(scores))
