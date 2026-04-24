from __future__ import annotations

import numpy as np

from lpas.solver.vertex_polishing import (
    augment_primal_constraints,
    compute_soft_activity_scores,
    generate_soft_active_set_candidates,
)


def test_soft_activity_score_is_highest_at_zero_slack() -> None:
    slacks = np.array([0.0, 1e-3, 1e-2], dtype=float)
    scores = compute_soft_activity_scores(slacks, tau=1e-2, method="rbf")
    assert scores[0] == np.max(scores)
    assert np.isclose(scores[0], 1.0)


def test_soft_activity_score_decreases_as_slack_grows() -> None:
    slacks = np.array([0.0, 5e-3, 2e-2], dtype=float)
    rbf_scores = compute_soft_activity_scores(slacks, tau=1e-2, method="rbf")
    reciprocal_scores = compute_soft_activity_scores(slacks, tau=1e-2, method="reciprocal")
    assert rbf_scores[0] > rbf_scores[1] > rbf_scores[2]
    assert reciprocal_scores[0] > reciprocal_scores[1] > reciprocal_scores[2]


def test_negative_slack_is_scored_without_hiding_infeasibility(small_lp) -> None:
    A_aug, b_aug = augment_primal_constraints(small_lp)
    infeasible_sample = np.array([[2.1, 2.05]], dtype=float)
    slacks = b_aug - A_aug @ infeasible_sample[0]
    scores = compute_soft_activity_scores(slacks, tau=1e-2, method="rbf")
    candidates = generate_soft_active_set_candidates(
        infeasible_sample,
        A_aug,
        b_aug,
        small_lp.n,
        tau=1e-2,
        sample_primal_violations=np.array([0.15], dtype=float),
    )
    assert np.all(np.isfinite(scores))
    assert candidates
    assert candidates[0].sample_primal_violation == 0.15


def test_augmented_constraints_include_nonnegativity_boundaries(small_lp) -> None:
    A_aug, b_aug = augment_primal_constraints(small_lp)
    assert A_aug.shape == (small_lp.m + small_lp.n, small_lp.n)
    np.testing.assert_allclose(A_aug[: small_lp.m], small_lp.A)
    np.testing.assert_allclose(A_aug[small_lp.m :], -np.eye(small_lp.n))
    np.testing.assert_allclose(b_aug[: small_lp.m], small_lp.b)
    np.testing.assert_allclose(b_aug[small_lp.m :], np.zeros(small_lp.n))


def test_candidate_generation_returns_unique_sets_of_size_n(small_lp) -> None:
    A_aug, b_aug = augment_primal_constraints(small_lp)
    elite_samples = np.array(
        [
            [1.99, 2.0],
            [2.0, 1.99],
            [1.98, 2.01],
        ],
        dtype=float,
    )
    candidates = generate_soft_active_set_candidates(
        elite_samples,
        A_aug,
        b_aug,
        small_lp.n,
        tau=1e-2,
        max_candidates_per_sample=12,
        max_total_candidates=12,
    )
    assert candidates
    assert all(len(candidate.active_indices) == small_lp.n for candidate in candidates)
    assert len({candidate.active_indices for candidate in candidates}) == len(candidates)
