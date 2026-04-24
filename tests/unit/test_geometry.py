from __future__ import annotations

import numpy as np

from lpas.geometry.active_set_similarity import combined_active_set_similarity, jaccard_similarity
from lpas.geometry.clustering import compute_cluster_support
from lpas.geometry.density_reward import compute_geometry_support


def test_jaccard_similarity_cases() -> None:
    a = np.array([True, False, True])
    b = np.array([True, False, True])
    c = np.array([False, True, False])
    d = np.array([True, True, False])
    assert jaccard_similarity(a, b) == 1.0
    assert jaccard_similarity(a, c) == 0.0
    assert jaccard_similarity(a, d) == 1.0 / 3.0
    assert jaccard_similarity(np.array([False, False]), np.array([False, False])) == 1.0


def test_combined_similarity_is_symmetric() -> None:
    value_ab = combined_active_set_similarity(
        np.array([True, False]),
        np.array([False, True]),
        np.array([True, True]),
        np.array([False, True]),
        beta=0.25,
    )
    value_ba = combined_active_set_similarity(
        np.array([True, True]),
        np.array([False, True]),
        np.array([True, False]),
        np.array([False, True]),
        beta=0.25,
    )
    assert value_ab == value_ba


def test_kernel_density_reward_highest_at_elite_point() -> None:
    X = np.array([[1.0, 1.0], [3.0, 3.0]])
    Y = np.array([[0.5, 0.5], [1.0, 1.0]])
    elite_X = np.array([[1.0, 1.0]])
    elite_Y = np.array([[0.5, 0.5]])
    reward = compute_geometry_support(X, Y, elite_X, elite_Y, sigma=1.0)
    assert reward[0] > reward[1]
    assert reward[0] == 1.0


def test_active_set_cluster_support_frequency_correct() -> None:
    primal = np.array([[True, False], [True, False], [False, True]])
    dual = np.array([[False], [False], [True]])
    support = compute_cluster_support(primal, dual, primal, dual)
    assert support[0] == support[1]
    assert support[0] > support[2]
