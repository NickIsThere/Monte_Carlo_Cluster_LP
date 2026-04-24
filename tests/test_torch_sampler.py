from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lpas.gpu.torch_sampler import select_elites_torch, update_distribution_torch


def test_topk_returns_correct_number_of_elites() -> None:
    X = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    Y = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    scores = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0], dtype=torch.float64)
    elite_X, elite_Y, elite_scores, elite_indices = select_elites_torch(X, Y, scores, 2)
    assert elite_X.shape[0] == 2
    assert elite_Y.shape[0] == 2
    assert elite_scores.shape[0] == 2
    assert elite_indices.shape[0] == 2


def test_elites_correspond_to_highest_scores() -> None:
    X = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    Y = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    scores = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0], dtype=torch.float64)
    _, _, elite_scores, elite_indices = select_elites_torch(X, Y, scores, 2)
    assert elite_scores.tolist() == [5.0, 4.0]
    assert elite_indices.tolist() == [1, 3]


def test_distribution_update_moves_mean_toward_elites() -> None:
    mu_x = torch.zeros(2, dtype=torch.float64)
    sigma_x = torch.ones(2, dtype=torch.float64)
    mu_y = torch.zeros(2, dtype=torch.float64)
    sigma_y = torch.ones(2, dtype=torch.float64)
    elite_X = torch.tensor([[2.0, 2.0], [4.0, 4.0]], dtype=torch.float64)
    elite_Y = torch.tensor([[1.0, 1.0], [3.0, 3.0]], dtype=torch.float64)
    new_mu_x, _, new_mu_y, _ = update_distribution_torch(
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        elite_X,
        elite_Y,
        alpha=0.5,
        min_sigma=1e-3,
        max_sigma=10.0,
    )
    assert torch.all(new_mu_x > mu_x)
    assert torch.all(new_mu_y > mu_y)


def test_sigma_is_clamped_by_min_sigma() -> None:
    mu_x = torch.zeros(2, dtype=torch.float64)
    sigma_x = torch.ones(2, dtype=torch.float64)
    mu_y = torch.zeros(2, dtype=torch.float64)
    sigma_y = torch.ones(2, dtype=torch.float64)
    elite_X = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    elite_Y = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float64)
    _, new_sigma_x, _, new_sigma_y = update_distribution_torch(
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        elite_X,
        elite_Y,
        alpha=1.0,
        min_sigma=0.25,
        max_sigma=10.0,
    )
    assert torch.all(new_sigma_x >= 0.25)
    assert torch.all(new_sigma_y >= 0.25)


def test_no_nans_after_update() -> None:
    mu_x = torch.zeros(2, dtype=torch.float64)
    sigma_x = torch.ones(2, dtype=torch.float64)
    mu_y = torch.zeros(2, dtype=torch.float64)
    sigma_y = torch.ones(2, dtype=torch.float64)
    elite_X = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    elite_Y = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float64)
    updated = update_distribution_torch(
        mu_x,
        sigma_x,
        mu_y,
        sigma_y,
        elite_X,
        elite_Y,
        alpha=0.3,
        min_sigma=0.25,
        max_sigma=10.0,
    )
    for tensor in updated:
        assert torch.isfinite(tensor).all()
