from __future__ import annotations

import math
from typing import Any

from lpas.backends.torch_device import _require_torch


def make_torch_generator(device: Any, seed: int | None) -> Any:
    torch = _require_torch()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def sample_nonnegative_normal_torch(
    mu_x: Any,
    sigma_x: Any,
    mu_y: Any,
    sigma_y: Any,
    sample_size: int,
    *,
    generator: Any,
) -> tuple[Any, Any]:
    torch = _require_torch()
    X = torch.relu(
        mu_x.unsqueeze(0) + sigma_x.unsqueeze(0) * torch.randn(sample_size, mu_x.shape[0], device=mu_x.device, dtype=mu_x.dtype, generator=generator)
    )
    Y = torch.relu(
        mu_y.unsqueeze(0) + sigma_y.unsqueeze(0) * torch.randn(sample_size, mu_y.shape[0], device=mu_y.device, dtype=mu_y.dtype, generator=generator)
    )
    return X, Y


def select_elites_torch(X: Any, Y: Any, scores: Any, elite_fraction: float | int) -> tuple[Any, Any, Any, Any]:
    if isinstance(elite_fraction, int):
        elite_count = elite_fraction
    else:
        elite_count = max(1, math.ceil(scores.shape[0] * elite_fraction))
    elite_count = min(elite_count, scores.shape[0])
    elite_scores, elite_indices = scores.topk(k=elite_count, largest=True, sorted=True)
    return X[elite_indices], Y[elite_indices], elite_scores, elite_indices


def update_distribution_torch(
    mu_x: Any,
    sigma_x: Any,
    mu_y: Any,
    sigma_y: Any,
    elite_X: Any,
    elite_Y: Any,
    *,
    alpha: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[Any, Any, Any, Any]:
    new_mu_x = elite_X.mean(dim=0)
    new_sigma_x = elite_X.std(dim=0, unbiased=False).clamp_min(min_sigma).clamp_max(max_sigma)
    new_mu_y = elite_Y.mean(dim=0)
    new_sigma_y = elite_Y.std(dim=0, unbiased=False).clamp_min(min_sigma).clamp_max(max_sigma)
    mu_x = alpha * new_mu_x + (1.0 - alpha) * mu_x
    sigma_x = alpha * new_sigma_x + (1.0 - alpha) * sigma_x
    mu_y = alpha * new_mu_y + (1.0 - alpha) * mu_y
    sigma_y = alpha * new_sigma_y + (1.0 - alpha) * sigma_y
    return mu_x, sigma_x, mu_y, sigma_y
