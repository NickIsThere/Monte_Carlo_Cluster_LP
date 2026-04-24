from __future__ import annotations

import numpy as np


def gaussian_rbf_from_squared_distance(squared_distance: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return np.exp(-np.asarray(squared_distance, dtype=float) / (sigma * sigma))


def pairwise_weighted_squared_distance(
    X: np.ndarray,
    Y: np.ndarray,
    elite_X: np.ndarray,
    elite_Y: np.ndarray,
    *,
    dual_weight: float = 1.0,
) -> np.ndarray:
    dx = X[:, None, :] - elite_X[None, :, :]
    dy = Y[:, None, :] - elite_Y[None, :, :]
    return np.sum(dx * dx, axis=2) + dual_weight * np.sum(dy * dy, axis=2)
