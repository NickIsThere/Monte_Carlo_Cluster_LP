from __future__ import annotations

import numpy as np

from lpas.geometry.kernels import gaussian_rbf_from_squared_distance, pairwise_weighted_squared_distance


def compute_geometry_support(
    X: np.ndarray,
    Y: np.ndarray,
    elite_X: np.ndarray | None,
    elite_Y: np.ndarray | None,
    *,
    sigma: float = 1.0,
    dual_weight: float = 1.0,
    mode: str = "max",
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if elite_X is None or elite_Y is None:
        return np.zeros(X.shape[0], dtype=float)
    elite_X = np.asarray(elite_X, dtype=float)
    elite_Y = np.asarray(elite_Y, dtype=float)
    if elite_X.size == 0 or elite_Y.size == 0:
        return np.zeros(X.shape[0], dtype=float)
    squared = pairwise_weighted_squared_distance(X, Y, elite_X, elite_Y, dual_weight=dual_weight)
    kernel = gaussian_rbf_from_squared_distance(squared, sigma)
    if mode == "max":
        return np.max(kernel, axis=1)
    if mode == "mean":
        return np.mean(kernel, axis=1)
    raise ValueError("mode must be 'max' or 'mean'")
