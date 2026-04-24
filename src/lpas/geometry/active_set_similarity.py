from __future__ import annotations

import numpy as np


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes must match")
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    intersection = np.sum(a & b)
    return float(intersection / union)


def combined_active_set_similarity(
    primal_a: np.ndarray,
    dual_a: np.ndarray,
    primal_b: np.ndarray,
    dual_b: np.ndarray,
    *,
    beta: float = 0.5,
) -> float:
    primal_sim = jaccard_similarity(primal_a, primal_b)
    dual_sim = jaccard_similarity(dual_a, dual_b)
    return float(beta * primal_sim + (1.0 - beta) * dual_sim)
