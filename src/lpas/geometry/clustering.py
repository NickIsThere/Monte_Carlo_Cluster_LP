from __future__ import annotations

from collections import Counter

import numpy as np


def active_pattern_key(primal_mask: np.ndarray, dual_mask: np.ndarray) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    return tuple(bool(v) for v in primal_mask), tuple(bool(v) for v in dual_mask)


def pattern_frequencies(primal_masks: np.ndarray, dual_masks: np.ndarray) -> Counter:
    counter: Counter = Counter()
    if primal_masks.size == 0 or dual_masks.size == 0:
        return counter
    for primal_mask, dual_mask in zip(primal_masks, dual_masks, strict=True):
        counter[active_pattern_key(primal_mask, dual_mask)] += 1
    return counter


def compute_cluster_support(
    primal_masks: np.ndarray,
    dual_masks: np.ndarray,
    elite_primal_masks: np.ndarray | None,
    elite_dual_masks: np.ndarray | None,
    *,
    smoothing: float = 0.0,
) -> np.ndarray:
    primal_masks = np.asarray(primal_masks, dtype=bool)
    dual_masks = np.asarray(dual_masks, dtype=bool)
    if elite_primal_masks is None or elite_dual_masks is None:
        return np.zeros(primal_masks.shape[0], dtype=float)
    elite_primal_masks = np.asarray(elite_primal_masks, dtype=bool)
    elite_dual_masks = np.asarray(elite_dual_masks, dtype=bool)
    counter = pattern_frequencies(elite_primal_masks, elite_dual_masks)
    total = max(sum(counter.values()), 0)
    if total == 0:
        return np.zeros(primal_masks.shape[0], dtype=float)
    denominator = total + smoothing * max(len(counter), 1)
    support = []
    for primal_mask, dual_mask in zip(primal_masks, dual_masks, strict=True):
        frequency = counter.get(active_pattern_key(primal_mask, dual_mask), 0)
        support.append((frequency + smoothing) / denominator)
    return np.asarray(support, dtype=float)
