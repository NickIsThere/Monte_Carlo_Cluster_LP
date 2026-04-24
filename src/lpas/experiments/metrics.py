from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from lpas.geometry.active_set_similarity import jaccard_similarity


def relative_objective_error(approximate: float | None, reference: float | None) -> float | None:
    if approximate is None or reference is None:
        return None
    return float(abs(approximate - reference) / max(abs(reference), 1e-12))


def objective_gap_to_reference(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(reference - candidate)


def active_set_jaccard(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(jaccard_similarity(predicted, truth))


def exact_active_set_match(predicted: np.ndarray, truth: np.ndarray) -> bool:
    predicted_mask = np.asarray(predicted, dtype=bool)
    truth_mask = np.asarray(truth, dtype=bool)
    if predicted_mask.shape != truth_mask.shape:
        raise ValueError("mask shapes must match")
    return bool(np.array_equal(predicted_mask, truth_mask))


def active_set_precision_recall(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = np.sum(predicted & truth)
    fp = np.sum(predicted & ~truth)
    fn = np.sum(~predicted & truth)
    precision = 1.0 if tp == 0 and fp == 0 else float(tp / max(tp + fp, 1))
    recall = 1.0 if tp == 0 and fn == 0 else float(tp / max(tp + fn, 1))
    return precision, recall


def top_k_active_set_overlap(ranked_indices: np.ndarray, truth: np.ndarray, k: int) -> float:
    ranking = np.asarray(ranked_indices, dtype=int)
    truth_mask = np.asarray(truth, dtype=bool)
    if truth_mask.ndim != 1:
        raise ValueError("truth must be a one-dimensional mask")
    truth_indices = np.flatnonzero(truth_mask)
    if truth_indices.size == 0:
        return 1.0
    if ranking.ndim != 1:
        raise ValueError("ranked_indices must be one-dimensional")
    top = set(int(index) for index in ranking[: max(int(k), 0)])
    covered = sum(1 for index in truth_indices if int(index) in top)
    return float(covered / truth_indices.size)


def first_threshold_crossing(values: Iterable[float | None], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value is not None and value >= threshold:
            return index
    return None


def safe_mean(values: Iterable[float | None]) -> float:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return math.nan
    return float(np.mean(numeric))


def summarize_history_gaps(history) -> list[float | None]:
    return [entry.best_certified_gap for entry in history]
