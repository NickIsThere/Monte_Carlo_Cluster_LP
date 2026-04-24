from __future__ import annotations

import numpy as np


def relative_objective_error(approximate: float | None, reference: float | None) -> float | None:
    if approximate is None or reference is None:
        return None
    return float(abs(approximate - reference) / max(abs(reference), 1e-12))


def active_set_precision_recall(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = np.sum(predicted & truth)
    fp = np.sum(predicted & ~truth)
    fn = np.sum(~predicted & truth)
    precision = 1.0 if tp == 0 and fp == 0 else float(tp / max(tp + fp, 1))
    recall = 1.0 if tp == 0 and fn == 0 else float(tp / max(tp + fn, 1))
    return precision, recall


def summarize_history_gaps(history) -> list[float | None]:
    return [entry.best_certified_gap for entry in history]
