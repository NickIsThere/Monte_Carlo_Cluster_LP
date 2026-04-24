from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.validation import ensure_batch


@dataclass(frozen=True)
class ActiveSetBatch:
    primal_active_mask: np.ndarray
    dual_active_mask: np.ndarray


def primal_active_mask(problem: LPProblem, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    slack = problem.b - problem.A @ np.asarray(x, dtype=float)
    return slack <= epsilon


def dual_active_mask(problem: LPProblem, y: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    slack = problem.A.T @ np.asarray(y, dtype=float) - problem.c
    return slack <= epsilon


def extract_active_sets(problem: LPProblem, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-6) -> ActiveSetBatch:
    X_batch = ensure_batch(X, expected_dim=problem.n, name="X")
    Y_batch = ensure_batch(Y, expected_dim=problem.m, name="Y")
    primal_slack = problem.b - X_batch @ problem.A.T
    dual_slack = Y_batch @ problem.A - problem.c
    return ActiveSetBatch(primal_active_mask=primal_slack <= epsilon, dual_active_mask=dual_slack <= epsilon)


def rank_active_constraints(primal_masks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if primal_masks.size == 0:
        return np.array([], dtype=int)
    mask_array = np.asarray(primal_masks, dtype=float)
    if weights is None:
        support = np.mean(mask_array, axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != mask_array.shape[0]:
            raise ValueError("weights must match the number of samples")
        total = np.sum(w)
        if total <= 0.0:
            support = np.zeros(mask_array.shape[1], dtype=float)
        else:
            support = (w[:, None] * mask_array).sum(axis=0) / total
    return np.argsort(-support)
