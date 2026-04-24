from __future__ import annotations

import numpy as np

from lpas.core.primal_dual import PrimalDualMetrics
from lpas.utils.config import ScoringConfig


def effective_gap(metrics: PrimalDualMetrics) -> np.ndarray:
    raw_gap = np.asarray(metrics.raw_gap, dtype=float)
    primal_violation = np.asarray(metrics.primal_violation_norm, dtype=float)
    dual_violation = np.asarray(metrics.dual_violation_norm, dtype=float)
    primal_feasible = np.asarray(metrics.primal_feasible, dtype=bool)
    dual_feasible = np.asarray(metrics.dual_feasible, dtype=bool)
    feasible_gap = primal_feasible & dual_feasible & (raw_gap >= 0.0)
    penalty_gap = np.abs(raw_gap) + primal_violation + dual_violation
    return np.where(feasible_gap, raw_gap, penalty_gap)


def _safe_values(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    if higher_is_better:
        fill_value = np.finfo(float).min
    else:
        fill_value = np.finfo(float).max
    array[~np.isfinite(array)] = fill_value
    return array


def rank_normalized(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    array = _safe_values(values, higher_is_better)
    if array.size == 1:
        return np.ones(1, dtype=float)
    if higher_is_better:
        order = np.argsort(array)
    else:
        order = np.argsort(-array)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=array.size)
    return ranks


def score_candidates(
    metrics: PrimalDualMetrics,
    *,
    geometry_support: np.ndarray | None = None,
    cluster_support: np.ndarray | None = None,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    cfg = config or ScoringConfig()
    n = np.asarray(metrics.primal_objective).shape[0]
    geo = np.zeros(n, dtype=float) if geometry_support is None else np.asarray(geometry_support, dtype=float)
    cluster = np.zeros(n, dtype=float) if cluster_support is None else np.asarray(cluster_support, dtype=float)
    if geo.shape != (n,) or cluster.shape != (n,):
        raise ValueError("geometry_support and cluster_support must be one-dimensional batch arrays")
    score = np.zeros(n, dtype=float)
    score += cfg.w_primal_obj * rank_normalized(np.asarray(metrics.primal_objective, dtype=float), higher_is_better=True)
    score += cfg.w_dual_obj * rank_normalized(np.asarray(metrics.dual_objective, dtype=float), higher_is_better=False)
    score += cfg.w_gap * rank_normalized(effective_gap(metrics), higher_is_better=False)
    score += cfg.w_pviol * rank_normalized(np.asarray(metrics.primal_violation_norm, dtype=float), higher_is_better=False)
    score += cfg.w_dviol * rank_normalized(np.asarray(metrics.dual_violation_norm, dtype=float), higher_is_better=False)
    score += cfg.w_comp * rank_normalized(np.asarray(metrics.complementarity_error, dtype=float), higher_is_better=False)
    score += cfg.w_geo * rank_normalized(geo, higher_is_better=True)
    score += cfg.w_active * rank_normalized(cluster, higher_is_better=True)
    return np.nan_to_num(score, nan=-1e12, posinf=1e12, neginf=-1e12)
