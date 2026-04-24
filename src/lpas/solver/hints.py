from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.core.active_set import rank_active_constraints
from lpas.experiments.metrics import active_set_jaccard, top_k_active_set_overlap
from lpas.solver.result import ArchiveEntry, WarmStartHint
from lpas.solver.warm_start import reconstruct_from_active_set
from lpas.utils.config import WarmStartConfig


@dataclass(frozen=True)
class ActiveSetHintEvaluation:
    selected_active_mask: np.ndarray
    support_ranking: np.ndarray
    support_values: np.ndarray
    top_k: int
    hint_active_set_jaccard: float
    constraints_in_top_k_support: float
    reconstruction: WarmStartHint


def _normalize_weights(scores: np.ndarray | None, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=float)
    if scores is None:
        return np.ones(count, dtype=float)
    weights = np.asarray(scores, dtype=float)
    if weights.shape != (count,):
        raise ValueError("scores must be a one-dimensional array aligned with the masks")
    finite = np.where(np.isfinite(weights), weights, np.nan)
    if np.isnan(finite).all():
        return np.ones(count, dtype=float)
    shifted = finite - np.nanmin(finite)
    shifted = np.nan_to_num(shifted, nan=0.0)
    if float(np.sum(shifted)) <= 0.0:
        return np.ones(count, dtype=float)
    return shifted + 1e-9


def archive_primal_masks(archive: list[ArchiveEntry]) -> np.ndarray:
    if not archive:
        return np.empty((0, 0), dtype=bool)
    return np.asarray([entry.primal_active_mask for entry in archive], dtype=bool)


def archive_scores(archive: list[ArchiveEntry]) -> np.ndarray:
    if not archive:
        return np.empty(0, dtype=float)
    return np.asarray([entry.score for entry in archive], dtype=float)


def select_best_sampled_active_set(primal_masks: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    masks = np.asarray(primal_masks, dtype=bool)
    if masks.size == 0:
        return np.empty(0, dtype=bool)
    if masks.ndim != 2:
        raise ValueError("primal_masks must be a two-dimensional boolean array")
    if scores is None:
        return masks[0].copy()
    score_array = np.asarray(scores, dtype=float)
    if score_array.shape != (masks.shape[0],):
        raise ValueError("scores must align with the number of masks")
    return masks[int(np.nanargmax(score_array))].copy()


def constraint_support(primal_masks: np.ndarray, scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(primal_masks, dtype=bool)
    if masks.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    if masks.ndim != 2:
        raise ValueError("primal_masks must be a two-dimensional boolean array")
    weights = _normalize_weights(scores, masks.shape[0])
    total = float(np.sum(weights))
    support = np.zeros(masks.shape[1], dtype=float) if total <= 0.0 else (weights[:, None] * masks.astype(float)).sum(axis=0) / total
    ranking = rank_active_constraints(masks, weights=weights)
    return support, ranking


def evaluate_active_set_hint(
    problem,
    primal_masks: np.ndarray,
    scores: np.ndarray | None,
    reference_active_mask: np.ndarray,
    *,
    top_k: int | None = None,
    warm_start_config: WarmStartConfig | None = None,
) -> ActiveSetHintEvaluation:
    masks = np.asarray(primal_masks, dtype=bool)
    reference_mask = np.asarray(reference_active_mask, dtype=bool)
    if masks.size == 0:
        selected_mask = np.zeros(problem.m, dtype=bool)
        support_values = np.zeros(problem.m, dtype=float)
        ranking = np.arange(problem.m, dtype=int)
    else:
        if masks.ndim == 1:
            masks = masks[None, :]
        selected_mask = select_best_sampled_active_set(masks, scores)
        support_values, ranking = constraint_support(masks, scores)
    resolved_top_k = min(problem.m, max(problem.n, int(top_k or problem.n)))
    reconstruction = reconstruct_from_active_set(problem, selected_mask, config=warm_start_config)
    return ActiveSetHintEvaluation(
        selected_active_mask=selected_mask,
        support_ranking=ranking,
        support_values=support_values,
        top_k=resolved_top_k,
        hint_active_set_jaccard=active_set_jaccard(selected_mask, reference_mask),
        constraints_in_top_k_support=top_k_active_set_overlap(ranking, reference_mask, resolved_top_k),
        reconstruction=reconstruction,
    )


def evaluate_archive_hint(
    problem,
    archive: list[ArchiveEntry],
    reference_active_mask: np.ndarray,
    *,
    top_k: int | None = None,
    warm_start_config: WarmStartConfig | None = None,
) -> ActiveSetHintEvaluation:
    return evaluate_active_set_hint(
        problem,
        archive_primal_masks(archive),
        archive_scores(archive),
        reference_active_mask,
        top_k=top_k,
        warm_start_config=warm_start_config,
    )
