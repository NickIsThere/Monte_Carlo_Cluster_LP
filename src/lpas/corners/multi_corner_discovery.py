from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.geometry.active_set_similarity import jaccard_similarity
from lpas.solver.result import ArchiveEntry, ScipySolveResult
from lpas.solver.vertex_polishing import (
    augment_primal_constraints,
    reconstruct_vertex_from_active_set,
)
from lpas.utils.config import FeasibilityConfig


def active_set_jaccard(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(jaccard_similarity(predicted, truth))


@dataclass(frozen=True)
class CornerCandidate:
    x: np.ndarray
    objective: float
    active_set: np.ndarray
    active_rank: int
    primal_violation: float
    reconstruction_residual: float
    source: str
    frequency: int
    mean_sample_distance: float
    is_feasible: bool
    max_constraint_violation: float = 0.0
    sum_constraint_violation: float = 0.0
    active_set_size: int = 0
    rank_of_active_matrix: int | None = None
    source_candidate_id: str = ""
    polishing_status: str = "NUMERIC_FAILURE"


@dataclass(frozen=True)
class MultiCornerDiscoveryResult:
    candidates: list[CornerCandidate]
    unique_vertices: list[CornerCandidate]
    best_vertex: CornerCandidate | None
    best_feasible_vertex: CornerCandidate | None
    best_infeasible_vertex: CornerCandidate | None
    num_raw_elites: int
    num_reconstructed: int
    num_unique_vertices: int
    num_feasible_vertices: int
    num_infeasible_vertices: int
    has_feasible_vertex: bool
    best_objective_any: float | None
    top_feasible_objectives: list[float]
    objective_diversity_feasible: float | None
    active_set_diversity_feasible: float | None
    best_vertex_frequency: int | None
    median_vertex_frequency: float | None
    max_vertex_frequency: int | None
    num_vertices_matching_reference_active_set: int
    num_vertices_containing_true_active_set: int
    optimum_jaccard: float | None
    contains_true_active_set: bool | None


def _augmented_active_mask(problem: LPProblem, active_indices: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(problem.m + problem.n, dtype=bool)
    mask[list(active_indices)] = True
    return mask


def _vertex_tolerance(problem: LPProblem, explicit_tolerance: float | None) -> float:
    if explicit_tolerance is not None:
        return float(explicit_tolerance)
    scale = max(1.0, float(np.max(np.abs(problem.b), initial=0.0)))
    return 1e-7 * scale


def _composite_activity_scores(
    slacks: np.ndarray,
    *,
    active_frequency: np.ndarray | None,
    dual_support: np.ndarray | None,
) -> np.ndarray:
    magnitude = np.maximum(np.abs(np.asarray(slacks, dtype=float)), 1e-12)
    scores = 1.0 / magnitude
    if active_frequency is not None:
        scores = scores + 2.0 * np.asarray(active_frequency, dtype=float)
    if dual_support is not None:
        dual_bonus = np.zeros_like(scores)
        dual_bonus[: dual_support.shape[0]] = np.asarray(dual_support, dtype=float)
        scores = scores + dual_bonus
    return scores


def _average_soft_frequency(
    problem: LPProblem,
    elite_samples: np.ndarray,
    *,
    tau: float = 1e-3,
) -> np.ndarray:
    if elite_samples.size == 0:
        return np.zeros(problem.m + problem.n, dtype=float)
    A_aug, b_aug = augment_primal_constraints(problem)
    scores = []
    for sample in elite_samples:
        slacks = b_aug - A_aug @ sample
        scores.append(np.exp(-np.abs(slacks) / tau))
    return np.mean(np.asarray(scores, dtype=float), axis=0)


def _dual_positive_support(dual_samples: np.ndarray | None, m_constraints: int) -> np.ndarray | None:
    if dual_samples is None:
        return None
    dual_array = np.asarray(dual_samples, dtype=float)
    if dual_array.size == 0:
        return np.zeros(m_constraints, dtype=float)
    if dual_array.ndim == 1:
        dual_array = dual_array[None, :]
    if dual_array.shape[1] != m_constraints:
        raise ValueError("dual_samples must align with the number of primal constraints")
    return np.mean(dual_array > 1e-8, axis=0, dtype=float)


def hard_active_set_recovery(problem: LPProblem, sample: np.ndarray, *, tolerance: float = 1e-6) -> tuple[int, ...]:
    A_aug, b_aug = augment_primal_constraints(problem)
    slacks = b_aug - A_aug @ np.asarray(sample, dtype=float)
    active = np.flatnonzero(np.abs(slacks) <= tolerance)
    return tuple(int(index) for index in active)


def recover_soft_active_set(
    problem: LPProblem,
    sample: np.ndarray,
    *,
    elite_active_frequency: np.ndarray | None = None,
    dual_support: np.ndarray | None = None,
    max_ranked_constraints: int | None = None,
) -> tuple[int, ...]:
    canonical_problem = problem.to_maximization()
    A_aug, b_aug = augment_primal_constraints(canonical_problem)
    x = np.asarray(sample, dtype=float)
    slacks = b_aug - A_aug @ x
    scores = _composite_activity_scores(
        slacks,
        active_frequency=elite_active_frequency,
        dual_support=dual_support,
    )
    ranked_indices = np.lexsort((np.abs(slacks), -scores))
    scan_limit = len(ranked_indices) if max_ranked_constraints is None else min(int(max_ranked_constraints), len(ranked_indices))
    selected: list[int] = []
    current_rank = 0
    for index in ranked_indices[:scan_limit]:
        candidate_indices = tuple(selected + [int(index)])
        rank = int(np.linalg.matrix_rank(A_aug[list(candidate_indices), :]))
        if rank <= current_rank:
            continue
        selected.append(int(index))
        current_rank = rank
        if current_rank >= canonical_problem.n:
            break
    return tuple(sorted(selected))


def _candidate_from_active_set(
    problem: LPProblem,
    active_indices: tuple[int, ...],
    *,
    source: str,
    sample_distances: np.ndarray,
    feasibility_config: FeasibilityConfig,
) -> CornerCandidate | None:
    source_candidate_id = f"{source}:{active_indices}"
    vertex = reconstruct_vertex_from_active_set(
        problem.to_maximization(),
        active_indices,
        source_candidate_id=source_candidate_id,
        feasibility_config=feasibility_config,
    )
    if vertex is None:
        return None
    active_mask = _augmented_active_mask(problem.to_maximization(), active_indices)
    rank = int(
        vertex.metadata.get(
            "rank",
            np.linalg.matrix_rank(augment_primal_constraints(problem.to_maximization())[0][list(active_indices)]),
        )
    )
    return CornerCandidate(
        x=np.asarray(vertex.x, dtype=float),
        objective=float(vertex.objective),
        active_set=active_mask,
        active_rank=rank,
        primal_violation=float(vertex.primal_violation),
        reconstruction_residual=float(vertex.reconstruction_residual),
        source=source,
        frequency=1,
        mean_sample_distance=float(np.mean(sample_distances)) if sample_distances.size else 0.0,
        is_feasible=bool(vertex.feasible),
        max_constraint_violation=float(vertex.max_constraint_violation),
        sum_constraint_violation=float(vertex.sum_constraint_violation),
        active_set_size=int(np.sum(active_mask)),
        rank_of_active_matrix=rank,
        source_candidate_id=str(vertex.metadata.get("source_candidate_id", source_candidate_id)),
        polishing_status=str(vertex.polishing_status),
    )


def _merge_candidates(left: CornerCandidate, right: CornerCandidate) -> CornerCandidate:
    total_frequency = left.frequency + right.frequency
    weighted_distance = (
        left.mean_sample_distance * left.frequency + right.mean_sample_distance * right.frequency
    ) / max(total_frequency, 1)
    chosen = right if _candidate_sort_key(right) > _candidate_sort_key(left) else left
    return replace(
        chosen,
        frequency=total_frequency,
        mean_sample_distance=float(weighted_distance),
    )


def _candidate_sort_key(candidate: CornerCandidate) -> tuple[float, float, float, float, float]:
    rank = candidate.rank_of_active_matrix if candidate.rank_of_active_matrix is not None else candidate.active_rank
    return (
        1.0 if candidate.is_feasible else 0.0,
        candidate.objective,
        -candidate.max_constraint_violation,
        -candidate.reconstruction_residual,
        float(rank),
    )


def rank_corner_candidates(candidates: list[CornerCandidate]) -> list[CornerCandidate]:
    return sorted(candidates, key=_candidate_sort_key, reverse=True)


def deduplicate_vertices(
    candidates: list[CornerCandidate],
    *,
    vertex_tolerance: float,
) -> list[CornerCandidate]:
    unique: list[CornerCandidate] = []
    for candidate in rank_corner_candidates(candidates):
        merged = False
        for index, existing in enumerate(unique):
            distance = float(np.linalg.norm(candidate.x - existing.x, ord=np.inf))
            if distance <= vertex_tolerance:
                unique[index] = _merge_candidates(existing, candidate)
                merged = True
                break
        if not merged:
            unique.append(candidate)
    return rank_corner_candidates(unique)


def _active_set_containment(predicted: np.ndarray, truth: np.ndarray) -> float:
    truth_mask = np.asarray(truth, dtype=bool)
    if not np.any(truth_mask):
        return 1.0
    predicted_mask = np.asarray(predicted, dtype=bool)
    overlap = float(np.sum(predicted_mask & truth_mask))
    return overlap / float(np.sum(truth_mask))


def _objective_diversity(candidates: list[CornerCandidate]) -> float | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return 0.0
    values = np.asarray([candidate.objective for candidate in candidates], dtype=float)
    return float(np.std(values))


def _active_set_diversity(candidates: list[CornerCandidate]) -> float | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return 0.0
    distances: list[float] = []
    for left_index, left in enumerate(candidates[:-1]):
        for right in candidates[left_index + 1 :]:
            distances.append(1.0 - active_set_jaccard(left.active_set, right.active_set))
    return float(np.mean(distances)) if distances else 0.0


def discover_multiple_corners(
    problem: LPProblem,
    elite_samples: np.ndarray,
    *,
    dual_samples: np.ndarray | None = None,
    reference_result: ScipySolveResult | None = None,
    elite_scores: np.ndarray | None = None,
    vertex_tolerance: float | None = None,
    max_ranked_constraints: int | None = None,
    feasibility_config: FeasibilityConfig | None = None,
) -> MultiCornerDiscoveryResult:
    canonical_problem = problem.to_maximization()
    samples = np.asarray(elite_samples, dtype=float)
    if samples.size == 0:
        return MultiCornerDiscoveryResult(
            candidates=[],
            unique_vertices=[],
            best_vertex=None,
            best_feasible_vertex=None,
            best_infeasible_vertex=None,
            num_raw_elites=0,
            num_reconstructed=0,
            num_unique_vertices=0,
            num_feasible_vertices=0,
            num_infeasible_vertices=0,
            has_feasible_vertex=False,
            best_objective_any=None,
            top_feasible_objectives=[],
            objective_diversity_feasible=None,
            active_set_diversity_feasible=None,
            best_vertex_frequency=None,
            median_vertex_frequency=None,
            max_vertex_frequency=None,
            num_vertices_matching_reference_active_set=0,
            num_vertices_containing_true_active_set=0,
            optimum_jaccard=None,
            contains_true_active_set=None,
        )
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[1] != canonical_problem.n:
        raise ValueError("elite_samples must align with the LP dimension")

    resolved_feasibility = feasibility_config or FeasibilityConfig()
    del elite_scores
    active_frequency = _average_soft_frequency(canonical_problem, samples)
    dual_support = _dual_positive_support(dual_samples, canonical_problem.m)
    reconstructed: list[CornerCandidate] = []
    active_set_counts: dict[tuple[int, ...], list[np.ndarray]] = {}
    for sample in samples:
        active_indices = recover_soft_active_set(
            canonical_problem,
            sample,
            elite_active_frequency=active_frequency,
            dual_support=dual_support,
            max_ranked_constraints=max_ranked_constraints,
        )
        if len(active_indices) < canonical_problem.n:
            continue
        active_set_counts.setdefault(active_indices, []).append(np.asarray(sample, dtype=float))

    for active_indices, associated_samples in active_set_counts.items():
        sample_matrix = np.asarray(associated_samples, dtype=float)
        candidate = _candidate_from_active_set(
            canonical_problem,
            active_indices,
            source="soft_active_set",
            sample_distances=np.linalg.norm(sample_matrix - np.mean(sample_matrix, axis=0), axis=1),
            feasibility_config=resolved_feasibility,
        )
        if candidate is None:
            continue
        reconstructed.append(replace(candidate, frequency=sample_matrix.shape[0]))

    tolerance = _vertex_tolerance(canonical_problem, vertex_tolerance)
    unique_vertices = deduplicate_vertices(reconstructed, vertex_tolerance=tolerance)
    feasible_vertices = [candidate for candidate in unique_vertices if candidate.is_feasible]
    infeasible_vertices = [candidate for candidate in unique_vertices if not candidate.is_feasible]
    best_feasible_vertex = None if not feasible_vertices else feasible_vertices[0]
    best_infeasible_vertex = None if not infeasible_vertices else infeasible_vertices[0]
    best_vertex = best_feasible_vertex
    best_any_vertex = None if not unique_vertices else unique_vertices[0]

    optimum_jaccard = None
    contains_true_active_set = None
    num_vertices_matching_reference_active_set = 0
    num_vertices_containing_true_active_set = 0
    if reference_result is not None and reference_result.augmented_primal_active_mask is not None:
        truth_mask = np.asarray(reference_result.augmented_primal_active_mask, dtype=bool)
        if best_feasible_vertex is not None:
            optimum_jaccard = active_set_jaccard(best_feasible_vertex.active_set, truth_mask)
            contains_true_active_set = bool(_active_set_containment(best_feasible_vertex.active_set, truth_mask) >= 1.0 - 1e-12)
        feasible_against_truth = feasible_vertices
        num_vertices_matching_reference_active_set = sum(
            int(np.array_equal(candidate.active_set, truth_mask))
            for candidate in feasible_against_truth
        )
        num_vertices_containing_true_active_set = sum(
            int(_active_set_containment(candidate.active_set, truth_mask) >= 1.0 - 1e-12)
            for candidate in feasible_against_truth
        )

    frequencies = [candidate.frequency for candidate in unique_vertices]
    top_feasible_objectives = [float(candidate.objective) for candidate in feasible_vertices[:10]]
    best_frequency_source = best_feasible_vertex if best_feasible_vertex is not None else best_any_vertex
    return MultiCornerDiscoveryResult(
        candidates=rank_corner_candidates(reconstructed),
        unique_vertices=unique_vertices,
        best_vertex=best_vertex,
        best_feasible_vertex=best_feasible_vertex,
        best_infeasible_vertex=best_infeasible_vertex,
        num_raw_elites=int(samples.shape[0]),
        num_reconstructed=len(reconstructed),
        num_unique_vertices=len(unique_vertices),
        num_feasible_vertices=len(feasible_vertices),
        num_infeasible_vertices=len(infeasible_vertices),
        has_feasible_vertex=bool(best_feasible_vertex is not None),
        best_objective_any=None if best_any_vertex is None else float(best_any_vertex.objective),
        top_feasible_objectives=top_feasible_objectives,
        objective_diversity_feasible=_objective_diversity(feasible_vertices),
        active_set_diversity_feasible=_active_set_diversity(feasible_vertices),
        best_vertex_frequency=None if best_frequency_source is None else int(best_frequency_source.frequency),
        median_vertex_frequency=None if not frequencies else float(np.median(np.asarray(frequencies, dtype=float))),
        max_vertex_frequency=None if not frequencies else int(max(frequencies)),
        num_vertices_matching_reference_active_set=num_vertices_matching_reference_active_set,
        num_vertices_containing_true_active_set=num_vertices_containing_true_active_set,
        optimum_jaccard=optimum_jaccard,
        contains_true_active_set=contains_true_active_set,
    )


def archive_elite_samples(archive: list[ArchiveEntry]) -> np.ndarray:
    if not archive:
        return np.empty((0, 0), dtype=float)
    return np.asarray([entry.x for entry in archive], dtype=float)
