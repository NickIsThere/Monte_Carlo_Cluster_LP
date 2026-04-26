from __future__ import annotations

import math
import time
from itertools import combinations, islice

import numpy as np

from lpas.core.feasibility import primal_feasibility_report
from lpas.core.lp_problem import LPProblem
from lpas.solver.result import (
    ArchiveEntry,
    PolishedVertex,
    SoftActiveSetCandidate,
    VertexPolishingResult,
    WarmStartHint,
)
from lpas.utils.config import FeasibilityConfig, VertexPolishingConfig


def augment_primal_constraints(problem: LPProblem) -> tuple[np.ndarray, np.ndarray]:
    """Return Ax <= b and x >= 0 as one augmented inequality system."""
    identity = np.eye(problem.n, dtype=float)
    A_aug = np.vstack([problem.A, -identity])
    b_aug = np.concatenate([problem.b, np.zeros(problem.n, dtype=float)])
    return A_aug, b_aug


def augmented_primal_slacks(problem: LPProblem, x: np.ndarray) -> np.ndarray:
    A_aug, b_aug = augment_primal_constraints(problem)
    return b_aug - A_aug @ np.asarray(x, dtype=float)


def compute_soft_activity_scores(slacks: np.ndarray, tau: float, method: str = "rbf") -> np.ndarray:
    """Convert slack magnitudes into soft activity scores in [0, 1]."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    slack_array = np.asarray(slacks, dtype=float)
    slack_magnitude = np.abs(slack_array)
    scaled = slack_magnitude / tau
    if method == "rbf":
        scores = np.exp(-(scaled**2))
    elif method == "reciprocal":
        scores = 1.0 / (1.0 + scaled)
    else:
        raise ValueError(f"unsupported soft activity method: {method}")
    return np.clip(scores, 0.0, 1.0)


def _resolve_sample_batch(samples: np.ndarray, n_variables: int) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return np.empty((0, n_variables), dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != n_variables:
        raise ValueError("elite_samples must be a two-dimensional array with one row per primal sample")
    return array


def _resolve_vector(values: np.ndarray | None, count: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (count,):
        raise ValueError(f"{name} must align with the number of samples")
    return array


def _rank_constraints(slacks: np.ndarray, soft_scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.abs(slacks), -soft_scores))


def _candidate_score(
    active_indices: tuple[int, ...],
    soft_scores: np.ndarray,
    sample_primal_violation: float | None,
) -> float:
    score = float(np.mean(soft_scores[list(active_indices)]))
    if sample_primal_violation is not None:
        score -= 1e-6 * max(sample_primal_violation, 0.0)
    return score


def generate_soft_active_set_candidates(
    elite_samples: np.ndarray,
    A_aug: np.ndarray,
    b_aug: np.ndarray,
    n_active: int,
    tau: float,
    *,
    method: str = "rbf",
    max_ranked_constraints: int | None = None,
    max_candidates_per_sample: int = 100,
    max_total_candidates: int = 5000,
    sample_objectives: np.ndarray | None = None,
    sample_primal_violations: np.ndarray | None = None,
) -> list[SoftActiveSetCandidate]:
    """Rank soft active-set guesses from elite primal samples."""
    samples = _resolve_sample_batch(elite_samples, A_aug.shape[1])
    if samples.shape[0] == 0 or n_active <= 0:
        return []
    objectives = _resolve_vector(sample_objectives, samples.shape[0], "sample_objectives")
    violations = _resolve_vector(sample_primal_violations, samples.shape[0], "sample_primal_violations")
    ranked_cap = max_ranked_constraints
    if ranked_cap is None:
        ranked_cap = max(n_active + 4, 2 * n_active)
    ranked_cap = max(n_active, min(int(ranked_cap), A_aug.shape[0]))
    scan_limit = max(max_candidates_per_sample, 1) * 4

    deduplicated: dict[tuple[int, ...], SoftActiveSetCandidate] = {}
    for sample_index, sample in enumerate(samples):
        slacks = b_aug - A_aug @ sample
        soft_scores = compute_soft_activity_scores(slacks, tau=tau, method=method)
        ranked_indices = tuple(int(i) for i in _rank_constraints(slacks, soft_scores)[:ranked_cap])
        sample_violation = None if violations is None else float(violations[sample_index])
        sample_objective = None if objectives is None else float(objectives[sample_index])
        sample_candidates: list[SoftActiveSetCandidate] = []
        for subset in islice(combinations(ranked_indices, n_active), scan_limit):
            active_indices = tuple(sorted(int(i) for i in subset))
            sample_candidates.append(
                SoftActiveSetCandidate(
                    active_indices=active_indices,
                    source_sample_index=sample_index,
                    score=_candidate_score(active_indices, soft_scores, sample_violation),
                    slacks=slacks.copy(),
                    soft_scores=soft_scores.copy(),
                    rank_method=method,
                    sample_objective=sample_objective,
                    sample_primal_violation=sample_violation,
                )
            )
        sample_candidates.sort(key=lambda candidate: (-candidate.score, candidate.active_indices))
        for candidate in sample_candidates[:max_candidates_per_sample]:
            existing = deduplicated.get(candidate.active_indices)
            if existing is None or candidate.score > existing.score:
                deduplicated[candidate.active_indices] = candidate

    candidates = sorted(deduplicated.values(), key=lambda candidate: (-candidate.score, candidate.active_indices))
    return candidates[:max_total_candidates]


def reconstruct_vertex_from_active_set(
    problem: LPProblem,
    active_indices: tuple[int, ...],
    *,
    feasibility_tol: float = 1e-8,
    residual_tol: float = 1e-8,
    source_sample_index: int | None = None,
    A_aug: np.ndarray | None = None,
    b_aug: np.ndarray | None = None,
    source_candidate_id: str | None = None,
    feasibility_config: FeasibilityConfig | None = None,
) -> PolishedVertex | None:
    """Solve the equality system induced by one augmented active set."""
    active_tuple = tuple(sorted(int(index) for index in active_indices))
    if len(active_tuple) != problem.n:
        return None
    resolved_A_aug, resolved_b_aug = augment_primal_constraints(problem) if A_aug is None or b_aug is None else (A_aug, b_aug)
    A_active = resolved_A_aug[list(active_tuple), :]
    b_active = resolved_b_aug[list(active_tuple)]
    rank = int(np.linalg.matrix_rank(A_active))
    if rank < problem.n:
        return None
    try:
        condition_number = float(np.linalg.cond(A_active))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(condition_number):
        return None
    try:
        x = np.linalg.solve(A_active, b_active)
    except np.linalg.LinAlgError:
        return None
    residual = float(np.linalg.norm(A_active @ x - b_active, ord=np.inf))
    feasibility_report = primal_feasibility_report(
        problem,
        x,
        tol=feasibility_tol,
        config=feasibility_config,
    )
    primal_violation = float(feasibility_report.total_violation)
    feasible = bool(residual <= residual_tol and feasibility_report.is_feasible)
    original_active_mask = (problem.b - problem.A @ x) <= feasibility_tol
    nonneg_active_mask = np.asarray(x <= feasibility_tol, dtype=bool)
    return PolishedVertex(
        x=np.asarray(x, dtype=float),
        objective=float(problem.maximization_objective_value(x)),
        active_indices=active_tuple,
        feasible=feasible,
        primal_violation=primal_violation,
        reconstruction_residual=residual,
        condition_number=condition_number,
        source_sample_index=source_sample_index,
        original_active_mask=np.asarray(original_active_mask, dtype=bool),
        nonneg_active_mask=nonneg_active_mask,
        max_constraint_violation=feasibility_report.max_constraint_violation,
        sum_constraint_violation=feasibility_report.total_violation,
        polishing_status="FEASIBLE_VERTEX" if feasible else "INFEASIBLE_VERTEX",
        metadata={
            "rank": rank,
            "source_candidate_id": source_candidate_id,
            "max_constraint_violation": feasibility_report.max_constraint_violation,
            "sum_constraint_violation": feasibility_report.total_violation,
        },
    )


def polished_vertex_to_warm_start_hint(vertex: PolishedVertex | None) -> WarmStartHint | None:
    if vertex is None:
        return None
    message = "Feasible vertex reconstructed from soft active-set polishing"
    if not vertex.feasible:
        message = "Soft active-set polishing reconstructed an infeasible candidate vertex"
    return WarmStartHint(
        candidate_x=vertex.x.copy(),
        active_constraint_indices=vertex.active_indices,
        rank=int(vertex.metadata.get("rank", vertex.x.shape[0])),
        feasible=vertex.feasible,
        objective=vertex.objective,
        message=message,
        constraint_system="augmented_primal",
    )


def polish_archive(
    problem: LPProblem,
    archive: list[ArchiveEntry],
    *,
    config: VertexPolishingConfig | None = None,
) -> VertexPolishingResult:
    """Run a bounded polishing pass over elite sampled primal points."""
    cfg = config or VertexPolishingConfig()
    if not cfg.enabled or not archive:
        return VertexPolishingResult(
            best_vertex=None,
            vertices=[],
            diagnostics={"enabled": cfg.enabled, "archive_size": len(archive), "polishing_time_seconds": 0.0},
        )

    canonical_problem = problem.to_maximization()
    elite_count = max(1, int(math.ceil(len(archive) * cfg.elite_fraction)))
    elite_archive = archive[:elite_count]
    A_aug, b_aug = augment_primal_constraints(canonical_problem)
    elite_samples = np.asarray([entry.x for entry in elite_archive], dtype=float)
    sample_objectives = np.asarray([entry.primal_objective for entry in elite_archive], dtype=float)
    sample_violations = np.asarray([entry.primal_violation for entry in elite_archive], dtype=float)
    raw_feasible_objective = None
    feasible_objectives = [entry.primal_objective for entry in archive if entry.primal_feasible]
    if feasible_objectives:
        raw_feasible_objective = float(max(feasible_objectives))

    start = time.perf_counter()
    candidates = generate_soft_active_set_candidates(
        elite_samples,
        A_aug,
        b_aug,
        canonical_problem.n,
        cfg.tau,
        method=cfg.method,
        max_ranked_constraints=cfg.max_ranked_constraints,
        max_candidates_per_sample=cfg.max_candidates_per_sample,
        max_total_candidates=cfg.max_total_candidates,
        sample_objectives=sample_objectives,
        sample_primal_violations=sample_violations,
    )
    vertices: list[PolishedVertex] = []
    feasible_vertices: list[PolishedVertex] = []
    for candidate in candidates:
        vertex = reconstruct_vertex_from_active_set(
            canonical_problem,
            candidate.active_indices,
            feasibility_tol=cfg.feasibility_tol,
            residual_tol=cfg.residual_tol,
            source_sample_index=candidate.source_sample_index,
            A_aug=A_aug,
            b_aug=b_aug,
            source_candidate_id=f"soft_active_set:{candidate.active_indices}",
            feasibility_config=cfg.feasibility,
        )
        if vertex is None:
            continue
        vertices.append(vertex)
        if vertex.feasible:
            feasible_vertices.append(vertex)

    best_vertex = None if not feasible_vertices else max(feasible_vertices, key=lambda vertex: vertex.objective)
    improvement = None
    if best_vertex is not None and raw_feasible_objective is not None:
        improvement = float(best_vertex.objective - raw_feasible_objective)

    diagnostics = {
        "enabled": True,
        "archive_size": len(archive),
        "elite_sample_count": elite_count,
        "vertices_reconstructed": len(vertices),
        "vertices_feasible": len(feasible_vertices),
        "augmented_constraint_count": A_aug.shape[0],
        "polishing_time_seconds": time.perf_counter() - start,
    }
    return VertexPolishingResult(
        best_vertex=best_vertex,
        vertices=vertices,
        candidates=candidates,
        candidates_tried=len(candidates),
        candidates_feasible=len(feasible_vertices),
        improvement_over_raw=improvement,
        recovered_active_set=None if best_vertex is None else best_vertex.active_indices,
        diagnostics=diagnostics,
    )
