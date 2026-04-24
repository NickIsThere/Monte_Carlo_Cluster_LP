from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from lpas.certificates.bounds import BoundCertificate, build_bound_certificate
from lpas.corners.multi_corner_discovery import MultiCornerDiscoveryResult, discover_multiple_corners
from lpas.core.active_set import extract_active_sets
from lpas.core.scoring import effective_gap, score_candidates
from lpas.experiments.metrics import active_set_jaccard, exact_active_set_match, objective_gap_to_reference
from lpas.geometry.clustering import compute_cluster_support
from lpas.geometry.density_reward import compute_geometry_support
from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.solver.result import ArchiveEntry, VertexPolishingResult
from lpas.solver.scipy_handoff import ScipySolveResult, solve_with_scipy
from lpas.solver.vertex_polishing import polish_archive
from lpas.core.primal_dual import evaluate_primal_dual_pairs
from lpas.utils.config import ScoringConfig, SolverConfig


GEOMETRY_AWARE_METHOD = "geometry_aware"
GEOMETRY_AWARE_POLISHED_METHOD = "geometry_aware_polished"
NAIVE_MONTE_CARLO_METHOD = "naive_monte_carlo"


@dataclass(frozen=True)
class BenchmarkIteration:
    iteration: int
    samples_seen: int
    elapsed_seconds: float
    best_feasible_objective: float | None
    best_objective_any_candidate: float
    best_primal_violation: float
    best_dual_violation: float
    best_gap: float
    best_complementarity_error: float
    active_set_recovery_accuracy: float


@dataclass
class MethodExperimentResult:
    problem_name: str
    family: str
    method: str
    seed: int
    n_variables: int
    n_constraints: int
    reference_result: ScipySolveResult
    history: list[BenchmarkIteration]
    best_feasible_objective: float | None
    best_objective_any_candidate: float
    best_primal_violation: float
    best_dual_violation: float
    best_gap: float
    best_complementarity_error: float
    active_set_recovery_accuracy: float
    exact_active_set_match: bool
    first_recovery_iteration: int | None
    time_to_identify_optimal_active_constraints: float | None
    wall_clock_seconds: float
    n_samples_total: int
    objective_gap_to_highs: float | None
    best_scored_entry: ArchiveEntry | None
    best_raw_gap: float | None = None
    best_feasible_primal_lower_bound: float | None = None
    best_feasible_dual_upper_bound: float | None = None
    best_certified_gap: float | None = None
    best_certified_relative_gap: float | None = None
    raw_best_x: np.ndarray | None = None
    raw_best_objective: float | None = None
    raw_best_primal_violation: float | None = None
    raw_active_set_similarity: float | None = None
    raw_exact_active_set_match: bool | None = None
    polished_x: np.ndarray | None = None
    polished_objective: float | None = None
    polished_primal_violation: float | None = None
    polished_active_set_similarity: float | None = None
    polished_exact_active_set_match: bool | None = None
    polishing_improved_solution: bool | None = None
    polished_certified_feasible: bool | None = None
    polishing_wall_clock_seconds: float | None = None
    polishing_candidates_generated: int = 0
    vertices_reconstructed: int = 0
    vertices_feasible: int = 0
    solution_source: str = "raw_sampling"
    polishing_result: VertexPolishingResult | None = None
    final_x: np.ndarray | None = None
    final_active_mask: np.ndarray | None = None
    elite_archive: list[ArchiveEntry] = field(default_factory=list)
    captured_x: np.ndarray | None = None
    captured_scores: np.ndarray | None = None
    captured_is_elite: np.ndarray | None = None
    bound_certificate: BoundCertificate | None = None
    corner_discovery: MultiCornerDiscoveryResult | None = None


def _make_archive_entry(X, Y, scores, metrics, active_sets, index: int) -> ArchiveEntry:
    return ArchiveEntry(
        x=X[index].copy(),
        y=Y[index].copy(),
        score=float(scores[index]),
        primal_objective=float(metrics.primal_objective[index]),
        dual_objective=float(metrics.dual_objective[index]),
        raw_gap=float(metrics.raw_gap[index]),
        primal_violation=float(metrics.primal_violation_norm[index]),
        dual_violation=float(metrics.dual_violation_norm[index]),
        complementarity_error=float(metrics.complementarity_error[index]),
        primal_feasible=bool(metrics.primal_feasible[index]),
        dual_feasible=bool(metrics.dual_feasible[index]),
        primal_active_mask=active_sets.primal_active_mask[index].copy(),
        dual_active_mask=active_sets.dual_active_mask[index].copy(),
    )


def _scoring_config_for_method(config: SolverConfig, method: str) -> ScoringConfig:
    if method in {GEOMETRY_AWARE_METHOD, GEOMETRY_AWARE_POLISHED_METHOD}:
        return config.scoring
    if method == NAIVE_MONTE_CARLO_METHOD:
        return replace(config.scoring, w_geo=0.0, w_active=0.0)
    raise ValueError(f"unknown method: {method}")


def _extract_elite_arrays(archive: list[ArchiveEntry]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not archive:
        return None, None, None, None
    elite_x = np.asarray([entry.x for entry in archive], dtype=float)
    elite_y = np.asarray([entry.y for entry in archive], dtype=float)
    elite_primal = np.asarray([entry.primal_active_mask for entry in archive], dtype=bool)
    elite_dual = np.asarray([entry.dual_active_mask for entry in archive], dtype=bool)
    return elite_x, elite_y, elite_primal, elite_dual


def _best_feasible_entry(archive: list[ArchiveEntry]) -> ArchiveEntry | None:
    feasible_entries = [entry for entry in archive if entry.primal_feasible]
    if not feasible_entries:
        return None
    return max(feasible_entries, key=lambda entry: entry.primal_objective)


def _reference_active_mask(reference: ScipySolveResult, n_constraints: int) -> np.ndarray:
    if reference.primal_active_mask is None:
        return np.zeros(n_constraints, dtype=bool)
    return np.asarray(reference.primal_active_mask, dtype=bool)


def _apply_vertex_polishing_variant(
    base_result: MethodExperimentResult,
    problem,
    config: SolverConfig,
    *,
    recovery_jaccard_threshold: float,
) -> MethodExperimentResult:
    polishing_result = polish_archive(problem.to_maximization(), base_result.elite_archive, config=config.vertex_polishing)
    polished_best = polishing_result.best_vertex
    polishing_seconds = float(polishing_result.diagnostics.get("polishing_time_seconds", 0.0))
    reference_mask = _reference_active_mask(base_result.reference_result, base_result.n_constraints)
    bound_certificate = build_bound_certificate(
        problem.to_maximization(),
        raw_primal_objective=None if base_result.bound_certificate is None else base_result.bound_certificate.raw_primal_objective,
        raw_dual_objective=None if base_result.bound_certificate is None else base_result.bound_certificate.raw_dual_objective,
        primal_x=None if base_result.bound_certificate is None else base_result.bound_certificate.best_feasible_primal_x,
        primal_objective=None
        if base_result.bound_certificate is None
        else base_result.bound_certificate.best_feasible_primal_lower_bound,
        dual_y=None if base_result.bound_certificate is None else base_result.bound_certificate.best_feasible_dual_y,
        dual_objective=None
        if base_result.bound_certificate is None
        else base_result.bound_certificate.best_feasible_dual_upper_bound,
        polished_vertex=polished_best,
        scipy_result=base_result.reference_result,
        feasibility_tol=config.feasibility_tol,
    )

    polished_active_set_similarity = None
    polished_exact_match = None
    polished_primal_violation = None
    polished_objective = None
    polished_x = None
    if polished_best is not None:
        polished_active_set_similarity = active_set_jaccard(polished_best.original_active_mask, reference_mask)
        polished_exact_match = exact_active_set_match(polished_best.original_active_mask, reference_mask)
        polished_primal_violation = polished_best.primal_violation
        polished_objective = polished_best.objective
        polished_x = polished_best.x.copy()

    final_objective = base_result.raw_best_objective
    final_x = None if base_result.raw_best_x is None else base_result.raw_best_x.copy()
    final_active_mask = None if base_result.final_active_mask is None else base_result.final_active_mask.copy()
    final_primal_violation = base_result.raw_best_primal_violation
    solution_source = "raw_sampling"

    polishing_improved = None
    if polished_best is not None:
        if base_result.raw_best_objective is None:
            polishing_improved = bool(polished_best.feasible)
        else:
            polishing_improved = bool(polished_best.objective > base_result.raw_best_objective + 1e-12)

    if polished_best is not None and polished_best.feasible:
        if base_result.raw_best_objective is None or polished_best.objective > base_result.raw_best_objective + 1e-12:
            final_objective = polished_best.objective
            final_x = polished_best.x.copy()
            final_active_mask = polished_best.original_active_mask.copy()
            final_primal_violation = polished_best.primal_violation
            solution_source = "vertex_polishing"

    active_set_recovery_accuracy = base_result.active_set_recovery_accuracy
    exact_match = base_result.exact_active_set_match
    first_recovery_iteration = base_result.first_recovery_iteration
    time_to_recovery = base_result.time_to_identify_optimal_active_constraints
    if polished_active_set_similarity is not None:
        active_set_recovery_accuracy = max(active_set_recovery_accuracy, polished_active_set_similarity)
        exact_match = exact_match or bool(polished_exact_match)
        if first_recovery_iteration is None and polished_active_set_similarity >= recovery_jaccard_threshold:
            first_recovery_iteration = len(base_result.history)
            time_to_recovery = base_result.wall_clock_seconds + polishing_seconds

    best_objective_any_candidate = base_result.best_objective_any_candidate
    best_primal_violation = base_result.best_primal_violation
    if polished_best is not None:
        best_objective_any_candidate = max(best_objective_any_candidate, polished_best.objective)
        best_primal_violation = min(best_primal_violation, polished_best.primal_violation)

    return replace(
        base_result,
        method=GEOMETRY_AWARE_POLISHED_METHOD,
        best_feasible_objective=final_objective,
        best_objective_any_candidate=best_objective_any_candidate,
        best_primal_violation=best_primal_violation,
        active_set_recovery_accuracy=active_set_recovery_accuracy,
        exact_active_set_match=exact_match,
        first_recovery_iteration=first_recovery_iteration,
        time_to_identify_optimal_active_constraints=time_to_recovery,
        wall_clock_seconds=base_result.wall_clock_seconds + polishing_seconds,
        objective_gap_to_highs=objective_gap_to_reference(final_objective, base_result.reference_result.objective),
        best_raw_gap=bound_certificate.raw_gap,
        best_feasible_primal_lower_bound=bound_certificate.best_feasible_primal_lower_bound,
        best_feasible_dual_upper_bound=bound_certificate.best_feasible_dual_upper_bound,
        best_certified_gap=bound_certificate.certified_gap,
        best_certified_relative_gap=bound_certificate.certified_relative_gap,
        polished_x=polished_x,
        polished_objective=polished_objective,
        polished_primal_violation=polished_primal_violation,
        polished_active_set_similarity=polished_active_set_similarity,
        polished_exact_active_set_match=polished_exact_match,
        polishing_improved_solution=polishing_improved,
        polished_certified_feasible=None if polished_best is None else polished_best.feasible,
        polishing_wall_clock_seconds=polishing_seconds,
        polishing_candidates_generated=polishing_result.candidates_tried,
        vertices_reconstructed=len(polishing_result.vertices),
        vertices_feasible=polishing_result.candidates_feasible,
        solution_source=solution_source,
        polishing_result=polishing_result,
        final_x=final_x,
        final_active_mask=final_active_mask,
        bound_certificate=bound_certificate,
    )


def run_sampling_method(
    *,
    problem_name: str,
    family: str,
    problem,
    method: str,
    config: SolverConfig,
    reference_result: ScipySolveResult | None = None,
    capture_samples: bool = False,
    recovery_jaccard_threshold: float = 0.8,
) -> MethodExperimentResult:
    canonical_problem = problem.to_maximization()
    reference = reference_result or solve_with_scipy(canonical_problem)
    reference_active_mask = (
        np.zeros(canonical_problem.m, dtype=bool)
        if reference.primal_active_mask is None
        else np.asarray(reference.primal_active_mask, dtype=bool)
    )

    sampler_seed = config.sampler.seed if config.sampler.seed is not None else config.seed
    sampler = GaussianAdaptiveSampler(
        canonical_problem.n,
        canonical_problem.m,
        replace(config.sampler, seed=sampler_seed),
    )
    scoring_config = _scoring_config_for_method(config, method)

    history: list[BenchmarkIteration] = []
    archive: list[ArchiveEntry] = []
    archive_limit = config.archive_limit_multiplier * config.batch_size
    elite_count = max(1, int(np.ceil(config.elite_fraction * config.batch_size)))
    best_scored_entry: ArchiveEntry | None = None
    best_feasible_entry: ArchiveEntry | None = None
    best_dual_feasible_entry: ArchiveEntry | None = None
    best_feasible_objective: float | None = None
    best_objective_any_candidate = -np.inf
    best_primal_violation = np.inf
    best_dual_violation = np.inf
    best_gap = np.inf
    best_complementarity = np.inf
    best_active_jaccard = 0.0
    exact_match = False
    first_recovery_iteration: int | None = None
    time_to_recovery: float | None = None

    captured_x_batches: list[np.ndarray] = []
    captured_score_batches: list[np.ndarray] = []
    captured_elite_batches: list[np.ndarray] = []

    start = time.perf_counter()
    for iteration in range(config.max_iter):
        if config.time_limit_seconds is not None and time.perf_counter() - start > config.time_limit_seconds:
            break

        X = sampler.sample_primal(config.batch_size)
        Y = sampler.sample_dual(config.batch_size)
        metrics = evaluate_primal_dual_pairs(canonical_problem, X, Y, feasibility_tol=config.feasibility_tol)
        active_sets = extract_active_sets(canonical_problem, X, Y, epsilon=config.active_tol)

        if method == GEOMETRY_AWARE_METHOD:
            elite_x, elite_y, elite_primal_masks, elite_dual_masks = _extract_elite_arrays(archive)
            geometry_support = compute_geometry_support(
                X,
                Y,
                elite_x,
                elite_y,
                sigma=scoring_config.geometry_sigma,
                dual_weight=scoring_config.geometry_dual_weight,
            )
            cluster_support = compute_cluster_support(
                active_sets.primal_active_mask,
                active_sets.dual_active_mask,
                elite_primal_masks,
                elite_dual_masks,
                smoothing=scoring_config.cluster_smoothing,
            )
        else:
            geometry_support = np.zeros(config.batch_size, dtype=float)
            cluster_support = np.zeros(config.batch_size, dtype=float)

        scores = score_candidates(
            metrics,
            geometry_support=geometry_support,
            cluster_support=cluster_support,
            config=scoring_config,
        )
        elite_indices = np.argsort(scores)[-elite_count:][::-1]
        new_entries = [_make_archive_entry(X, Y, scores, metrics, active_sets, int(index)) for index in elite_indices]
        archive.extend(new_entries)
        archive.sort(key=lambda entry: entry.score, reverse=True)
        archive = archive[:archive_limit]

        if method == GEOMETRY_AWARE_METHOD:
            sampler.update(X[elite_indices], Y[elite_indices], scores[elite_indices])

        batch_best_index = int(np.argmax(scores))
        candidate_best = _make_archive_entry(X, Y, scores, metrics, active_sets, batch_best_index)
        if best_scored_entry is None or candidate_best.score > best_scored_entry.score:
            best_scored_entry = candidate_best

        current_best_objective = float(np.max(metrics.primal_objective))
        best_objective_any_candidate = max(best_objective_any_candidate, current_best_objective)
        best_primal_violation = min(best_primal_violation, float(np.min(metrics.primal_violation_norm)))
        best_dual_violation = min(best_dual_violation, float(np.min(metrics.dual_violation_norm)))
        best_gap = min(best_gap, float(np.min(effective_gap(metrics))))
        best_complementarity = min(best_complementarity, float(np.min(metrics.complementarity_error)))

        feasible_mask = np.asarray(metrics.primal_feasible, dtype=bool)
        if np.any(feasible_mask):
            feasible_indices = np.flatnonzero(feasible_mask)
            feasible_best_index = int(feasible_indices[np.argmax(np.asarray(metrics.primal_objective, dtype=float)[feasible_mask])])
            feasible_best_entry = _make_archive_entry(X, Y, scores, metrics, active_sets, feasible_best_index)
            feasible_best = feasible_best_entry.primal_objective
            best_feasible_objective = feasible_best if best_feasible_objective is None else max(best_feasible_objective, feasible_best)
            if best_feasible_entry is None or feasible_best_entry.primal_objective > best_feasible_entry.primal_objective:
                best_feasible_entry = feasible_best_entry

        dual_feasible_mask = np.asarray(metrics.dual_feasible, dtype=bool)
        if np.any(dual_feasible_mask):
            dual_feasible_indices = np.flatnonzero(dual_feasible_mask)
            dual_best_index = int(
                dual_feasible_indices[np.argmin(np.asarray(metrics.dual_objective, dtype=float)[dual_feasible_mask])]
            )
            dual_best_entry = _make_archive_entry(X, Y, scores, metrics, active_sets, dual_best_index)
            if best_dual_feasible_entry is None or dual_best_entry.dual_objective < best_dual_feasible_entry.dual_objective:
                best_dual_feasible_entry = dual_best_entry

        batch_jaccards = np.asarray(
            [active_set_jaccard(mask, reference_active_mask) for mask in active_sets.primal_active_mask],
            dtype=float,
        )
        best_active_jaccard = max(best_active_jaccard, float(np.max(batch_jaccards, initial=0.0)))
        exact_match = exact_match or bool(
            np.any([exact_active_set_match(mask, reference_active_mask) for mask in active_sets.primal_active_mask])
        )

        if first_recovery_iteration is None and best_active_jaccard >= recovery_jaccard_threshold:
            first_recovery_iteration = iteration + 1
            time_to_recovery = time.perf_counter() - start

        if capture_samples:
            elite_mask = np.zeros(config.batch_size, dtype=bool)
            elite_mask[elite_indices] = True
            captured_x_batches.append(X.copy())
            captured_score_batches.append(np.asarray(scores, dtype=float))
            captured_elite_batches.append(elite_mask)

        history.append(
            BenchmarkIteration(
                iteration=iteration + 1,
                samples_seen=(iteration + 1) * config.batch_size,
                elapsed_seconds=time.perf_counter() - start,
                best_feasible_objective=best_feasible_objective,
                best_objective_any_candidate=best_objective_any_candidate,
                best_primal_violation=best_primal_violation,
                best_dual_violation=best_dual_violation,
                best_gap=best_gap,
                best_complementarity_error=best_complementarity,
                active_set_recovery_accuracy=best_active_jaccard,
            )
        )

    wall_clock = time.perf_counter() - start
    captured_x = None if not captured_x_batches else np.vstack(captured_x_batches)
    captured_scores = None if not captured_score_batches else np.concatenate(captured_score_batches)
    captured_is_elite = None if not captured_elite_batches else np.concatenate(captured_elite_batches)
    raw_best_entry = best_feasible_entry
    raw_active_set_similarity = None
    raw_exact_match = None
    raw_best_x = None
    raw_best_primal_violation = None
    final_active_mask = None
    if raw_best_entry is not None:
        raw_best_x = raw_best_entry.x.copy()
        raw_best_primal_violation = raw_best_entry.primal_violation
        final_active_mask = raw_best_entry.primal_active_mask.copy()
        raw_active_set_similarity = active_set_jaccard(raw_best_entry.primal_active_mask, reference_active_mask)
        raw_exact_match = exact_active_set_match(raw_best_entry.primal_active_mask, reference_active_mask)

    bound_certificate = build_bound_certificate(
        canonical_problem,
        raw_x=None if raw_best_entry is None else raw_best_entry.x,
        raw_y=None if raw_best_entry is None else raw_best_entry.y,
        raw_primal_objective=None if raw_best_entry is None else raw_best_entry.primal_objective,
        raw_dual_objective=None if raw_best_entry is None else raw_best_entry.dual_objective,
        primal_x=None if best_feasible_entry is None else best_feasible_entry.x,
        primal_objective=None if best_feasible_entry is None else best_feasible_entry.primal_objective,
        dual_y=None if best_dual_feasible_entry is None else best_dual_feasible_entry.y,
        dual_objective=None if best_dual_feasible_entry is None else best_dual_feasible_entry.dual_objective,
        polished_vertex=None,
        scipy_result=reference,
        feasibility_tol=config.feasibility_tol,
    )
    corner_discovery = discover_multiple_corners(
        canonical_problem,
        np.asarray([entry.x for entry in archive], dtype=float) if archive else np.empty((0, canonical_problem.n), dtype=float),
        dual_samples=np.asarray([entry.y for entry in archive], dtype=float) if archive else None,
        reference_result=reference,
    )

    return MethodExperimentResult(
        problem_name=problem_name,
        family=family,
        method=method,
        seed=int(sampler_seed or 0),
        n_variables=canonical_problem.n,
        n_constraints=canonical_problem.m,
        reference_result=reference,
        history=history,
        best_feasible_objective=best_feasible_objective,
        best_objective_any_candidate=best_objective_any_candidate,
        best_primal_violation=best_primal_violation,
        best_dual_violation=best_dual_violation,
        best_gap=best_gap,
        best_raw_gap=bound_certificate.raw_gap,
        best_feasible_primal_lower_bound=bound_certificate.best_feasible_primal_lower_bound,
        best_feasible_dual_upper_bound=bound_certificate.best_feasible_dual_upper_bound,
        best_certified_gap=bound_certificate.certified_gap,
        best_certified_relative_gap=bound_certificate.certified_relative_gap,
        best_complementarity_error=best_complementarity,
        active_set_recovery_accuracy=best_active_jaccard,
        exact_active_set_match=exact_match,
        first_recovery_iteration=first_recovery_iteration,
        time_to_identify_optimal_active_constraints=time_to_recovery,
        wall_clock_seconds=wall_clock,
        n_samples_total=len(history) * config.batch_size,
        objective_gap_to_highs=objective_gap_to_reference(best_feasible_objective, reference.objective),
        best_scored_entry=best_scored_entry,
        raw_best_x=raw_best_x,
        raw_best_objective=best_feasible_objective,
        raw_best_primal_violation=raw_best_primal_violation,
        raw_active_set_similarity=raw_active_set_similarity,
        raw_exact_active_set_match=raw_exact_match,
        polished_x=None,
        polished_objective=None,
        polished_primal_violation=None,
        polished_active_set_similarity=None,
        polished_exact_active_set_match=None,
        polishing_improved_solution=None,
        polished_certified_feasible=None,
        polishing_wall_clock_seconds=0.0,
        polishing_candidates_generated=0,
        vertices_reconstructed=0,
        vertices_feasible=0,
        solution_source="raw_sampling",
        polishing_result=None,
        final_x=raw_best_x,
        final_active_mask=final_active_mask,
        elite_archive=archive,
        captured_x=captured_x,
        captured_scores=captured_scores,
        captured_is_elite=captured_is_elite,
        bound_certificate=bound_certificate,
        corner_discovery=corner_discovery,
    )


def run_problem_comparison(
    *,
    problem_name: str,
    family: str,
    problem,
    config: SolverConfig,
    capture_samples: bool = False,
    recovery_jaccard_threshold: float = 0.8,
) -> list[MethodExperimentResult]:
    reference = solve_with_scipy(problem)
    geometry_result = run_sampling_method(
        problem_name=problem_name,
        family=family,
        problem=problem,
        method=GEOMETRY_AWARE_METHOD,
        config=config,
        reference_result=reference,
        capture_samples=capture_samples,
        recovery_jaccard_threshold=recovery_jaccard_threshold,
    )
    results = [
        geometry_result,
        run_sampling_method(
            problem_name=problem_name,
            family=family,
            problem=problem,
            method=NAIVE_MONTE_CARLO_METHOD,
            config=config,
            reference_result=reference,
            capture_samples=capture_samples,
            recovery_jaccard_threshold=recovery_jaccard_threshold,
        ),
    ]
    if config.vertex_polishing.enabled:
        results.append(
            _apply_vertex_polishing_variant(
                geometry_result,
                problem,
                config,
                recovery_jaccard_threshold=recovery_jaccard_threshold,
            )
        )
    return results


def run_random_dense_suite(
    instances,
    *,
    config: SolverConfig,
    capture_samples: bool = False,
    recovery_jaccard_threshold: float = 0.8,
) -> list[MethodExperimentResult]:
    results: list[MethodExperimentResult] = []
    for instance in instances:
        results.extend(
            run_problem_comparison(
                problem_name=instance.name,
                family=instance.family,
                problem=instance.problem,
                config=config,
                capture_samples=capture_samples,
                recovery_jaccard_threshold=recovery_jaccard_threshold,
            )
        )
    return results
