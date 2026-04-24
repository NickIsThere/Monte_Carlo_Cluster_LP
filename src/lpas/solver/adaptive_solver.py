from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from lpas.core.active_set import extract_active_sets
from lpas.core.certificates import select_best_certificate
from lpas.core.lp_problem import LPProblem
from lpas.core.primal_dual import evaluate_primal_dual_pairs
from lpas.core.scoring import score_candidates
from lpas.geometry.clustering import active_pattern_key, compute_cluster_support
from lpas.geometry.density_reward import compute_geometry_support
from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.solver.result import ArchiveEntry, IterationMetrics, SolverResult, SolverStatus
from lpas.solver.vertex_polishing import polish_archive, polished_vertex_to_warm_start_hint
from lpas.solver.warm_start import reconstruct_from_active_set, reconstruct_from_archive
from lpas.utils.config import SolverConfig


class AdaptiveLPSolver:
    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()

    def _make_archive_entry(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        scores: np.ndarray,
        metrics,
        active_sets,
        index: int,
    ) -> ArchiveEntry:
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

    def _dominant_active_pattern(self, archive: list[ArchiveEntry]) -> tuple[tuple[bool, ...], tuple[bool, ...]] | None:
        if not archive:
            return None
        counter = Counter(
            active_pattern_key(entry.primal_active_mask, entry.dual_active_mask)
            for entry in archive
        )
        return counter.most_common(1)[0][0]

    def solve(self, problem: LPProblem) -> SolverResult:
        canonical_problem = problem.to_maximization()
        sampler = GaussianAdaptiveSampler(canonical_problem.n, canonical_problem.m, self.config.sampler)
        archive: list[ArchiveEntry] = []
        history: list[IterationMetrics] = []
        archive_limit = self.config.archive_limit_multiplier * self.config.batch_size
        elite_count = max(1, math.ceil(self.config.batch_size * self.config.elite_fraction))
        best_certificate: ArchiveEntry | None = None
        best_feasible: ArchiveEntry | None = None
        best_scored: ArchiveEntry | None = None
        best_score_seen = -np.inf
        stagnation = 0
        status = SolverStatus.MAX_ITER_REACHED
        start_time = time.perf_counter()

        for iteration in range(self.config.max_iter):
            if self.config.time_limit_seconds is not None:
                if time.perf_counter() - start_time > self.config.time_limit_seconds:
                    status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                    break

            X = sampler.sample_primal(self.config.batch_size)
            Y = sampler.sample_dual(self.config.batch_size)
            metrics = evaluate_primal_dual_pairs(
                canonical_problem,
                X,
                Y,
                feasibility_tol=self.config.feasibility_tol,
            )
            active_sets = extract_active_sets(canonical_problem, X, Y, epsilon=self.config.active_tol)

            if archive:
                elite_X = np.array([entry.x for entry in archive], dtype=float)
                elite_Y = np.array([entry.y for entry in archive], dtype=float)
                elite_primal_masks = np.array([entry.primal_active_mask for entry in archive], dtype=bool)
                elite_dual_masks = np.array([entry.dual_active_mask for entry in archive], dtype=bool)
            else:
                elite_X = None
                elite_Y = None
                elite_primal_masks = None
                elite_dual_masks = None

            geometry_support = compute_geometry_support(
                X,
                Y,
                elite_X,
                elite_Y,
                sigma=self.config.scoring.geometry_sigma,
                dual_weight=self.config.scoring.geometry_dual_weight,
            )
            cluster_support = compute_cluster_support(
                active_sets.primal_active_mask,
                active_sets.dual_active_mask,
                elite_primal_masks,
                elite_dual_masks,
                smoothing=self.config.scoring.cluster_smoothing,
            )
            scores = score_candidates(
                metrics,
                geometry_support=geometry_support,
                cluster_support=cluster_support,
                config=self.config.scoring,
            )

            elite_indices = np.argsort(scores)[-elite_count:][::-1]
            elite_X_batch = X[elite_indices]
            elite_Y_batch = Y[elite_indices]
            sampler.update(elite_X_batch, elite_Y_batch, scores[elite_indices])

            new_entries = [
                self._make_archive_entry(X, Y, scores, metrics, active_sets, int(index))
                for index in elite_indices
            ]
            archive.extend(new_entries)
            archive.sort(key=lambda entry: entry.score, reverse=True)
            archive = archive[:archive_limit]

            candidate_best = max(new_entries, key=lambda entry: entry.score)
            if best_scored is None or candidate_best.score > best_scored.score:
                best_scored = candidate_best

            feasible_mask = np.asarray(metrics.primal_feasible, dtype=bool)
            if np.any(feasible_mask):
                feasible_indices = np.flatnonzero(feasible_mask)
                feasible_best_index = int(
                    feasible_indices[np.argmax(np.asarray(metrics.primal_objective, dtype=float)[feasible_mask])]
                )
                feasible_best = self._make_archive_entry(X, Y, scores, metrics, active_sets, feasible_best_index)
                if best_feasible is None or feasible_best.primal_objective > best_feasible.primal_objective:
                    best_feasible = feasible_best

            certified_index = select_best_certificate(metrics, gap_tol=self.config.gap_tol)
            if certified_index is not None:
                candidate_certificate = self._make_archive_entry(X, Y, scores, metrics, active_sets, int(certified_index))
                candidate_gap = max(candidate_certificate.raw_gap, 0.0)
                if best_certificate is None or candidate_gap < max(best_certificate.raw_gap, 0.0):
                    best_certificate = candidate_certificate

            improved = False
            current_best_score = float(np.max(scores))
            if current_best_score > best_score_seen + 1e-12:
                best_score_seen = current_best_score
                improved = True

            dominant_pattern = self._dominant_active_pattern(archive)
            history.append(
                IterationMetrics(
                    iteration=iteration,
                    best_score=current_best_score,
                    mean_score=float(np.mean(scores)),
                    best_feasible_primal_objective=None if best_feasible is None else best_feasible.primal_objective,
                    best_certified_gap=None if best_certificate is None else max(best_certificate.raw_gap, 0.0),
                    mean_primal_violation=float(np.mean(metrics.primal_violation_norm)),
                    mean_dual_violation=float(np.mean(metrics.dual_violation_norm)),
                    dominant_active_pattern=dominant_pattern,
                    sampler_mu_x=sampler.primal_mean,
                    sampler_sigma_x=sampler.primal_sigma,
                    sampler_mu_y=sampler.dual_mean,
                    sampler_sigma_y=sampler.dual_sigma,
                )
            )

            if best_certificate is not None and best_certificate.raw_gap <= self.config.gap_tol:
                status = SolverStatus.OPTIMAL_CERTIFIED
                break

            if np.all(sampler.primal_sigma <= self.config.sampler.sigma_min * self.config.variance_collapse_factor) and np.all(
                sampler.dual_sigma <= self.config.sampler.sigma_min * self.config.variance_collapse_factor
            ):
                status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                break

            if improved:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= self.config.patience:
                status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                break
        else:
            status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED

        raw_selected = best_feasible or best_scored or best_certificate
        polishing_result = polish_archive(
            canonical_problem,
            archive,
            config=self.config.vertex_polishing,
        )
        polished_best = polishing_result.best_vertex
        if polished_best is not None and polished_best.feasible and status == SolverStatus.MAX_ITER_REACHED:
            status = SolverStatus.APPROXIMATE

        use_polished = bool(
            polished_best is not None
            and polished_best.feasible
            and (
                raw_selected is None
                or not raw_selected.primal_feasible
                or polished_best.objective > raw_selected.primal_objective + 1e-12
            )
        )

        if polished_best is not None and polished_best.feasible:
            warm_start_hint = polished_vertex_to_warm_start_hint(polished_best)
        elif raw_selected is not None:
            warm_start_hint = reconstruct_from_active_set(
                canonical_problem,
                raw_selected.primal_active_mask,
                config=self.config.warm_start,
            )
            if not warm_start_hint.feasible:
                warm_start_hint = reconstruct_from_archive(canonical_problem, archive, config=self.config.warm_start)
        else:
            warm_start_hint = reconstruct_from_archive(canonical_problem, archive, config=self.config.warm_start)

        best_active_set = None
        if raw_selected is not None:
            best_active_set = (raw_selected.primal_active_mask.copy(), raw_selected.dual_active_mask.copy())
        else:
            dominant_pattern = self._dominant_active_pattern(archive)
            if dominant_pattern is not None:
                best_active_set = (
                    np.asarray(dominant_pattern[0], dtype=bool),
                    np.asarray(dominant_pattern[1], dtype=bool),
                )

        raw_best_nonneg_active_mask = None
        if raw_selected is not None:
            raw_best_nonneg_active_mask = np.asarray(raw_selected.x <= self.config.active_tol, dtype=bool)

        final_best_x = None
        final_best_y = None
        final_best_primal_objective = None
        final_best_dual_objective = None
        final_best_gap = None
        final_best_primal_violation = None
        final_best_dual_violation = None
        final_best_complementarity_error = None
        solution_source = "none"

        if use_polished and polished_best is not None:
            final_best_x = polished_best.x.copy()
            final_best_primal_objective = polished_best.objective
            final_best_primal_violation = polished_best.primal_violation
            solution_source = "vertex_polishing"
        elif raw_selected is not None:
            final_best_x = raw_selected.x.copy()
            final_best_y = raw_selected.y.copy()
            final_best_primal_objective = raw_selected.primal_objective
            final_best_dual_objective = raw_selected.dual_objective
            final_best_gap = raw_selected.raw_gap
            final_best_primal_violation = raw_selected.primal_violation
            final_best_dual_violation = raw_selected.dual_violation
            final_best_complementarity_error = raw_selected.complementarity_error
            solution_source = "raw_sampling"

        polishing_improved_solution = None
        if polished_best is not None:
            if raw_selected is None or not raw_selected.primal_feasible:
                polishing_improved_solution = bool(polished_best.feasible)
            else:
                polishing_improved_solution = bool(polished_best.objective > raw_selected.primal_objective + 1e-12)

        return SolverResult(
            best_x=final_best_x,
            best_y=final_best_y,
            best_primal_objective=final_best_primal_objective,
            best_dual_objective=final_best_dual_objective,
            best_gap=final_best_gap,
            best_primal_violation=final_best_primal_violation,
            best_dual_violation=final_best_dual_violation,
            best_complementarity_error=final_best_complementarity_error,
            best_active_set=best_active_set,
            iterations=len(history),
            status=status,
            history=history,
            warm_start_hint=warm_start_hint,
            archive_size=len(archive),
            best_score=None if raw_selected is None else raw_selected.score,
            raw_best_x=None if raw_selected is None else raw_selected.x.copy(),
            raw_best_y=None if raw_selected is None else raw_selected.y.copy(),
            raw_best_primal_objective=None if raw_selected is None else raw_selected.primal_objective,
            raw_best_primal_violation=None if raw_selected is None else raw_selected.primal_violation,
            raw_best_active_mask=None if raw_selected is None else raw_selected.primal_active_mask.copy(),
            raw_best_nonneg_active_mask=raw_best_nonneg_active_mask,
            polished_best_x=None if polished_best is None else polished_best.x.copy(),
            polished_best_primal_objective=None if polished_best is None else polished_best.objective,
            polished_best_primal_violation=None if polished_best is None else polished_best.primal_violation,
            polished_best_active_mask=None if polished_best is None else polished_best.original_active_mask.copy(),
            polished_best_nonneg_active_mask=None if polished_best is None else polished_best.nonneg_active_mask.copy(),
            polished_best_active_indices=None if polished_best is None else polished_best.active_indices,
            polishing_result=polishing_result,
            solution_source=solution_source,
            polishing_improved_solution=polishing_improved_solution,
            polished_certified_feasible=None if polished_best is None else polished_best.feasible,
        )
