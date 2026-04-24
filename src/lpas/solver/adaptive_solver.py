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

            feasible_entries = [entry for entry in new_entries if entry.primal_feasible]
            if feasible_entries:
                feasible_best = max(feasible_entries, key=lambda entry: entry.primal_objective)
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

        selected = best_certificate or best_feasible or best_scored
        if selected is not None:
            warm_start_hint = reconstruct_from_active_set(
                canonical_problem,
                selected.primal_active_mask,
                config=self.config.warm_start,
            )
            if not warm_start_hint.feasible:
                warm_start_hint = reconstruct_from_archive(canonical_problem, archive, config=self.config.warm_start)
        else:
            warm_start_hint = reconstruct_from_archive(canonical_problem, archive, config=self.config.warm_start)
        best_active_set = None
        if selected is not None:
            best_active_set = (selected.primal_active_mask.copy(), selected.dual_active_mask.copy())
        else:
            dominant_pattern = self._dominant_active_pattern(archive)
            if dominant_pattern is not None:
                best_active_set = (
                    np.asarray(dominant_pattern[0], dtype=bool),
                    np.asarray(dominant_pattern[1], dtype=bool),
                )
        return SolverResult(
            best_x=None if selected is None else selected.x.copy(),
            best_y=None if selected is None else selected.y.copy(),
            best_primal_objective=None if selected is None else selected.primal_objective,
            best_dual_objective=None if selected is None else selected.dual_objective,
            best_gap=None if selected is None else selected.raw_gap,
            best_primal_violation=None if selected is None else selected.primal_violation,
            best_dual_violation=None if selected is None else selected.dual_violation,
            best_complementarity_error=None if selected is None else selected.complementarity_error,
            best_active_set=best_active_set,
            iterations=len(history),
            status=status,
            history=history,
            warm_start_hint=warm_start_hint,
            archive_size=len(archive),
            best_score=None if selected is None else selected.score,
        )
