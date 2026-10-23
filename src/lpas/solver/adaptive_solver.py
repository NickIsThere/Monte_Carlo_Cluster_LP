from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from lpas.certificates.bounds import build_bound_certificate
from lpas.core.active_set import extract_active_sets, primal_active_mask, dual_active_mask
from lpas.core.lp_problem import LPProblem
from lpas.core.primal_dual import evaluate_primal_dual_pairs, normalized_kkt_residual
from lpas.core.scoring import score_candidates
from lpas.geometry.clustering import active_pattern_key, compute_cluster_support
from lpas.geometry.density_reward import compute_geometry_support
from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.solver.result import ArchiveEntry, IterationMetrics, SolverResult, SolverStatus
from lpas.solver.pdhg import PDHGRefiner
from lpas.solver.feasibility_recovery import recover_primal_points
from lpas.solver.selection import archive_entry_key
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
        kkt_residual: float = float("inf"),
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
            kkt_residual=kkt_residual,
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
        start_time = time.perf_counter()
        deadline = None if self.config.time_limit_seconds is None else start_time + self.config.time_limit_seconds
        search_deadline = deadline
        if deadline is not None and self.config.recovery.enabled:
            search_deadline = start_time + self.config.time_limit_seconds * (1.0 - self.config.recovery.time_fraction)
        canonical_problem = problem.to_maximization()
        sampler = GaussianAdaptiveSampler(canonical_problem.n, canonical_problem.m, self.config.sampler)
        mode = self.config.search_mode
        refiner = None if mode == "sampling" else PDHGRefiner(canonical_problem, self.config.pdhg)
        # A persistent zero-start trajectory also anchors the hybrid; random
        # proposals can explore other faces without resetting this trajectory.
        anchor_X = np.zeros((1, canonical_problem.n))
        anchor_Y = np.zeros((1, canonical_problem.m))
        archive: list[ArchiveEntry] = []
        history: list[IterationMetrics] = []
        archive_limit = self.config.archive_limit_multiplier * self.config.batch_size
        elite_count = max(1, math.ceil(self.config.batch_size * self.config.elite_fraction))
        best_feasible: ArchiveEntry | None = None
        best_dual_feasible: ArchiveEntry | None = None
        best_scored: ArchiveEntry | None = None
        best_kkt_seen = np.inf
        stagnation = 0
        status = SolverStatus.MAX_ITER_REACHED
        sampling_seconds = 0.0
        directed_seconds = 0.0
        samples_evaluated = 0
        pdhg_pair_steps = 0
        termination_reason = "max_iterations"

        # Retain the shared initial point when it is feasible. Otherwise PDHG
        # could lose a valid incumbent just by taking its first exploratory step.
        initial_metrics = evaluate_primal_dual_pairs(canonical_problem, anchor_X, anchor_Y, feasibility_tol=self.config.feasibility_tol)
        initial_active = extract_active_sets(canonical_problem, anchor_X, anchor_Y, epsilon=self.config.active_tol)
        initial_kkt = float(normalized_kkt_residual(canonical_problem, initial_metrics)[0])
        initial = self._make_archive_entry(anchor_X, anchor_Y, np.zeros(1), initial_metrics, initial_active, 0, initial_kkt)
        best_scored = initial
        best_kkt_entry = initial
        best_kkt_seen = initial_kkt
        best_feasible = initial if initial.primal_feasible else None
        best_dual_feasible = initial if initial.dual_feasible else None
        samples_evaluated = 1

        for iteration in range(self.config.max_iter):
            if search_deadline is not None:
                if time.perf_counter() >= search_deadline:
                    status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                    termination_reason = "time_limit"
                    break

            stage_start = time.perf_counter()
            if mode == "pdhg":
                X, Y = anchor_X, anchor_Y
            else:
                X = sampler.sample_primal(self.config.batch_size)
                Y = sampler.sample_dual(self.config.batch_size)
                if mode == "hybrid":
                    X[0], Y[0] = anchor_X[0], anchor_Y[0]
            sampling_seconds += time.perf_counter() - stage_start
            if refiner is not None:
                stage_start = time.perf_counter()
                X, Y, steps = refiner.refine(X, Y, deadline=search_deadline)
                directed_seconds += time.perf_counter() - stage_start
                pdhg_pair_steps += steps * X.shape[0]
                anchor_X, anchor_Y = X[:1].copy(), Y[:1].copy()
            samples_evaluated += X.shape[0]
            metrics = evaluate_primal_dual_pairs(
                canonical_problem,
                X,
                Y,
                feasibility_tol=self.config.feasibility_tol,
            )
            active_sets = extract_active_sets(canonical_problem, X, Y, epsilon=self.config.active_tol)
            residuals = normalized_kkt_residual(canonical_problem, metrics)

            if archive and mode != "pdhg":
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

            elite_indices = np.argsort(scores)[-min(elite_count, X.shape[0]):][::-1]
            elite_X_batch = X[elite_indices]
            elite_Y_batch = Y[elite_indices]
            if mode != "pdhg":
                sampler.update(elite_X_batch, elite_Y_batch, scores[elite_indices])

            new_entries = [
                self._make_archive_entry(X, Y, scores, metrics, active_sets, int(index), float(residuals[index]))
                for index in elite_indices
            ]
            archive.extend(new_entries)
            archive.sort(key=archive_entry_key, reverse=True)
            archive = archive[:archive_limit]

            best_residual_index = int(np.argmin(residuals))
            candidate_best = self._make_archive_entry(X, Y, scores, metrics, active_sets, best_residual_index, float(residuals[best_residual_index]))
            if candidate_best.kkt_residual < best_kkt_entry.kkt_residual:
                best_kkt_entry = candidate_best
            if best_scored is None or archive_entry_key(candidate_best) > archive_entry_key(best_scored):
                best_scored = candidate_best

            improved = False
            feasible_mask = np.asarray(metrics.primal_feasible, dtype=bool)
            if np.any(feasible_mask):
                feasible_indices = np.flatnonzero(feasible_mask)
                feasible_best_index = int(
                    feasible_indices[np.argmax(np.asarray(metrics.primal_objective, dtype=float)[feasible_mask])]
                )
                feasible_best = self._make_archive_entry(X, Y, scores, metrics, active_sets, feasible_best_index, float(residuals[feasible_best_index]))
                if best_feasible is None or feasible_best.primal_objective > best_feasible.primal_objective:
                    best_feasible = feasible_best
                    improved = True

            dual_feasible_mask = np.asarray(metrics.dual_feasible, dtype=bool)
            if np.any(dual_feasible_mask):
                dual_feasible_indices = np.flatnonzero(dual_feasible_mask)
                dual_best_index = int(
                    dual_feasible_indices[np.argmin(np.asarray(metrics.dual_objective, dtype=float)[dual_feasible_mask])]
                )
                dual_best = self._make_archive_entry(X, Y, scores, metrics, active_sets, dual_best_index, float(residuals[dual_best_index]))
                if best_dual_feasible is None or dual_best.dual_objective < best_dual_feasible.dual_objective:
                    best_dual_feasible = dual_best
                    improved = True

            current_best_score = float(np.max(scores))
            current_kkt = float(np.min(residuals))
            if current_kkt < best_kkt_seen - 1e-12:
                best_kkt_seen = current_kkt
                improved = True

            best_certified_gap = None
            if best_feasible is not None and best_dual_feasible is not None:
                candidate_gap = float(best_dual_feasible.dual_objective - best_feasible.primal_objective)
                if np.isfinite(candidate_gap) and candidate_gap >= 0.0:
                    best_certified_gap = candidate_gap

            dominant_pattern = self._dominant_active_pattern(archive)
            history.append(
                IterationMetrics(
                    iteration=iteration,
                    best_score=current_best_score,
                    mean_score=float(np.mean(scores)),
                    best_feasible_primal_objective=None if best_feasible is None else best_feasible.primal_objective,
                    best_certified_gap=best_certified_gap,
                    mean_primal_violation=float(np.mean(metrics.primal_violation_norm)),
                    mean_dual_violation=float(np.mean(metrics.dual_violation_norm)),
                    dominant_active_pattern=dominant_pattern,
                    sampler_mu_x=sampler.primal_mean,
                    sampler_sigma_x=sampler.primal_sigma,
                    sampler_mu_y=sampler.dual_mean,
                    sampler_sigma_y=sampler.dual_sigma,
                    elapsed_seconds=time.perf_counter() - start_time,
                    best_kkt_residual=best_kkt_seen,
                    best_feasible_dual_objective=None if best_dual_feasible is None else best_dual_feasible.dual_objective,
                )
            )

            if best_certified_gap is not None and 0.0 <= best_certified_gap <= self.config.gap_tol:
                status = SolverStatus.OPTIMAL_CERTIFIED
                termination_reason = "certified_gap"
                break

            if self.config.stop_on_variance_collapse and mode == "sampling" and np.all(sampler.primal_sigma <= self.config.sampler.sigma_min * self.config.variance_collapse_factor) and np.all(
                sampler.dual_sigma <= self.config.sampler.sigma_min * self.config.variance_collapse_factor
            ):
                status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                termination_reason = "variance_collapse"
                break

            if improved:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= self.config.patience:
                status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED
                termination_reason = "stagnation"
                break
        else:
            status = SolverStatus.APPROXIMATE if best_feasible is not None else SolverStatus.MAX_ITER_REACHED

        recovery_start = time.perf_counter()
        recovered_incumbent = None
        recovery_attempts = recovery_steps = recovery_feasible = 0
        if self.config.recovery.enabled and (deadline is None or recovery_start < deadline):
            # Keep the best KKT pair independently of feasible-objective archive
            # pruning. Good infeasible PDHG iterates are essential recovery inputs.
            ordered = [best_kkt_entry]
            if archive:
                ordered.append(max(archive, key=lambda entry: entry.primal_objective))
            ordered.extend(sorted(archive, key=lambda entry: entry.kkt_residual))
            targets = []
            for entry in ordered:
                if any(np.array_equal(entry.x, other.x) for other in targets):
                    continue
                targets.append(entry)
                if len(targets) >= self.config.recovery.max_candidates:
                    break
            repaired_X, recovery_steps = recover_primal_points(
                canonical_problem, np.asarray([entry.x for entry in targets]),
                max_steps=self.config.recovery.max_steps,
                tolerance=self.config.feasibility_tol * 0.1, deadline=deadline,
            )
            recovery_attempts = len(targets)
            repaired_Y = np.asarray([entry.y for entry in targets])
            repaired_metrics = evaluate_primal_dual_pairs(canonical_problem, repaired_X, repaired_Y, feasibility_tol=self.config.feasibility_tol)
            repaired_active = extract_active_sets(canonical_problem, repaired_X, repaired_Y, epsilon=self.config.active_tol)
            repaired_residuals = normalized_kkt_residual(canonical_problem, repaired_metrics)
            best_kkt_seen = min(best_kkt_seen, float(np.min(repaired_residuals)))
            for index in np.flatnonzero(repaired_metrics.primal_feasible):
                recovery_feasible += 1
                entry = self._make_archive_entry(repaired_X, repaired_Y, np.zeros(len(targets)), repaired_metrics, repaired_active, int(index), float(repaired_residuals[index]))
                if best_feasible is None or entry.primal_objective > best_feasible.primal_objective:
                    best_feasible = recovered_incumbent = entry
                archive.append(entry)
            archive.sort(key=archive_entry_key, reverse=True)
            archive = archive[:archive_limit]
            samples_evaluated += len(targets)
        recovery_seconds = time.perf_counter() - recovery_start if self.config.recovery.enabled else 0.0
        if best_feasible is not None and status == SolverStatus.MAX_ITER_REACHED:
            status = SolverStatus.APPROXIMATE

        raw_selected = best_feasible or best_scored or best_dual_feasible
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
            raw_best_nonneg_active_mask = np.asarray(np.abs(raw_selected.x) <= self.config.active_tol, dtype=bool)

        bound_certificate = build_bound_certificate(
            canonical_problem,
            raw_x=None if raw_selected is None else raw_selected.x,
            raw_y=None if raw_selected is None else raw_selected.y,
            raw_primal_objective=None if raw_selected is None else raw_selected.primal_objective,
            raw_dual_objective=None if raw_selected is None else raw_selected.dual_objective,
            primal_x=None if best_feasible is None else best_feasible.x,
            primal_objective=None if best_feasible is None else best_feasible.primal_objective,
            dual_y=None if best_dual_feasible is None else best_dual_feasible.y,
            dual_objective=None if best_dual_feasible is None else best_dual_feasible.dual_objective,
            polished_vertex=polished_best,
            scipy_result=None,
            feasibility_tol=self.config.feasibility_tol,
        )
        if bound_certificate.has_valid_certificate and bound_certificate.certified_gap <= self.config.gap_tol:
            if status != SolverStatus.OPTIMAL_CERTIFIED:
                termination_reason = "recovery_certified_gap" if recovered_incumbent is not None else "polished_certified_gap"
            status = SolverStatus.OPTIMAL_CERTIFIED
        elif status == SolverStatus.OPTIMAL_CERTIFIED:
            # A replacement point can invalidate a previously valid bound gap.
            status = SolverStatus.APPROXIMATE
            termination_reason = "uncertified_after_polishing"

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
            final_best_dual_objective = bound_certificate.best_feasible_dual_upper_bound
            final_best_gap = bound_certificate.certified_gap
            final_best_primal_violation = polished_best.primal_violation
            final_best_dual_violation = None if not np.isfinite(bound_certificate.dual_violation) else bound_certificate.dual_violation
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
            solution_source = "feasibility_recovery" if raw_selected is recovered_incumbent else "raw_sampling"

        if final_best_x is not None:
            # Report the returned point's active set, including after polishing.
            final_best_y = None if best_dual_feasible is None else best_dual_feasible.y.copy()
            if final_best_y is None and raw_selected is not None:
                final_best_y = raw_selected.y.copy()
            best_active_set = (
                primal_active_mask(canonical_problem, final_best_x, self.config.active_tol),
                np.zeros(canonical_problem.n, dtype=bool) if final_best_y is None else dual_active_mask(canonical_problem, final_best_y, self.config.active_tol),
            )
            if final_best_y is not None:
                final_metrics = evaluate_primal_dual_pairs(canonical_problem, final_best_x, final_best_y, feasibility_tol=self.config.feasibility_tol)
                final_best_dual_objective = float(final_metrics.dual_objective[0])
                final_best_dual_violation = float(final_metrics.dual_violation_norm[0])
                final_best_complementarity_error = float(final_metrics.complementarity_error[0])
            final_best_gap = bound_certificate.certified_gap

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
            best_raw_gap=bound_certificate.raw_gap,
            best_feasible_primal_lower_bound=bound_certificate.best_feasible_primal_lower_bound,
            best_feasible_dual_upper_bound=bound_certificate.best_feasible_dual_upper_bound,
            best_certified_gap=bound_certificate.certified_gap,
            best_certified_relative_gap=bound_certificate.certified_relative_gap,
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
            bound_certificate=bound_certificate,
            diagnostics={
                "search_mode": mode,
                "runtime_seconds": time.perf_counter() - start_time,
                "sampling_time_seconds": sampling_seconds,
                "directed_time_seconds": directed_seconds,
                "candidates_evaluated": samples_evaluated,
                "pdhg_pair_steps": pdhg_pair_steps,
                "best_kkt_residual": None if not np.isfinite(best_kkt_seen) else best_kkt_seen,
                "termination_reason": termination_reason,
                "recovery_enabled": self.config.recovery.enabled,
                "recovery_time_seconds": recovery_seconds,
                "recovery_attempts": recovery_attempts,
                "recovery_point_steps": recovery_steps,
                "recovery_feasible_points": recovery_feasible,
                "recovery_improved_incumbent": recovered_incumbent is not None,
            },
        )
