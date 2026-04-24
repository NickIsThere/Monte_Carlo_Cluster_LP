from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy as np

from lpas.backends import concatenate_metric_dicts, create_backend, elite_count_from_fraction, slice_metric_dict
from lpas.backends.base import CandidateBackend
from lpas.core.lp_problem import LPProblem
from lpas.solver.result import ParallelIterationMetrics, ParallelSolverResult
from lpas.utils.config import ParallelSolverConfig


def _chunk_sizes(total: int, chunk_size: int) -> list[int]:
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def _active_frequency_entropy(active_frequency: np.ndarray) -> float:
    if active_frequency.size == 0:
        return 0.0
    clipped = np.clip(active_frequency, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return float(np.mean(entropy))


def _validate_parallel_config(config: ParallelSolverConfig) -> ParallelSolverConfig:
    if config.samples_per_iteration <= 0:
        raise ValueError("samples_per_iteration must be positive")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.chunk_size is not None and config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    if config.chunk_size is not None and config.chunk_size > config.samples_per_iteration:
        return replace(config, chunk_size=config.samples_per_iteration)
    return config


def _build_result(
    *,
    best_record: dict[str, Any],
    active_statistics: dict[str, Any],
    history: list[ParallelIterationMetrics],
    backend: CandidateBackend,
) -> ParallelSolverResult:
    return ParallelSolverResult(
        best_x=np.asarray(best_record["x"], dtype=float),
        best_y=np.asarray(best_record["y"], dtype=float),
        best_score=float(best_record["score"]),
        best_primal_objective=float(best_record["primal_objective"]),
        best_dual_objective=float(best_record["dual_objective"]),
        best_gap=float(best_record["gap"]),
        best_primal_violation=float(best_record["primal_violation"]),
        best_dual_violation=float(best_record["dual_violation"]),
        best_complementarity_error=float(best_record["complementarity"]),
        likely_active_constraints=np.asarray(active_statistics["likely_active_constraints"], dtype=int),
        active_frequencies=np.asarray(active_statistics["active_frequency"], dtype=float),
        history=history,
        backend=backend.requested_backend,
        device=backend.device_name,
        dtype=backend.dtype_name,
    )


def solve_with_backend(problem: LPProblem, config: ParallelSolverConfig, backend: CandidateBackend) -> ParallelSolverResult:
    config = _validate_parallel_config(config)
    canonical_problem = problem.to_maximization()
    problem_data = backend.prepare_problem(canonical_problem)
    state = backend.initialize_distribution(canonical_problem, config)
    total_samples = config.samples_per_iteration
    chunk_size = config.chunk_size or total_samples
    elite_count = elite_count_from_fraction(total_samples, config.elite_fraction)
    active_frequency = None
    final_active_statistics = {
        "active_frequency": np.zeros(canonical_problem.m, dtype=float),
        "likely_active_constraints": np.array([], dtype=int),
    }
    history: list[ParallelIterationMetrics] = []
    best_record: dict[str, Any] | None = None

    for iteration in range(config.iterations):
        local_X = []
        local_Y = []
        local_scores = []
        local_metrics = []
        backend.sync()
        start = time.perf_counter()

        for current_chunk in _chunk_sizes(total_samples, chunk_size):
            X_chunk, Y_chunk = backend.sample_candidates(state, current_chunk)
            metrics = backend.evaluate_batch(
                problem_data,
                X_chunk,
                Y_chunk,
                active_epsilon=config.backend.active_epsilon,
            )
            scores = backend.score_batch(
                problem_data,
                X_chunk,
                Y_chunk,
                metrics,
                config.scoring,
                active_frequency=active_frequency,
                active_epsilon=config.backend.active_epsilon,
                dual_positive_epsilon=config.backend.dual_positive_epsilon,
            )
            keep_count = min(current_chunk, elite_count)
            elite_X, elite_Y, elite_scores, elite_indices = backend.select_elites(X_chunk, Y_chunk, scores, keep_count)
            local_X.append(elite_X)
            local_Y.append(elite_Y)
            local_scores.append(elite_scores)
            local_metrics.append(slice_metric_dict(backend, metrics, elite_indices))

        merged_X = backend.concatenate(local_X, axis=0)
        merged_Y = backend.concatenate(local_Y, axis=0)
        merged_scores = backend.concatenate(local_scores, axis=0)
        merged_metrics = concatenate_metric_dicts(backend, local_metrics)
        elite_X, elite_Y, elite_scores, elite_indices = backend.select_elites(merged_X, merged_Y, merged_scores, elite_count)
        elite_metrics = slice_metric_dict(backend, merged_metrics, elite_indices)
        state = backend.update_distribution(state, elite_X, elite_Y, config)
        active_statistics = backend.compute_active_statistics(
            problem_data,
            elite_X,
            elite_Y,
            active_epsilon=config.backend.active_epsilon,
            dual_positive_epsilon=config.backend.dual_positive_epsilon,
        )
        active_frequency = active_statistics["active_frequency"]
        final_active_statistics = {
            "active_frequency": backend.to_numpy(active_statistics["active_frequency"]),
            "likely_active_constraints": backend.to_numpy(active_statistics["likely_active_constraints"]),
        }

        current_best = {
            "x": backend.to_numpy(elite_X[0]),
            "y": backend.to_numpy(elite_Y[0]),
            "score": backend.to_scalar(elite_scores[0]),
            "primal_objective": backend.to_scalar(elite_metrics["primal_objective"][0]),
            "dual_objective": backend.to_scalar(elite_metrics["dual_objective"][0]),
            "gap": backend.to_scalar(elite_metrics["gap"][0]),
            "primal_violation": backend.to_scalar(elite_metrics["primal_violation"][0]),
            "dual_violation": backend.to_scalar(elite_metrics["dual_violation"][0]),
            "complementarity": backend.to_scalar(elite_metrics["complementarity"][0]),
        }
        if best_record is None or current_best["score"] > best_record["score"]:
            best_record = current_best

        backend.sync()
        elapsed = time.perf_counter() - start
        elite_scores_np = backend.to_numpy(elite_scores)
        elite_gap_np = backend.to_numpy(elite_metrics["gap"])
        elite_pviol_np = backend.to_numpy(elite_metrics["primal_violation"])
        elite_dviol_np = backend.to_numpy(elite_metrics["dual_violation"])
        history.append(
            ParallelIterationMetrics(
                iteration=iteration,
                samples_evaluated=total_samples,
                best_score=float(best_record["score"]),
                best_primal_objective=float(best_record["primal_objective"]),
                best_dual_objective=float(best_record["dual_objective"]),
                best_gap=float(best_record["gap"]),
                best_primal_violation=float(best_record["primal_violation"]),
                best_dual_violation=float(best_record["dual_violation"]),
                best_complementarity=float(best_record["complementarity"]),
                elite_mean_score=float(np.mean(elite_scores_np)),
                elite_mean_gap=float(np.mean(elite_gap_np)),
                elite_mean_primal_violation=float(np.mean(elite_pviol_np)),
                elite_mean_dual_violation=float(np.mean(elite_dviol_np)),
                active_frequency_entropy=_active_frequency_entropy(final_active_statistics["active_frequency"]),
                elapsed_time_seconds=float(elapsed),
                samples_per_second=float(total_samples / elapsed) if elapsed > 0.0 else math.inf,
                backend_name=backend.requested_backend,
                device_name=backend.device_name,
            )
        )

    if best_record is None:
        raise RuntimeError("The parallel solver did not evaluate any candidates.")
    return _build_result(best_record=best_record, active_statistics=final_active_statistics, history=history, backend=backend)


class ParallelLPSolver:
    def __init__(self, config: ParallelSolverConfig | None = None) -> None:
        self.config = _validate_parallel_config(config or ParallelSolverConfig())

    def solve(self, problem: LPProblem) -> ParallelSolverResult:
        backend = create_backend(self.config.backend.backend, dtype=self.config.backend.dtype)
        return solve_with_backend(problem, self.config, backend)
