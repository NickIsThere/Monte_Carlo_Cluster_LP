from __future__ import annotations

from typing import Any

import numpy as np

from lpas.backends.base import CandidateBackend, BackendInfo, numpy_dtype_from_name
from lpas.core.lp_problem import LPProblem
from lpas.utils.config import ParallelScoreConfig, ParallelSolverConfig
from lpas.utils.random import make_rng


def evaluate_primal_dual_batch_numpy(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    active_epsilon: float = 1e-5,
) -> dict[str, np.ndarray]:
    AX = X @ A.T
    primal_slack = b[None, :] - AX
    primal_violation = np.maximum(AX - b[None, :], 0.0).sum(axis=1)
    primal_violation += np.maximum(-X, 0.0).sum(axis=1)

    YA = Y @ A
    dual_slack = YA - c[None, :]
    dual_violation = np.maximum(c[None, :] - YA, 0.0).sum(axis=1)
    dual_violation += np.maximum(-Y, 0.0).sum(axis=1)

    primal_objective = X @ c
    dual_objective = Y @ b
    gap = dual_objective - primal_objective
    complementarity = np.abs(Y * primal_slack).sum(axis=1)
    active_mask = primal_slack <= active_epsilon
    active_count = active_mask.sum(axis=1)
    return {
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "gap": gap,
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "complementarity": complementarity,
        "primal_slack": primal_slack,
        "dual_slack": dual_slack,
        "active_mask": active_mask,
        "active_count": active_count,
    }


def score_primal_dual_batch_numpy(
    metrics: dict[str, np.ndarray],
    weights: ParallelScoreConfig,
    *,
    active_support: np.ndarray | None = None,
    active_agreement: np.ndarray | None = None,
    active_conflict: np.ndarray | None = None,
) -> np.ndarray:
    score = weights.objective * np.asarray(metrics["primal_objective"], dtype=float)
    score -= weights.gap * np.maximum(np.asarray(metrics["gap"], dtype=float), 0.0)
    score -= weights.primal_violation * np.asarray(metrics["primal_violation"], dtype=float)
    score -= weights.dual_violation * np.asarray(metrics["dual_violation"], dtype=float)
    score -= weights.complementarity * np.asarray(metrics["complementarity"], dtype=float)
    if active_support is not None:
        score += weights.active_support * np.asarray(active_support, dtype=float)
    if active_agreement is not None:
        score += weights.active_agreement * np.asarray(active_agreement, dtype=float)
    if active_conflict is not None:
        score -= weights.active_conflict * np.asarray(active_conflict, dtype=float)
    return np.nan_to_num(score, nan=-1e12, posinf=1e12, neginf=-1e12)


def active_support_numpy(active_mask: np.ndarray, active_frequency: np.ndarray | None) -> np.ndarray:
    if active_frequency is None:
        return np.zeros(active_mask.shape[0], dtype=float)
    return (active_mask.astype(float) * active_frequency[None, :]).sum(axis=1)


def active_dual_agreement_numpy(
    Y: np.ndarray,
    active_mask: np.ndarray,
    *,
    dual_positive_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    dual_positive = Y > dual_positive_epsilon
    agreement = (dual_positive & active_mask).sum(axis=1, dtype=float)
    conflict = (dual_positive & ~active_mask).sum(axis=1, dtype=float)
    return agreement, conflict


class NumpyBackend(CandidateBackend):
    def __init__(self, requested_backend: str = "numpy_cpu", *, dtype: str = "float32") -> None:
        self.dtype = numpy_dtype_from_name(dtype)
        super().__init__(requested_backend, dtype=dtype)
        self.info = BackendInfo(
            requested_backend=requested_backend,
            resolved_backend="numpy_cpu",
            device_name="cpu",
            dtype=dtype,
        )

    def prepare_problem(self, problem: LPProblem) -> dict[str, np.ndarray]:
        return {
            "A": np.asarray(problem.A, dtype=self.dtype),
            "b": np.asarray(problem.b, dtype=self.dtype),
            "c": np.asarray(problem.c, dtype=self.dtype),
        }

    def to_backend_array(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=self.dtype)

    def initialize_distribution(self, problem: LPProblem, config: ParallelSolverConfig) -> dict[str, Any]:
        return {
            "mu_x": np.full(problem.n, config.sampler.primal_init_mean, dtype=self.dtype),
            "sigma_x": np.full(problem.n, config.sampler.sigma_init, dtype=self.dtype),
            "mu_y": np.full(problem.m, config.sampler.dual_init_mean, dtype=self.dtype),
            "sigma_y": np.full(problem.m, config.sampler.sigma_init, dtype=self.dtype),
            "rng": make_rng(config.sampler.seed),
        }

    def sample_candidates(self, state: dict[str, Any], sample_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = state["rng"]
        X = state["mu_x"][None, :] + state["sigma_x"][None, :] * rng.standard_normal(size=(sample_size, state["mu_x"].shape[0]))
        Y = state["mu_y"][None, :] + state["sigma_y"][None, :] * rng.standard_normal(size=(sample_size, state["mu_y"].shape[0]))
        return np.maximum(X.astype(self.dtype, copy=False), 0.0), np.maximum(Y.astype(self.dtype, copy=False), 0.0)

    def evaluate_batch(
        self,
        problem_data: dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        *,
        active_epsilon: float,
    ) -> dict[str, np.ndarray]:
        return evaluate_primal_dual_batch_numpy(problem_data["A"], problem_data["b"], problem_data["c"], X, Y, active_epsilon=active_epsilon)

    def score_batch(
        self,
        problem_data: dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        metrics: dict[str, np.ndarray],
        weights: ParallelScoreConfig,
        *,
        active_frequency: np.ndarray | None,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> np.ndarray:
        active_mask = np.asarray(metrics["active_mask"], dtype=bool)
        active_support = None
        if weights.active_support != 0.0 and active_frequency is not None:
            active_support = active_support_numpy(active_mask, active_frequency)
        active_agreement = None
        active_conflict = None
        if weights.active_agreement != 0.0 or weights.active_conflict != 0.0:
            active_agreement, active_conflict = active_dual_agreement_numpy(
                Y,
                active_mask,
                dual_positive_epsilon=dual_positive_epsilon,
            )
        return score_primal_dual_batch_numpy(
            metrics,
            weights,
            active_support=active_support,
            active_agreement=active_agreement,
            active_conflict=active_conflict,
        )

    def select_elites(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        scores: np.ndarray,
        elite_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if elite_count <= 0:
            raise ValueError("elite_count must be positive")
        elite_count = min(elite_count, scores.shape[0])
        if elite_count == scores.shape[0]:
            elite_indices = np.argsort(scores)[::-1]
        else:
            elite_indices = np.argpartition(scores, -elite_count)[-elite_count:]
            elite_indices = elite_indices[np.argsort(scores[elite_indices])[::-1]]
        return X[elite_indices], Y[elite_indices], scores[elite_indices], elite_indices

    def update_distribution(
        self,
        state: dict[str, Any],
        elite_X: np.ndarray,
        elite_Y: np.ndarray,
        config: ParallelSolverConfig,
    ) -> dict[str, Any]:
        alpha = config.sampler.alpha
        min_sigma = config.sampler.sigma_min
        max_sigma = config.sampler.sigma_max
        new_mu_x = elite_X.mean(axis=0)
        new_sigma_x = elite_X.std(axis=0, ddof=0)
        new_mu_y = elite_Y.mean(axis=0)
        new_sigma_y = elite_Y.std(axis=0, ddof=0)
        state["mu_x"] = alpha * new_mu_x + (1.0 - alpha) * state["mu_x"]
        state["mu_y"] = alpha * new_mu_y + (1.0 - alpha) * state["mu_y"]
        state["sigma_x"] = np.clip(alpha * new_sigma_x + (1.0 - alpha) * state["sigma_x"], min_sigma, max_sigma)
        state["sigma_y"] = np.clip(alpha * new_sigma_y + (1.0 - alpha) * state["sigma_y"], min_sigma, max_sigma)
        return state

    def compute_active_statistics(
        self,
        problem_data: dict[str, np.ndarray],
        elite_X: np.ndarray,
        elite_Y: np.ndarray,
        *,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> dict[str, np.ndarray]:
        del elite_Y, dual_positive_epsilon
        primal_slack = problem_data["b"][None, :] - elite_X @ problem_data["A"].T
        active_mask = primal_slack <= active_epsilon
        active_frequency = active_mask.astype(float).mean(axis=0)
        return {
            "elite_active_mask": active_mask,
            "active_frequency": active_frequency,
            "likely_active_constraints": np.flatnonzero(active_frequency >= 0.5),
        }

    def concatenate(self, values: Any, *, axis: int = 0) -> np.ndarray:
        return np.concatenate(list(values), axis=axis)

    def take(self, value: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return value[indices]

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def to_scalar(self, value: Any) -> float:
        return float(np.asarray(value).item())
