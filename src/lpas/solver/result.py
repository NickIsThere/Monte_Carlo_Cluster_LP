from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SolverStatus(str, Enum):
    OPTIMAL_CERTIFIED = "OPTIMAL_CERTIFIED"
    APPROXIMATE = "APPROXIMATE"
    MAX_ITER_REACHED = "MAX_ITER_REACHED"
    INFEASIBLE_SUSPECTED = "INFEASIBLE_SUSPECTED"
    UNBOUNDED_SUSPECTED = "UNBOUNDED_SUSPECTED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True)
class ArchiveEntry:
    x: np.ndarray
    y: np.ndarray
    score: float
    primal_objective: float
    dual_objective: float
    raw_gap: float
    primal_violation: float
    dual_violation: float
    complementarity_error: float
    primal_feasible: bool
    dual_feasible: bool
    primal_active_mask: np.ndarray
    dual_active_mask: np.ndarray


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    best_score: float
    mean_score: float
    best_feasible_primal_objective: float | None
    best_certified_gap: float | None
    mean_primal_violation: float
    mean_dual_violation: float
    dominant_active_pattern: tuple[tuple[bool, ...], tuple[bool, ...]] | None
    sampler_mu_x: np.ndarray
    sampler_sigma_x: np.ndarray
    sampler_mu_y: np.ndarray
    sampler_sigma_y: np.ndarray


@dataclass(frozen=True)
class WarmStartHint:
    candidate_x: np.ndarray | None
    active_constraint_indices: tuple[int, ...]
    rank: int
    feasible: bool
    objective: float | None
    message: str


@dataclass(frozen=True)
class ScipySolveResult:
    success: bool
    status: int
    message: str
    x: np.ndarray | None
    objective: float | None
    primal_active_mask: np.ndarray | None = None
    raw_status: object | None = None


@dataclass
class SolverResult:
    best_x: np.ndarray | None
    best_y: np.ndarray | None
    best_primal_objective: float | None
    best_dual_objective: float | None
    best_gap: float | None
    best_primal_violation: float | None
    best_dual_violation: float | None
    best_complementarity_error: float | None
    best_active_set: tuple[np.ndarray, np.ndarray] | None
    iterations: int
    status: SolverStatus
    history: list[IterationMetrics] = field(default_factory=list)
    warm_start_hint: WarmStartHint | None = None
    scipy_result: ScipySolveResult | None = None
    archive_size: int = 0
    best_score: float | None = None
