from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lpas.certificates.bounds import BoundCertificate
    from lpas.corners.multi_corner_discovery import MultiCornerDiscoveryResult


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
class ParallelIterationMetrics:
    iteration: int
    samples_evaluated: int
    best_score: float
    best_primal_objective: float
    best_dual_objective: float
    best_gap: float
    best_primal_violation: float
    best_dual_violation: float
    best_complementarity: float
    elite_mean_score: float
    elite_mean_gap: float
    elite_mean_primal_violation: float
    elite_mean_dual_violation: float
    active_frequency_entropy: float
    elapsed_time_seconds: float
    samples_per_second: float
    backend_name: str
    device_name: str


@dataclass(frozen=True)
class WarmStartHint:
    candidate_x: np.ndarray | None
    active_constraint_indices: tuple[int, ...]
    rank: int
    feasible: bool
    objective: float | None
    message: str
    constraint_system: str = "original"


@dataclass(frozen=True)
class SoftActiveSetCandidate:
    active_indices: tuple[int, ...]
    source_sample_index: int | None
    score: float
    slacks: np.ndarray
    soft_scores: np.ndarray
    rank_method: str
    sample_objective: float | None = None
    sample_primal_violation: float | None = None


@dataclass(frozen=True)
class PolishedVertex:
    x: np.ndarray
    objective: float
    active_indices: tuple[int, ...]
    feasible: bool
    primal_violation: float
    reconstruction_residual: float
    condition_number: float | None
    source_sample_index: int | None
    original_active_mask: np.ndarray
    nonneg_active_mask: np.ndarray
    max_constraint_violation: float = 0.0
    sum_constraint_violation: float = 0.0
    polishing_status: str = "NUMERIC_FAILURE"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VertexPolishingResult:
    best_vertex: PolishedVertex | None
    vertices: list[PolishedVertex]
    candidates: list[SoftActiveSetCandidate] = field(default_factory=list)
    candidates_tried: int = 0
    candidates_feasible: int = 0
    improvement_over_raw: float | None = None
    recovered_active_set: tuple[int, ...] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScipySolveResult:
    success: bool
    status: int
    message: str
    x: np.ndarray | None
    objective: float | None
    primal_active_mask: np.ndarray | None = None
    nonneg_active_mask: np.ndarray | None = None
    augmented_primal_active_mask: np.ndarray | None = None
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
    best_raw_gap: float | None = None
    best_feasible_primal_lower_bound: float | None = None
    best_feasible_dual_upper_bound: float | None = None
    best_certified_gap: float | None = None
    best_certified_relative_gap: float | None = None
    history: list[IterationMetrics] = field(default_factory=list)
    warm_start_hint: WarmStartHint | None = None
    scipy_result: ScipySolveResult | None = None
    archive_size: int = 0
    best_score: float | None = None
    raw_best_x: np.ndarray | None = None
    raw_best_y: np.ndarray | None = None
    raw_best_primal_objective: float | None = None
    raw_best_primal_violation: float | None = None
    raw_best_active_mask: np.ndarray | None = None
    raw_best_nonneg_active_mask: np.ndarray | None = None
    polished_best_x: np.ndarray | None = None
    polished_best_primal_objective: float | None = None
    polished_best_primal_violation: float | None = None
    polished_best_active_mask: np.ndarray | None = None
    polished_best_nonneg_active_mask: np.ndarray | None = None
    polished_best_active_indices: tuple[int, ...] | None = None
    polishing_result: VertexPolishingResult | None = None
    solution_source: str = "none"
    raw_vs_scipy_active_set_jaccard: float | None = None
    polished_vs_scipy_active_set_jaccard: float | None = None
    polishing_improved_solution: bool | None = None
    polished_certified_feasible: bool | None = None
    bound_certificate: BoundCertificate | None = None
    corner_discovery: MultiCornerDiscoveryResult | None = None


@dataclass
class ParallelSolverResult:
    best_x: np.ndarray
    best_y: np.ndarray
    best_score: float
    best_primal_objective: float
    best_dual_objective: float
    best_gap: float
    best_primal_violation: float
    best_dual_violation: float
    best_complementarity_error: float
    likely_active_constraints: np.ndarray
    active_frequencies: np.ndarray
    history: list[ParallelIterationMetrics] = field(default_factory=list)
    backend: str = "numpy_cpu"
    device: str = "cpu"
    dtype: str = "float32"
