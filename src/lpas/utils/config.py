from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeasibilityConfig:
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-7
    nonnegativity_tolerance: float = 1e-8


@dataclass(frozen=True)
class SamplerConfig:
    seed: int | None = 0
    alpha: float = 0.7
    sigma_init: float = 2.0
    sigma_min: float = 1e-6
    sigma_max: float = 10.0
    primal_init_mean: float = 0.5
    dual_init_mean: float = 0.5
    max_retries: int = 8
    primal_violation_threshold: float = 1.0
    dual_violation_threshold: float = 1.0


@dataclass(frozen=True)
class ScoringConfig:
    w_primal_obj: float = 1.0
    w_dual_obj: float = 0.25
    w_gap: float = 1.5
    w_pviol: float = 2.0
    w_dviol: float = 2.0
    w_comp: float = 1.0
    w_geo: float = 0.2
    w_active: float = 0.2
    geometry_sigma: float = 1.0
    geometry_dual_weight: float = 0.5
    active_similarity_beta: float = 0.5
    cluster_smoothing: float = 0.0


@dataclass(frozen=True)
class WarmStartConfig:
    feasibility_tol: float = 1e-7
    max_combinations: int = 512


@dataclass(frozen=True)
class VertexPolishingConfig:
    enabled: bool = True
    elite_fraction: float = 0.05
    tau: float = 1e-2
    method: str = "rbf"
    max_ranked_constraints: int | None = None
    max_candidates_per_sample: int = 100
    max_total_candidates: int = 5000
    feasibility_tol: float = 1e-8
    residual_tol: float = 1e-8
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "auto_torch"
    dtype: str = "float32"
    active_epsilon: float = 1e-5
    dual_positive_epsilon: float = 1e-6


@dataclass(frozen=True)
class ParallelScoreConfig:
    objective: float = 1.0
    gap: float = 1.0
    primal_violation: float = 100.0
    dual_violation: float = 100.0
    complementarity: float = 0.1
    active_support: float = 0.0
    active_agreement: float = 0.0
    active_conflict: float = 0.0


@dataclass(frozen=True)
class ParallelSolverConfig:
    samples_per_iteration: int = 100_000
    chunk_size: int | None = None
    elite_fraction: float = 0.05
    iterations: int = 100
    backend: BackendConfig = field(default_factory=BackendConfig)
    scoring: ParallelScoreConfig = field(default_factory=ParallelScoreConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass(frozen=True)
class SolverConfig:
    batch_size: int = 1024
    max_iter: int = 500
    elite_fraction: float = 0.1
    seed: int | None = 0
    active_tol: float = 1e-6
    feasibility_tol: float = 1e-7
    gap_tol: float = 1e-6
    patience: int = 50
    time_limit_seconds: float | None = None
    archive_limit_multiplier: int = 5
    variance_collapse_factor: float = 1.0001
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    vertex_polishing: VertexPolishingConfig = field(default_factory=VertexPolishingConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
