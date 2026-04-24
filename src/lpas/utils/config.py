from __future__ import annotations

from dataclasses import dataclass, field


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
