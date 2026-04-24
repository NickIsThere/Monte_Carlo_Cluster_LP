from __future__ import annotations

import numpy as np
import pytest

from lpas.core.lp_problem import LPProblem
from lpas.experiments.generators import degenerate_lp, infeasible_lp, thin_feasible_region_lp, tiny_known_lp, unbounded_lp
from lpas.utils.config import SamplerConfig, ScoringConfig, SolverConfig


@pytest.fixture
def small_lp() -> LPProblem:
    return tiny_known_lp()


@pytest.fixture
def degenerate_problem() -> LPProblem:
    return degenerate_lp()


@pytest.fixture
def infeasible_problem() -> LPProblem:
    return infeasible_lp()


@pytest.fixture
def unbounded_problem() -> LPProblem:
    return unbounded_lp()


@pytest.fixture
def thin_problem() -> LPProblem:
    return thin_feasible_region_lp()


@pytest.fixture
def fast_solver_config() -> SolverConfig:
    sampler = SamplerConfig(
        seed=42,
        alpha=0.6,
        sigma_init=2.5,
        sigma_min=1e-5,
        sigma_max=8.0,
        primal_init_mean=0.75,
        dual_init_mean=0.75,
        max_retries=6,
        primal_violation_threshold=0.5,
        dual_violation_threshold=0.5,
    )
    scoring = ScoringConfig(
        w_primal_obj=1.0,
        w_dual_obj=0.25,
        w_gap=1.5,
        w_pviol=2.0,
        w_dviol=2.0,
        w_comp=1.0,
        w_geo=0.15,
        w_active=0.15,
        geometry_sigma=1.5,
        geometry_dual_weight=0.5,
    )
    return SolverConfig(
        batch_size=512,
        max_iter=80,
        elite_fraction=0.1,
        seed=42,
        active_tol=1e-2,
        feasibility_tol=1e-7,
        gap_tol=1e-5,
        patience=20,
        scoring=scoring,
        sampler=sampler,
    )


@pytest.fixture
def benchmark_solver_config() -> SolverConfig:
    sampler = SamplerConfig(
        seed=7,
        alpha=0.6,
        sigma_init=2.5,
        sigma_min=1e-5,
        sigma_max=8.0,
        primal_init_mean=0.75,
        dual_init_mean=0.75,
    )
    return SolverConfig(
        batch_size=512,
        max_iter=80,
        elite_fraction=0.1,
        seed=7,
        active_tol=1e-2,
        feasibility_tol=1e-7,
        gap_tol=1e-5,
        patience=20,
        sampler=sampler,
    )


@pytest.fixture
def optimal_pair() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([2.0, 2.0])
    y = np.array([2.0, 1.0, 0.0])
    return x, y
