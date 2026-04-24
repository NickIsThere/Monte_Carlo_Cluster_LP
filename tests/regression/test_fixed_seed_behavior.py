from __future__ import annotations

import numpy as np

from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.utils.config import SamplerConfig


def test_fixed_seed_sampler_behavior_is_reproducible() -> None:
    config = SamplerConfig(seed=42)
    sampler_a = GaussianAdaptiveSampler(3, 2, config)
    sampler_b = GaussianAdaptiveSampler(3, 2, config)
    np.testing.assert_allclose(sampler_a.sample_primal(16), sampler_b.sample_primal(16))
    np.testing.assert_allclose(sampler_a.sample_dual(16), sampler_b.sample_dual(16))


def test_fixed_seed_solver_behavior_is_reproducible(small_lp, fast_solver_config) -> None:
    result_a = AdaptiveLPSolver(fast_solver_config).solve(small_lp)
    result_b = AdaptiveLPSolver(fast_solver_config).solve(small_lp)
    assert result_a.best_primal_objective == result_b.best_primal_objective
    np.testing.assert_allclose(result_a.best_x, result_b.best_x)
