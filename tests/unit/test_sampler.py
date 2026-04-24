from __future__ import annotations

import numpy as np

from lpas.experiments.generators import random_feasible_lp
from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.sampling.truncated_sampler import TruncatedSampler
from lpas.utils.config import SamplerConfig


def test_sample_shapes_and_nonnegativity() -> None:
    sampler = GaussianAdaptiveSampler(3, 2, SamplerConfig(seed=1))
    X = sampler.sample_primal(5)
    Y = sampler.sample_dual(7)
    assert X.shape == (5, 3)
    assert Y.shape == (7, 2)
    assert np.all(X >= 0.0)
    assert np.all(Y >= 0.0)


def test_update_moves_mean_toward_elites() -> None:
    sampler = GaussianAdaptiveSampler(2, 2, SamplerConfig(seed=1, primal_init_mean=0.0, dual_init_mean=0.0, alpha=0.0))
    sampler.update(np.array([[2.0, 2.0], [4.0, 4.0]]), np.array([[1.0, 1.0], [3.0, 3.0]]))
    np.testing.assert_allclose(sampler.primal_mean, np.array([3.0, 3.0]))
    np.testing.assert_allclose(sampler.dual_mean, np.array([2.0, 2.0]))


def test_variance_floor_prevents_collapse() -> None:
    sampler = GaussianAdaptiveSampler(2, 2, SamplerConfig(seed=1, sigma_min=0.5, alpha=0.0))
    sampler.update(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))
    assert np.all(sampler.primal_sigma >= 0.5)
    assert np.all(sampler.dual_sigma >= 0.5)


def test_fixed_seed_is_reproducible() -> None:
    cfg = SamplerConfig(seed=123)
    sampler_a = GaussianAdaptiveSampler(2, 2, cfg)
    sampler_b = GaussianAdaptiveSampler(2, 2, cfg)
    np.testing.assert_allclose(sampler_a.sample_primal(4), sampler_b.sample_primal(4))
    np.testing.assert_allclose(sampler_a.sample_dual(4), sampler_b.sample_dual(4))


def test_no_nan_values_after_many_updates() -> None:
    sampler = GaussianAdaptiveSampler(2, 2, SamplerConfig(seed=1))
    elite = np.array([[1.0, 2.0], [2.0, 1.0]])
    for _ in range(50):
        sampler.update(elite, elite)
    assert np.all(np.isfinite(sampler.primal_mean))
    assert np.all(np.isfinite(sampler.primal_sigma))


def test_sampler_handles_tiny_elite_set() -> None:
    sampler = GaussianAdaptiveSampler(2, 2, SamplerConfig(seed=1))
    sampler.update(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
    assert sampler.primal_mean.shape == (2,)


def test_truncated_sampler_returns_batch_size() -> None:
    problem = random_feasible_lp(2, 3, seed=0)
    sampler = TruncatedSampler(problem, config=SamplerConfig(seed=0, max_retries=2, primal_violation_threshold=0.1, dual_violation_threshold=0.1))
    X = sampler.sample_primal(8)
    Y = sampler.sample_dual(8)
    assert X.shape == (8, 2)
    assert Y.shape == (8, 3)
