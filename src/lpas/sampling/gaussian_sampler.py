from __future__ import annotations

import numpy as np

from lpas.sampling.adaptive_distribution import DiagonalGaussianState
from lpas.sampling.base_sampler import BaseSampler
from lpas.utils.config import SamplerConfig
from lpas.utils.random import make_rng


class GaussianAdaptiveSampler(BaseSampler):
    def __init__(self, primal_dim: int, dual_dim: int, config: SamplerConfig | None = None) -> None:
        self.config = config or SamplerConfig()
        self.rng = make_rng(self.config.seed)
        self.primal_state = DiagonalGaussianState(
            mean=np.full(primal_dim, self.config.primal_init_mean, dtype=float),
            sigma=np.full(primal_dim, self.config.sigma_init, dtype=float),
            alpha=self.config.alpha,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )
        self.dual_state = DiagonalGaussianState(
            mean=np.full(dual_dim, self.config.dual_init_mean, dtype=float),
            sigma=np.full(dual_dim, self.config.sigma_init, dtype=float),
            alpha=self.config.alpha,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )

    def sample_primal(self, batch_size: int) -> np.ndarray:
        return np.maximum(self.primal_state.sample(self.rng, batch_size), 0.0)

    def sample_dual(self, batch_size: int) -> np.ndarray:
        return np.maximum(self.dual_state.sample(self.rng, batch_size), 0.0)

    def update(self, elite_primal: np.ndarray, elite_dual: np.ndarray, elite_scores: np.ndarray | None = None) -> None:
        if elite_primal.size == 0 or elite_dual.size == 0:
            return
        self.primal_state.update(np.maximum(np.asarray(elite_primal, dtype=float), 0.0))
        self.dual_state.update(np.maximum(np.asarray(elite_dual, dtype=float), 0.0))

    @property
    def primal_mean(self) -> np.ndarray:
        return self.primal_state.mean.copy()

    @property
    def primal_sigma(self) -> np.ndarray:
        return self.primal_state.sigma.copy()

    @property
    def dual_mean(self) -> np.ndarray:
        return self.dual_state.mean.copy()

    @property
    def dual_sigma(self) -> np.ndarray:
        return self.dual_state.sigma.copy()
