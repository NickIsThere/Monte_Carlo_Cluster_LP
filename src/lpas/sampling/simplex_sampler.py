from __future__ import annotations

import numpy as np

from lpas.sampling.base_sampler import BaseSampler
from lpas.utils.random import make_rng


class SimplexSampler(BaseSampler):
    """Experimental nonnegative sampler for examples and ablations."""

    def __init__(self, primal_dim: int, dual_dim: int, seed: int | None = 0, scale: float = 1.0) -> None:
        self.primal_dim = primal_dim
        self.dual_dim = dual_dim
        self.scale = scale
        self.rng = make_rng(seed)

    def _sample_dirichlet_scaled(self, batch_size: int, dim: int) -> np.ndarray:
        base = self.rng.dirichlet(np.ones(dim), size=batch_size)
        radial = self.rng.gamma(shape=2.0, scale=self.scale, size=(batch_size, 1))
        return base * radial * dim

    def sample_primal(self, batch_size: int) -> np.ndarray:
        return self._sample_dirichlet_scaled(batch_size, self.primal_dim)

    def sample_dual(self, batch_size: int) -> np.ndarray:
        return self._sample_dirichlet_scaled(batch_size, self.dual_dim)

    def update(self, elite_primal: np.ndarray, elite_dual: np.ndarray, elite_scores: np.ndarray | None = None) -> None:
        total = float(np.mean(np.sum(elite_primal, axis=1))) if elite_primal.size else self.scale
        self.scale = max(total / max(self.primal_dim, 1), 1e-6)
