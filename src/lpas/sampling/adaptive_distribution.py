from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiagonalGaussianState:
    mean: np.ndarray
    sigma: np.ndarray
    alpha: float
    sigma_min: float
    sigma_max: float

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return rng.normal(loc=self.mean, scale=self.sigma, size=(batch_size, self.mean.shape[0]))

    def update(self, elite_samples: np.ndarray) -> None:
        elite_samples = np.asarray(elite_samples, dtype=float)
        if elite_samples.ndim != 2 or elite_samples.shape[1] != self.mean.shape[0]:
            raise ValueError("elite_samples must match the distribution dimension")
        elite_mean = np.mean(elite_samples, axis=0)
        elite_sigma = np.std(elite_samples, axis=0, ddof=0)
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * elite_mean
        self.sigma = self.alpha * self.sigma + (1.0 - self.alpha) * elite_sigma
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
