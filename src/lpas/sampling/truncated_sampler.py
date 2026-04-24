from __future__ import annotations

import numpy as np

from lpas.core.feasibility import dual_violation_norm, primal_violation_norm
from lpas.core.lp_problem import LPProblem
from lpas.sampling.base_sampler import BaseSampler
from lpas.sampling.gaussian_sampler import GaussianAdaptiveSampler
from lpas.utils.config import SamplerConfig


class TruncatedSampler(BaseSampler):
    def __init__(
        self,
        problem: LPProblem,
        base_sampler: GaussianAdaptiveSampler | None = None,
        config: SamplerConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or SamplerConfig()
        self.base_sampler = base_sampler or GaussianAdaptiveSampler(problem.n, problem.m, self.config)

    def _sample_with_filter(self, batch_size: int, *, primal: bool) -> np.ndarray:
        accepted: list[np.ndarray] = []
        attempts = 0
        fallback = None
        while len(accepted) < batch_size and attempts < self.config.max_retries:
            remaining = batch_size - len(accepted)
            samples = self.base_sampler.sample_primal(remaining) if primal else self.base_sampler.sample_dual(remaining)
            fallback = samples
            if primal:
                mask = np.array(
                    [primal_violation_norm(self.problem, sample) <= self.config.primal_violation_threshold for sample in samples],
                    dtype=bool,
                )
            else:
                mask = np.array(
                    [dual_violation_norm(self.problem, sample) <= self.config.dual_violation_threshold for sample in samples],
                    dtype=bool,
                )
            accepted.extend(sample for sample in samples[mask])
            attempts += 1
        if len(accepted) >= batch_size:
            return np.asarray(accepted[:batch_size], dtype=float)
        if fallback is None:
            fallback = self.base_sampler.sample_primal(batch_size) if primal else self.base_sampler.sample_dual(batch_size)
        needed = batch_size - len(accepted)
        if needed > 0:
            accepted.extend(sample for sample in fallback[:needed])
        return np.asarray(accepted[:batch_size], dtype=float)

    def sample_primal(self, batch_size: int) -> np.ndarray:
        return self._sample_with_filter(batch_size, primal=True)

    def sample_dual(self, batch_size: int) -> np.ndarray:
        return self._sample_with_filter(batch_size, primal=False)

    def update(self, elite_primal: np.ndarray, elite_dual: np.ndarray, elite_scores: np.ndarray | None = None) -> None:
        self.base_sampler.update(elite_primal, elite_dual, elite_scores)
