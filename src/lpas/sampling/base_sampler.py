from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSampler(ABC):
    @abstractmethod
    def sample_primal(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample_dual(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, elite_primal: np.ndarray, elite_dual: np.ndarray, elite_scores: np.ndarray | None = None) -> None:
        raise NotImplementedError
