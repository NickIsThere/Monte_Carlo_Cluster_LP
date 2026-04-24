from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.config import ParallelScoreConfig, ParallelSolverConfig

VALID_BACKENDS = frozenset(
    {
        "numpy_cpu",
        "torch_cpu",
        "torch_cuda",
        "torch_mps",
        "auto_torch",
        "numba_cpu",
    }
)
VALID_DTYPES = frozenset({"float32", "float64"})


@dataclass(frozen=True)
class BackendInfo:
    requested_backend: str
    resolved_backend: str
    device_name: str
    dtype: str


class CandidateBackend(ABC):
    def __init__(self, requested_backend: str, *, dtype: str) -> None:
        self.info = BackendInfo(
            requested_backend=requested_backend,
            resolved_backend=requested_backend,
            device_name="cpu",
            dtype=validate_dtype_name(dtype),
        )

    @abstractmethod
    def prepare_problem(self, problem: LPProblem) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_backend_array(self, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def initialize_distribution(self, problem: LPProblem, config: ParallelSolverConfig) -> Any:
        raise NotImplementedError

    @abstractmethod
    def sample_candidates(self, state: Any, sample_size: int) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_batch(self, problem_data: Any, X: Any, Y: Any, *, active_epsilon: float) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def score_batch(
        self,
        problem_data: Any,
        X: Any,
        Y: Any,
        metrics: dict[str, Any],
        weights: ParallelScoreConfig,
        *,
        active_frequency: Any | None,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def select_elites(self, X: Any, Y: Any, scores: Any, elite_count: int) -> tuple[Any, Any, Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_distribution(self, state: Any, elite_X: Any, elite_Y: Any, config: ParallelSolverConfig) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_active_statistics(
        self,
        problem_data: Any,
        elite_X: Any,
        elite_Y: Any,
        *,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def concatenate(self, values: Sequence[Any], *, axis: int = 0) -> Any:
        raise NotImplementedError

    @abstractmethod
    def take(self, value: Any, indices: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self, value: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def to_scalar(self, value: Any) -> float:
        raise NotImplementedError

    def sync(self) -> None:
        return None

    @property
    def requested_backend(self) -> str:
        return self.info.requested_backend

    @property
    def resolved_backend(self) -> str:
        return self.info.resolved_backend

    @property
    def device_name(self) -> str:
        return self.info.device_name

    @property
    def dtype_name(self) -> str:
        return self.info.dtype


def validate_backend_name(name: str) -> str:
    if name not in VALID_BACKENDS:
        allowed = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(f"Unsupported backend {name!r}. Expected one of: {allowed}")
    return name


def validate_dtype_name(dtype: str) -> str:
    if dtype not in VALID_DTYPES:
        allowed = ", ".join(sorted(VALID_DTYPES))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {allowed}")
    return dtype


def numpy_dtype_from_name(dtype: str) -> np.dtype:
    validate_dtype_name(dtype)
    return np.dtype(dtype)


def elite_count_from_fraction(total_samples: int, elite_fraction: float) -> int:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in the interval (0, 1]")
    return max(1, math.ceil(total_samples * elite_fraction))


def slice_metric_dict(backend: CandidateBackend, metrics: dict[str, Any], indices: Any) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            sliced[key] = None
            continue
        sliced[key] = backend.take(value, indices)
    return sliced


def concatenate_metric_dicts(backend: CandidateBackend, metric_batches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not metric_batches:
        raise ValueError("metric_batches must not be empty")
    merged: dict[str, Any] = {}
    keys = metric_batches[0].keys()
    for key in keys:
        values = [metrics[key] for metrics in metric_batches]
        if values[0] is None:
            merged[key] = None
            continue
        merged[key] = backend.concatenate(values, axis=0)
    return merged
