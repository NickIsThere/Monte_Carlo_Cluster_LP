from __future__ import annotations

from lpas.backends.base import (
    BackendInfo,
    CandidateBackend,
    VALID_BACKENDS,
    concatenate_metric_dicts,
    elite_count_from_fraction,
    slice_metric_dict,
    validate_backend_name,
)


def create_backend(backend: str, *, dtype: str = "float32") -> CandidateBackend:
    validate_backend_name(backend)
    if backend == "numpy_cpu":
        from lpas.backends.numpy_backend import NumpyBackend

        return NumpyBackend(backend, dtype=dtype)
    if backend == "numba_cpu":
        from lpas.backends.numba_backend import NumbaBackend

        return NumbaBackend(backend, dtype=dtype)
    from lpas.backends.torch_backend import TorchBackend

    return TorchBackend(backend, dtype=dtype)


__all__ = [
    "BackendInfo",
    "CandidateBackend",
    "VALID_BACKENDS",
    "concatenate_metric_dicts",
    "create_backend",
    "elite_count_from_fraction",
    "slice_metric_dict",
    "validate_backend_name",
]
