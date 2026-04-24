from __future__ import annotations

from typing import Any

import numpy as np


def as_float_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def as_matrix(value: Any, *, name: str) -> np.ndarray:
    return as_float_array(value, ndim=2, name=name)


def as_vector(value: Any, *, name: str) -> np.ndarray:
    return as_float_array(value, ndim=1, name=name)


def validate_lp_dimensions(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError("A must have nonzero dimensions")
    if b.shape != (A.shape[0],):
        raise ValueError(f"b must have shape {(A.shape[0],)}, got {b.shape}")
    if c.shape != (A.shape[1],):
        raise ValueError(f"c must have shape {(A.shape[1],)}, got {c.shape}")


def ensure_batch(matrix: np.ndarray, *, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape ({expected_dim},) or (batch, {expected_dim})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
