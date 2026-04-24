from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("A Torch backend was requested, but PyTorch is not installed.") from exc
    return torch


def resolve_torch_device(backend: str):
    torch = _require_torch()
    if backend == "torch_cpu":
        return torch.device("cpu")
    if backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested backend torch_cuda, but CUDA is not available.")
        return torch.device("cuda")
    if backend == "torch_mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("Requested backend torch_mps, but MPS is not available.")
        return torch.device("mps")
    if backend == "auto_torch":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported torch backend {backend!r}")


def torch_dtype_from_name(dtype: str, device) -> Any:
    torch = _require_torch()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        if device.type == "mps":
            raise ValueError("The torch_mps backend only supports float32 in this solver.")
        return torch.float64
    raise ValueError(f"Unsupported Torch dtype {dtype!r}")


def torch_device_name(device) -> str:
    torch = _require_torch()
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


def synchronize_torch_device(device) -> None:
    torch = _require_torch()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
