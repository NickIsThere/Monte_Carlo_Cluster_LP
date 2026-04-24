from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lpas.backends.torch_device import resolve_torch_device


def test_torch_cpu_resolves_to_cpu() -> None:
    assert resolve_torch_device("torch_cpu").type == "cpu"


def test_torch_cuda_raises_if_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch_cuda"):
        resolve_torch_device("torch_cuda")


def test_torch_mps_raises_if_mps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        with pytest.raises(RuntimeError, match="torch_mps"):
            resolve_torch_device("torch_mps")
        return
    monkeypatch.setattr(mps_backend, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch_mps"):
        resolve_torch_device("torch_mps")


def test_auto_torch_never_returns_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: False)
    assert resolve_torch_device("auto_torch").type == "cpu"


def test_unavailable_gpu_does_not_silently_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_torch_device("torch_cuda")
