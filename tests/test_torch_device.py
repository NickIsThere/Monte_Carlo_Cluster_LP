from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lpas.backends.torch_device import resolve_torch_device


class _MPSBackendStub:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _patch_mps_available(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        monkeypatch.setattr(torch.backends, "mps", _MPSBackendStub(available), raising=False)
        return
    monkeypatch.setattr(mps_backend, "is_available", lambda: available)


def test_torch_cpu_resolves_to_cpu() -> None:
    assert resolve_torch_device("torch_cpu").type == "cpu"


def test_torch_cuda_raises_if_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch_cuda"):
        resolve_torch_device("torch_cuda")


def test_torch_mps_raises_if_mps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mps_available(monkeypatch, False)
    with pytest.raises(RuntimeError, match="torch_mps"):
        resolve_torch_device("torch_mps")


def test_auto_torch_never_returns_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _patch_mps_available(monkeypatch, False)
    assert resolve_torch_device("auto_torch").type == "cpu"


def test_unavailable_gpu_does_not_silently_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_torch_device("torch_cuda")


def test_auto_torch_prefers_mps_over_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_available(monkeypatch, True)
    assert resolve_torch_device("auto_torch").type == "mps"


def test_auto_torch_ignores_cuda_when_mps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_available(monkeypatch, False)
    assert resolve_torch_device("auto_torch").type == "cpu"


def test_explicit_torch_cuda_is_unchanged_by_auto_torch_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_available(monkeypatch, False)
    assert resolve_torch_device("torch_cuda").type == "cuda"
