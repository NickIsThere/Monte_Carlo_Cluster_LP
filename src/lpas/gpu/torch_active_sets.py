from __future__ import annotations

from typing import Any

from lpas.backends.torch_device import _require_torch


def active_mask_from_slack_torch(primal_slack: Any, *, active_epsilon: float) -> Any:
    return primal_slack <= active_epsilon


def active_count_torch(active_mask: Any) -> Any:
    return active_mask.sum(dim=1)


def active_frequency_torch(elite_active_masks: Any) -> Any:
    return elite_active_masks.float().mean(dim=0)


def active_support_reward_torch(active_mask: Any, active_frequency: Any | None) -> Any:
    torch = _require_torch()
    if active_frequency is None:
        return torch.zeros(active_mask.shape[0], device=active_mask.device, dtype=torch.float32)
    return (active_mask.float() * active_frequency.unsqueeze(0)).sum(dim=1)


def dual_active_agreement_torch(
    Y: Any,
    active_mask: Any,
    *,
    dual_positive_epsilon: float,
) -> tuple[Any, Any]:
    dual_positive = Y > dual_positive_epsilon
    agreement = (dual_positive & active_mask).float().sum(dim=1)
    conflict = (dual_positive & ~active_mask).float().sum(dim=1)
    return agreement, conflict


def active_frequency_entropy_torch(active_frequency: Any) -> Any:
    torch = _require_torch()
    clipped = active_frequency.clamp(min=1e-12, max=1.0 - 1e-12)
    entropy = -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
    return entropy.mean()
