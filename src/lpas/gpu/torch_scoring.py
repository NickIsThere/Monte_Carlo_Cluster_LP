from __future__ import annotations

from typing import Any

from lpas.backends.torch_device import _require_torch
from lpas.utils.config import ParallelScoreConfig


def evaluate_primal_dual_batch_torch(
    A: Any,
    b: Any,
    c: Any,
    X: Any,
    Y: Any,
    *,
    active_epsilon: float = 1e-5,
) -> dict[str, Any]:
    torch = _require_torch()
    AX = X @ A.transpose(0, 1)
    primal_slack = b.unsqueeze(0) - AX
    primal_violation = torch.relu(AX - b.unsqueeze(0)).sum(dim=1)
    primal_violation = primal_violation + torch.relu(-X).sum(dim=1)

    YA = Y @ A
    dual_slack = YA - c.unsqueeze(0)
    dual_violation = torch.relu(c.unsqueeze(0) - YA).sum(dim=1)
    dual_violation = dual_violation + torch.relu(-Y).sum(dim=1)

    primal_objective = X @ c
    dual_objective = Y @ b
    gap = dual_objective - primal_objective
    complementarity = torch.abs(Y * primal_slack).sum(dim=1)
    active_mask = primal_slack <= active_epsilon
    active_count = active_mask.sum(dim=1)
    return {
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "gap": gap,
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "complementarity": complementarity,
        "primal_slack": primal_slack,
        "dual_slack": dual_slack,
        "active_mask": active_mask,
        "active_count": active_count,
    }


def score_primal_dual_batch_torch(
    metrics: dict[str, Any],
    weights: ParallelScoreConfig,
    *,
    active_support: Any | None = None,
    active_agreement: Any | None = None,
    active_conflict: Any | None = None,
) -> Any:
    torch = _require_torch()
    score = weights.objective * metrics["primal_objective"]
    score = score - weights.gap * torch.relu(metrics["gap"])
    score = score - weights.primal_violation * metrics["primal_violation"]
    score = score - weights.dual_violation * metrics["dual_violation"]
    score = score - weights.complementarity * metrics["complementarity"]
    if active_support is not None:
        score = score + weights.active_support * active_support
    if active_agreement is not None:
        score = score + weights.active_agreement * active_agreement
    if active_conflict is not None:
        score = score - weights.active_conflict * active_conflict
    finite_min = torch.full_like(score, -1e12)
    finite_max = torch.full_like(score, 1e12)
    score = torch.where(torch.isnan(score), finite_min, score)
    score = torch.minimum(torch.maximum(score, finite_min), finite_max)
    return score
