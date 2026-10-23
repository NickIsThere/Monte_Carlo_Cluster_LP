"""Bounded primal recovery by most-violated-halfspace projections.

This does not solve an LP or optimize its objective. Every proposed point still
needs an independent feasibility check before it can supply a primal bound.
"""
from __future__ import annotations

import time

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.validation import ensure_batch


def recover_primal_points(
    problem: LPProblem, X: np.ndarray, *, max_steps: int = 256,
    tolerance: float = 1e-8, deadline: float | None = None,
) -> tuple[np.ndarray, int]:
    if max_steps <= 0 or not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Recovery requires positive max_steps and finite nonnegative tolerance")
    points = np.maximum(ensure_batch(X, expected_dim=problem.n, name="X").copy(), 0.0)
    squared_norms = np.sum(problem.A * problem.A, axis=1)
    if np.any((squared_norms == 0.0) & (problem.b < -tolerance)):
        return points, 0
    safe_norms = np.where(squared_norms > 0.0, squared_norms, 1.0)
    row_norms = np.sqrt(safe_norms)
    point_steps = 0
    for _ in range(max_steps):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        violations = points @ problem.A.T - problem.b
        active_points = np.flatnonzero(np.max(violations, axis=1) > tolerance)
        if active_points.size == 0:
            break
        selected_rows = np.argmax(violations[active_points] / row_norms, axis=1)
        excess = violations[active_points, selected_rows]
        # A small inward margin avoids returning just outside a boundary due to
        # rounding. Clipping is projection onto the nonnegative orthant.
        movement = (excess + tolerance * 0.1) / safe_norms[selected_rows]
        points[active_points] = np.maximum(
            points[active_points] - movement[:, None] * problem.A[selected_rows], 0.0,
        )
        point_steps += active_points.size
    return points, int(point_steps)
