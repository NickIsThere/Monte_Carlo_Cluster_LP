"""Diagonally preconditioned PDHG for max c'x, Ax <= b, x >= 0.

The saddle function is c'x + y'(b-Ax), maximized in x and minimized in y.
Absolute row/column sums give a safe diagonal step choice without an SVD:
||diag(sqrt(sigma)) A diag(sqrt(tau))||_2 <= step_scale < 1.
"""
from __future__ import annotations

import time

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.config import PDHGConfig
from lpas.utils.validation import ensure_batch


class PDHGRefiner:
    def __init__(self, problem: LPProblem, config: PDHGConfig | None = None) -> None:
        self.problem = problem.to_maximization()
        self.config = config or PDHGConfig()
        absolute = np.abs(self.problem.A)
        columns = absolute.sum(axis=0)
        rows = absolute.sum(axis=1)
        self.tau = self.config.step_scale / np.where(columns > 0.0, columns, 1.0)
        self.sigma = self.config.step_scale / np.where(rows > 0.0, rows, 1.0)

    def refine(
        self, X: np.ndarray, Y: np.ndarray, *, deadline: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        X = ensure_batch(X, expected_dim=self.problem.n, name="X").copy()
        Y = ensure_batch(Y, expected_dim=self.problem.m, name="Y").copy()
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same batch size")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("PDHG candidates must be finite")
        steps = 0
        for _ in range(self.config.steps_per_iteration):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            previous_X = X
            X = np.maximum(X + self.tau * (self.problem.c - Y @ self.problem.A), 0.0)
            extrapolated_X = 2.0 * X - previous_X
            Y = np.maximum(Y + self.sigma * (extrapolated_X @ self.problem.A.T - self.problem.b), 0.0)
            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
                raise FloatingPointError("PDHG produced non-finite candidates")
            steps += 1
        return X, Y, steps
