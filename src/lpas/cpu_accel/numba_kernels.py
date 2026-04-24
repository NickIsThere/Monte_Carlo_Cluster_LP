from __future__ import annotations

from typing import Any

import numpy as np

try:
    import numba as nb
except ModuleNotFoundError:  # pragma: no cover - exercised via backend availability tests
    nb = None


if nb is not None:

    @nb.njit(parallel=True, fastmath=False)
    def _evaluate_primal_dual_batch_numba(
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        active_epsilon: float,
    ) -> tuple[np.ndarray, ...]:
        K = X.shape[0]
        m, n = A.shape
        primal_objective = np.zeros(K, dtype=X.dtype)
        dual_objective = np.zeros(K, dtype=X.dtype)
        gap = np.zeros(K, dtype=X.dtype)
        primal_violation = np.zeros(K, dtype=X.dtype)
        dual_violation = np.zeros(K, dtype=X.dtype)
        complementarity = np.zeros(K, dtype=X.dtype)
        active_count = np.zeros(K, dtype=np.int64)

        for k in nb.prange(K):
            primal_obj = 0.0
            dual_obj = 0.0
            primal_viol = 0.0
            dual_viol = 0.0
            comp = 0.0
            active = 0

            for j in range(n):
                x_val = X[k, j]
                primal_obj += x_val * c[j]
                if x_val < 0.0:
                    primal_viol += -x_val

            for i in range(m):
                y_val = Y[k, i]
                dual_obj += y_val * b[i]
                if y_val < 0.0:
                    dual_viol += -y_val

                ax = 0.0
                for j in range(n):
                    ax += X[k, j] * A[i, j]
                slack = b[i] - ax
                if slack <= active_epsilon:
                    active += 1
                if slack < 0.0:
                    primal_viol += -slack
                comp += abs(y_val * slack)

            for j in range(n):
                ya = 0.0
                for i in range(m):
                    ya += Y[k, i] * A[i, j]
                dual_slack = ya - c[j]
                if dual_slack < 0.0:
                    dual_viol += -dual_slack

            primal_objective[k] = primal_obj
            dual_objective[k] = dual_obj
            gap[k] = dual_obj - primal_obj
            primal_violation[k] = primal_viol
            dual_violation[k] = dual_viol
            complementarity[k] = comp
            active_count[k] = active

        return (
            primal_objective,
            dual_objective,
            gap,
            primal_violation,
            dual_violation,
            complementarity,
            active_count,
        )


def evaluate_primal_dual_batch_numba(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    active_epsilon: float = 1e-5,
) -> dict[str, Any]:
    if nb is None:
        raise RuntimeError("The numba_cpu backend was requested, but Numba is not installed.")
    (
        primal_objective,
        dual_objective,
        gap,
        primal_violation,
        dual_violation,
        complementarity,
        active_count,
    ) = _evaluate_primal_dual_batch_numba(A, b, c, X, Y, active_epsilon)
    return {
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "gap": gap,
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "complementarity": complementarity,
        "active_count": active_count,
        "primal_slack": None,
        "dual_slack": None,
        "active_mask": None,
    }
