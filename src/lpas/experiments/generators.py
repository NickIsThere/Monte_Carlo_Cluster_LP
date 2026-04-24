from __future__ import annotations

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.random import make_rng


def tiny_known_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        b=np.array([4.0, 2.0, 3.0]),
        c=np.array([3.0, 2.0]),
        sense="max",
    )


def degenerate_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 1.0]]),
        b=np.array([1.0]),
        c=np.array([1.0, 1.0]),
        sense="max",
    )


def thin_feasible_region_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        b=np.array([1.0, -0.999]),
        c=np.array([1.0, 0.0]),
        sense="max",
    )


def infeasible_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0], [-1.0]]),
        b=np.array([0.0, -1.0]),
        c=np.array([1.0]),
        sense="max",
    )


def unbounded_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[-1.0]]),
        b=np.array([0.0]),
        c=np.array([1.0]),
        sense="max",
    )


def random_feasible_lp(n: int, m: int, seed: int | None = 0) -> LPProblem:
    rng = make_rng(seed)
    m = max(m, n)
    x_feasible = rng.uniform(0.0, 2.0, size=n)
    identity = np.eye(n)
    extra_count = m - n
    extra = rng.uniform(0.1, 1.5, size=(extra_count, n)) if extra_count > 0 else np.empty((0, n))
    A = np.vstack([identity, extra])
    slack = rng.uniform(0.5, 2.0, size=m)
    b = A @ x_feasible + slack
    c = rng.uniform(0.5, 2.0, size=n)
    return LPProblem(A=A, b=b, c=c, sense="max")


def structured_benchmark_lp(n: int, extra_constraints: int = 1, seed: int | None = 0) -> LPProblem:
    rng = make_rng(seed)
    upper_bounds = rng.uniform(1.0, 3.0, size=n)
    box = np.eye(n)
    extras = []
    rhs = list(upper_bounds)
    for _ in range(extra_constraints):
        weights = rng.uniform(0.3, 1.5, size=n)
        scale = float(rng.uniform(0.55, 0.85))
        extras.append(weights)
        rhs.append(scale * float(weights @ upper_bounds))
    A = np.vstack([box, np.asarray(extras, dtype=float)]) if extras else box
    b = np.asarray(rhs, dtype=float)
    c = rng.uniform(0.5, 2.0, size=n)
    return LPProblem(A=A, b=b, c=c, sense="max")


def benchmark_lp_suite(seed: int = 0, count: int = 10) -> list[LPProblem]:
    problems = [tiny_known_lp(), degenerate_lp(), thin_feasible_region_lp()]
    rng = make_rng(seed)
    while len(problems) < count:
        index = len(problems)
        n = 2 + (index % 2)
        problems.append(
            structured_benchmark_lp(
                n=n,
                extra_constraints=0,
                seed=int(rng.integers(0, 1_000_000)),
            )
        )
    return problems[:count]
