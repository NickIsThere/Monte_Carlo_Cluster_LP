from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.experiments.random_dense import generate_controlled_optimum_lp
from lpas.utils.random import make_rng


@dataclass(frozen=True)
class ToyLPCase:
    name: str
    problem: LPProblem
    dimension: int
    axis_limits: tuple[float, ...]
    description: str


def triangle_unique_optimum() -> ToyLPCase:
    return ToyLPCase(
        name="triangle_unique",
        problem=LPProblem(
            A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
            b=np.array([4.0, 2.0, 3.0], dtype=float),
            c=np.array([3.0, 2.0], dtype=float),
            sense="max",
        ),
        dimension=2,
        axis_limits=(2.6, 3.4),
        description="Triangle-like feasible polygon with a clear optimal corner.",
    )


def trapezoid_case() -> ToyLPCase:
    return ToyLPCase(
        name="trapezoid",
        problem=LPProblem(
            A=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.4, 1.0],
                ],
                dtype=float,
            ),
            b=np.array([3.2, 2.8, 4.6, 3.1], dtype=float),
            c=np.array([2.5, 1.8], dtype=float),
            sense="max",
        ),
        dimension=2,
        axis_limits=(3.5, 3.0),
        description="Rectangle/trapezoid style region with several visible corners.",
    )


def flat_objective_case() -> ToyLPCase:
    return ToyLPCase(
        name="near_flat",
        problem=LPProblem(
            A=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.8, 1.0],
                ],
                dtype=float,
            ),
            b=np.array([3.0, 3.2, 4.9, 6.2], dtype=float),
            c=np.array([1.0, 0.98], dtype=float),
            sense="max",
        ),
        dimension=2,
        axis_limits=(3.3, 3.4),
        description="Nearly flat objective with multiple near-optimal points.",
    )


def random_polygon_case(seed: int = 0) -> ToyLPCase:
    rng = make_rng(seed)
    anchor = rng.uniform(0.7, 1.6, size=2)
    A = rng.uniform(0.15, 1.4, size=(6, 2))
    b = A @ anchor + rng.uniform(0.3, 1.1, size=6)
    c = rng.uniform(0.5, 1.5, size=2)
    limits = tuple(float(np.max(b / np.maximum(A[:, axis], 1e-6))) for axis in range(2))
    padded_limits = tuple(limit * 1.05 for limit in limits)
    return ToyLPCase(
        name="random_polygon",
        problem=LPProblem(A=A, b=b, c=c, sense="max"),
        dimension=2,
        axis_limits=padded_limits,
        description="Random bounded polygon-like LP with dense constraints.",
    )


def simple_3d_case(seed: int = 0) -> ToyLPCase:
    instance = generate_controlled_optimum_lp(n_variables=3, n_constraints=7, seed=seed, name="toy3d_controlled")
    axis_limits = tuple(float(np.max(instance.problem.b / np.maximum(instance.problem.A[:, axis], 1e-6))) * 1.05 for axis in range(3))
    return ToyLPCase(
        name="polytope_3d",
        problem=instance.problem,
        dimension=3,
        axis_limits=axis_limits,
        description="Simple 3D dense LP visualized through pairwise projections.",
    )


def default_toy_cases(seed: int = 0) -> list[ToyLPCase]:
    return [
        triangle_unique_optimum(),
        trapezoid_case(),
        flat_objective_case(),
        random_polygon_case(seed=seed),
        simple_3d_case(seed=seed),
    ]
