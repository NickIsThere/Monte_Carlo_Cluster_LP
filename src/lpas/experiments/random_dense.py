from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.utils.random import make_rng


@dataclass(frozen=True)
class DenseLPInstance:
    name: str
    family: str
    seed: int
    problem: LPProblem
    feasible_point: np.ndarray
    planted_optimum: np.ndarray | None = None
    planted_active_mask: np.ndarray | None = None


def _positive_full_rank_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    for _ in range(256):
        matrix = rng.uniform(0.15, 1.6, size=(size, size))
        if np.linalg.matrix_rank(matrix) == size:
            return matrix
    raise RuntimeError("failed to generate a full-rank dense positive matrix")


def generate_bounded_dense_lp(n_variables: int, n_constraints: int, seed: int, *, name: str | None = None) -> DenseLPInstance:
    rng = make_rng(seed)
    m = max(int(n_constraints), int(n_variables))
    n = int(n_variables)
    feasible_point = rng.uniform(0.25, 1.5, size=n)
    A = rng.uniform(0.2, 1.5, size=(m, n))
    b = A @ feasible_point + rng.uniform(0.3, 1.3, size=m)
    c = rng.uniform(0.2, 1.2, size=n)
    return DenseLPInstance(
        name=name or f"bounded_dense_n{n}_m{m}_s{seed}",
        family="bounded_dense",
        seed=seed,
        problem=LPProblem(A=A, b=b, c=c, sense="max"),
        feasible_point=feasible_point,
    )


def generate_controlled_optimum_lp(
    n_variables: int,
    n_constraints: int,
    seed: int,
    *,
    name: str | None = None,
) -> DenseLPInstance:
    rng = make_rng(seed)
    n = int(n_variables)
    m = max(int(n_constraints), n)
    x_star = rng.uniform(0.35, 1.4, size=n)
    A_active = _positive_full_rank_matrix(rng, n)
    y_star = rng.uniform(0.3, 1.2, size=n)
    c = A_active.T @ y_star

    extra_count = m - n
    if extra_count > 0:
        A_extra = rng.uniform(0.2, 1.6, size=(extra_count, n))
        b_extra = A_extra @ x_star + rng.uniform(0.2, 1.0, size=extra_count)
        A = np.vstack([A_active, A_extra])
        b = np.concatenate([A_active @ x_star, b_extra])
    else:
        A = A_active
        b = A_active @ x_star

    active_mask = np.zeros(m, dtype=bool)
    active_mask[:n] = True
    permutation = rng.permutation(m)
    A = A[permutation]
    b = b[permutation]
    active_mask = active_mask[permutation]

    return DenseLPInstance(
        name=name or f"controlled_optimum_n{n}_m{m}_s{seed}",
        family="controlled_optimum",
        seed=seed,
        problem=LPProblem(A=A, b=b, c=c, sense="max"),
        feasible_point=x_star,
        planted_optimum=x_star,
        planted_active_mask=active_mask,
    )


def generate_narrow_region_lp(n_variables: int, n_constraints: int, seed: int, *, name: str | None = None) -> DenseLPInstance:
    rng = make_rng(seed)
    n = int(n_variables)
    m = max(int(n_constraints), n)
    x_star = rng.uniform(0.25, 1.0, size=n)
    A_active = _positive_full_rank_matrix(rng, n)
    y_star = rng.uniform(0.4, 1.1, size=n)
    c = A_active.T @ y_star

    extra_count = m - n
    extras = []
    rhs = []
    for index in range(extra_count):
        base = A_active[index % n]
        perturbation = rng.normal(loc=0.0, scale=0.03, size=n)
        row = np.clip(base + perturbation, 0.05, None)
        extras.append(row)
        rhs.append(float(row @ x_star + rng.uniform(0.015, 0.12)))

    if extras:
        A = np.vstack([A_active, np.asarray(extras, dtype=float)])
        b = np.concatenate([A_active @ x_star, np.asarray(rhs, dtype=float)])
    else:
        A = A_active
        b = A_active @ x_star

    active_mask = np.zeros(m, dtype=bool)
    active_mask[:n] = True
    permutation = rng.permutation(m)
    A = A[permutation]
    b = b[permutation]
    active_mask = active_mask[permutation]

    return DenseLPInstance(
        name=name or f"narrow_region_n{n}_m{m}_s{seed}",
        family="narrow_region",
        seed=seed,
        problem=LPProblem(A=A, b=b, c=c, sense="max"),
        feasible_point=x_star,
        planted_optimum=x_star,
        planted_active_mask=active_mask,
    )


def build_random_dense_suite(
    *,
    n_variables: list[int],
    n_constraints: list[int],
    n_instances: int,
    seed: int = 0,
) -> list[DenseLPInstance]:
    rng = make_rng(seed)
    generators: list[tuple[str, Callable[..., DenseLPInstance]]] = [
        ("bounded_dense", generate_bounded_dense_lp),
        ("controlled_optimum", generate_controlled_optimum_lp),
        ("narrow_region", generate_narrow_region_lp),
    ]
    suite: list[DenseLPInstance] = []

    for n in n_variables:
        for m in n_constraints:
            if m < n:
                continue
            for instance_index in range(int(n_instances)):
                family_name, generator = generators[instance_index % len(generators)]
                instance_seed = int(rng.integers(0, 1_000_000))
                instance_name = f"{family_name}_n{n}_m{m}_i{instance_index}"
                suite.append(
                    generator(
                        n_variables=n,
                        n_constraints=m,
                        seed=instance_seed,
                        name=instance_name,
                    )
                )
    return suite
