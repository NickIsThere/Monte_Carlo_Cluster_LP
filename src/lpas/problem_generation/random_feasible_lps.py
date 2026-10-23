from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.solver.scipy_handoff import ScipySolveResult, solve_with_scipy
from lpas.utils.random import make_rng


@dataclass(frozen=True)
class GeneratedFeasibleLP:
    name: str
    seed: int
    problem: LPProblem
    feasible_point: np.ndarray
    upper_bounds: np.ndarray
    scipy_reference: ScipySolveResult


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return matrix / safe_norms


def generate_random_feasible_bounded_lp(
    n_variables: int,
    n_constraints: int,
    seed: int,
    *,
    max_attempts: int = 64,
) -> GeneratedFeasibleLP:
    if n_variables <= 0:
        raise ValueError("n_variables must be positive")
    if n_constraints < n_variables:
        raise ValueError("n_constraints must be at least n_variables to allow upper bounds")

    n = int(n_variables)
    m = int(n_constraints)
    base_constraints = max(m - n, 0)
    rng = make_rng(seed)

    for attempt in range(max_attempts):
        attempt_seed = int(seed + attempt)
        local_rng = make_rng(attempt_seed)
        upper_bounds = local_rng.uniform(1.0, 3.0, size=n)
        feasible_point = upper_bounds * local_rng.uniform(0.2, 0.8, size=n)

        A_base = local_rng.normal(loc=0.0, scale=1.0, size=(base_constraints, n)) if base_constraints > 0 else np.empty((0, n))
        if base_constraints > 0:
            A_base = _normalize_rows(A_base)
            positive_slack = local_rng.uniform(0.1, 1.0, size=base_constraints)
            b_base = A_base @ feasible_point + positive_slack
        else:
            b_base = np.empty(0, dtype=float)

        A = np.vstack([A_base, np.eye(n, dtype=float)])
        b = np.concatenate([b_base, upper_bounds])
        c = local_rng.uniform(0.2, 1.5, size=n)
        problem = LPProblem(A=A, b=b, c=c, sense="max")
        reference = solve_with_scipy(problem)
        if not reference.success or reference.x is None or reference.objective is None:
            continue
        if np.linalg.norm(reference.x, ord=np.inf) <= 1e-6:
            continue
        if reference.objective <= float(c @ feasible_point) + 1e-6:
            continue
        return GeneratedFeasibleLP(
            name=f"scaling_n{n}_m{m}_s{seed}",
            seed=seed,
            problem=problem,
            feasible_point=np.asarray(feasible_point, dtype=float),
            upper_bounds=np.asarray(upper_bounds, dtype=float),
            scipy_reference=reference,
        )

    raise RuntimeError(
        f"failed to generate a feasible, bounded, nontrivial LP for n={n} m={m} after {max_attempts} attempts"
    )
