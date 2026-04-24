from __future__ import annotations

from itertools import combinations

import numpy as np

from lpas.core.feasibility import is_primal_feasible
from lpas.core.lp_problem import LPProblem
from lpas.solver.result import ArchiveEntry, WarmStartHint
from lpas.utils.config import WarmStartConfig


def reconstruct_from_active_set(
    problem: LPProblem,
    active_mask: np.ndarray,
    *,
    config: WarmStartConfig | None = None,
) -> WarmStartHint:
    cfg = config or WarmStartConfig()
    active_indices = tuple(int(i) for i in np.flatnonzero(np.asarray(active_mask, dtype=bool)))
    if len(active_indices) < problem.n:
        return WarmStartHint(
            candidate_x=None,
            active_constraint_indices=active_indices,
            rank=0,
            feasible=False,
            objective=None,
            message="Not enough active constraints to reconstruct a vertex",
            constraint_system="original",
        )
    tested = 0
    for subset in combinations(active_indices, problem.n):
        tested += 1
        if tested > cfg.max_combinations:
            break
        A_active = problem.A[list(subset)]
        b_active = problem.b[list(subset)]
        rank = int(np.linalg.matrix_rank(A_active))
        if rank < problem.n:
            continue
        try:
            candidate = np.linalg.solve(A_active, b_active)
        except np.linalg.LinAlgError:
            continue
        feasible = is_primal_feasible(problem, candidate, tol=cfg.feasibility_tol)
        if feasible:
            objective = float(problem.c @ candidate)
            return WarmStartHint(
                candidate_x=candidate,
                active_constraint_indices=tuple(subset),
                rank=rank,
                feasible=True,
                objective=objective,
                message="Feasible vertex reconstructed from active constraints",
                constraint_system="original",
            )
    return WarmStartHint(
        candidate_x=None,
        active_constraint_indices=active_indices,
        rank=0,
        feasible=False,
        objective=None,
        message="No feasible full-rank active subset found",
        constraint_system="original",
    )


def reconstruct_from_archive(
    problem: LPProblem,
    archive: list[ArchiveEntry],
    *,
    config: WarmStartConfig | None = None,
) -> WarmStartHint:
    if not archive:
        return WarmStartHint(
            candidate_x=None,
            active_constraint_indices=(),
            rank=0,
            feasible=False,
            objective=None,
            message="Archive is empty",
            constraint_system="original",
        )
    counts: dict[tuple[bool, ...], int] = {}
    for entry in archive:
        key = tuple(bool(v) for v in entry.primal_active_mask)
        counts[key] = counts.get(key, 0) + 1
    dominant_key = max(counts.items(), key=lambda item: item[1])[0]
    return reconstruct_from_active_set(problem, np.asarray(dominant_key, dtype=bool), config=config)
