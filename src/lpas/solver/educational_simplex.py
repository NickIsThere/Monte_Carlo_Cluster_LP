from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.solver.vertex_polishing import reconstruct_vertex_from_active_set


@dataclass(frozen=True)
class EducationalSimplexResult:
    status: str
    x: np.ndarray | None
    objective: float | None
    basis: tuple[int, ...] | None
    iterations: int
    visited_bases: int
    runtime_seconds: float


def origin_basis(problem: LPProblem) -> tuple[int, ...]:
    canonical_problem = problem.to_maximization()
    return tuple(range(canonical_problem.m, canonical_problem.m + canonical_problem.n))


def solve_educational_simplex(
    problem: LPProblem,
    *,
    initial_basis: tuple[int, ...] | None = None,
    max_iterations: int = 128,
    improvement_tol: float = 1e-10,
) -> EducationalSimplexResult:
    canonical_problem = problem.to_maximization()
    current_basis = tuple(sorted(initial_basis or origin_basis(canonical_problem)))
    current_vertex = reconstruct_vertex_from_active_set(canonical_problem, current_basis)
    if current_vertex is None or not current_vertex.feasible:
        fallback_basis = origin_basis(canonical_problem)
        current_basis = tuple(sorted(fallback_basis))
        current_vertex = reconstruct_vertex_from_active_set(canonical_problem, current_basis)
    start = time.perf_counter()
    if current_vertex is None or not current_vertex.feasible:
        return EducationalSimplexResult(
            status="no_feasible_basis",
            x=None,
            objective=None,
            basis=None,
            iterations=0,
            visited_bases=0,
            runtime_seconds=time.perf_counter() - start,
        )

    visited = {current_basis}
    iterations = 0
    total_constraints = canonical_problem.m + canonical_problem.n
    current_objective = float(current_vertex.objective)

    while iterations < max_iterations:
        best_neighbor_basis: tuple[int, ...] | None = None
        best_neighbor_vertex = None
        basis_set = set(current_basis)
        nonbasic = [index for index in range(total_constraints) if index not in basis_set]
        for leaving_index in range(len(current_basis)):
            retained = list(current_basis)
            del retained[leaving_index]
            for entering in nonbasic:
                candidate_basis = tuple(sorted(retained + [int(entering)]))
                if candidate_basis in visited:
                    continue
                visited.add(candidate_basis)
                candidate_vertex = reconstruct_vertex_from_active_set(canonical_problem, candidate_basis)
                if candidate_vertex is None or not candidate_vertex.feasible:
                    continue
                if candidate_vertex.objective > current_objective + improvement_tol:
                    if best_neighbor_vertex is None or candidate_vertex.objective > best_neighbor_vertex.objective:
                        best_neighbor_basis = candidate_basis
                        best_neighbor_vertex = candidate_vertex
        if best_neighbor_basis is None or best_neighbor_vertex is None:
            break
        current_basis = best_neighbor_basis
        current_vertex = best_neighbor_vertex
        current_objective = float(current_vertex.objective)
        iterations += 1

    return EducationalSimplexResult(
        status="optimal",
        x=np.asarray(current_vertex.x, dtype=float),
        objective=float(current_vertex.objective),
        basis=current_basis,
        iterations=iterations,
        visited_bases=len(visited),
        runtime_seconds=time.perf_counter() - start,
    )
