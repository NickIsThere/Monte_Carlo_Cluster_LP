from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lpas.core.feasibility import dual_violation_norm, is_dual_feasible, is_primal_feasible, primal_violation_norm
from lpas.core.lp_problem import LPProblem
from lpas.solver.result import PolishedVertex, ScipySolveResult


@dataclass(frozen=True)
class BoundCertificate:
    raw_primal_objective: float | None
    raw_dual_objective: float | None
    raw_gap: float | None
    raw_sampled_gap: float | None
    raw_primal_violation: float | None
    raw_dual_violation: float | None
    raw_gap_is_diagnostic_only: bool

    best_feasible_primal_lower_bound: float | None
    best_feasible_primal_x: np.ndarray | None
    primal_feasible: bool
    primal_violation: float

    best_feasible_dual_upper_bound: float | None
    best_feasible_dual_y: np.ndarray | None
    dual_feasible: bool
    dual_violation: float

    certified_gap: float | None
    certified_relative_gap: float | None
    has_valid_certificate: bool

    polished_vertex_objective: float | None
    polished_vertex_x: np.ndarray | None
    polished_vertex_feasible: bool
    best_polished_feasible_vertex_objective: float | None
    best_polished_feasible_vertex_violation: float | None

    scipy_objective: float | None
    scipy_certified_objective: float | None
    scipy_x: np.ndarray | None
    scipy_status: str | None


def _objective_or_default(value: float | None, fallback: float | None) -> float | None:
    return fallback if value is None else float(value)


def _primal_objective(problem: LPProblem, x: np.ndarray | None, objective: float | None) -> float | None:
    if objective is not None:
        return float(objective)
    if x is None:
        return None
    return _objective_or_default(objective, problem.maximization_objective_value(x))


def _dual_objective(problem: LPProblem, y: np.ndarray | None, objective: float | None) -> float | None:
    if objective is not None:
        return float(objective)
    if y is None:
        return None
    return _objective_or_default(objective, float(problem.b @ np.asarray(y, dtype=float)))


def _relative_gap(lower_bound: float, upper_bound: float) -> float:
    denominator = max(1.0, abs(lower_bound), abs(upper_bound))
    return float((upper_bound - lower_bound) / denominator)


def build_bound_certificate(
    problem: LPProblem,
    *,
    raw_x: np.ndarray | None = None,
    raw_y: np.ndarray | None = None,
    raw_primal_objective: float | None = None,
    raw_dual_objective: float | None = None,
    primal_x: np.ndarray | None = None,
    primal_objective: float | None = None,
    dual_y: np.ndarray | None = None,
    dual_objective: float | None = None,
    polished_vertex: PolishedVertex | None = None,
    scipy_result: ScipySolveResult | None = None,
    feasibility_tol: float = 1e-7,
) -> BoundCertificate:
    canonical_problem = problem.to_maximization()

    resolved_raw_primal_objective = _primal_objective(canonical_problem, raw_x, raw_primal_objective)
    resolved_raw_dual_objective = _dual_objective(canonical_problem, raw_y, raw_dual_objective)
    raw_primal_violation = None
    raw_dual_violation = None
    raw_primal_feasible = False
    raw_dual_feasible = False
    if raw_x is not None:
        raw_primal_violation = float(primal_violation_norm(canonical_problem, raw_x))
        raw_primal_feasible = bool(is_primal_feasible(canonical_problem, raw_x, tol=feasibility_tol))
    if raw_y is not None:
        raw_dual_violation = float(dual_violation_norm(canonical_problem, raw_y))
        raw_dual_feasible = bool(is_dual_feasible(canonical_problem, raw_y, tol=feasibility_tol))
    raw_gap = None
    if resolved_raw_primal_objective is not None and resolved_raw_dual_objective is not None:
        raw_gap = float(resolved_raw_dual_objective - resolved_raw_primal_objective)
    raw_gap_is_diagnostic_only = bool(not (raw_primal_feasible and raw_dual_feasible))

    resolved_primal_x = None if primal_x is None else np.asarray(primal_x, dtype=float)
    # Bound values must come from the vectors that are actually verified.
    resolved_primal_objective = None if resolved_primal_x is None else float(canonical_problem.c @ resolved_primal_x)
    primal_feasible = False
    primal_violation = math.inf
    if resolved_primal_x is not None:
        primal_feasible = bool(is_primal_feasible(canonical_problem, resolved_primal_x, tol=feasibility_tol))
        primal_violation = float(primal_violation_norm(canonical_problem, resolved_primal_x))
        if not primal_feasible:
            resolved_primal_x = None
            resolved_primal_objective = None

    resolved_dual_y = None if dual_y is None else np.asarray(dual_y, dtype=float)
    resolved_dual_objective = None if resolved_dual_y is None else float(canonical_problem.b @ resolved_dual_y)
    dual_feasible = False
    dual_violation = math.inf
    if resolved_dual_y is not None:
        dual_feasible = bool(is_dual_feasible(canonical_problem, resolved_dual_y, tol=feasibility_tol))
        dual_violation = float(dual_violation_norm(canonical_problem, resolved_dual_y))
        if not dual_feasible:
            resolved_dual_y = None
            resolved_dual_objective = None

    polished_vertex_objective = None if polished_vertex is None else float(polished_vertex.objective)
    polished_vertex_x = None if polished_vertex is None else np.asarray(polished_vertex.x, dtype=float)
    polished_vertex_feasible = bool(
        polished_vertex is not None and polished_vertex.feasible
        and is_primal_feasible(canonical_problem, polished_vertex.x, tol=feasibility_tol)
    )
    best_polished_feasible_vertex_objective = None
    best_polished_feasible_vertex_violation = None
    if polished_vertex_feasible and polished_vertex is not None:
        best_polished_feasible_vertex_objective = float(canonical_problem.c @ polished_vertex.x)
        best_polished_feasible_vertex_violation = float(polished_vertex.primal_violation)
        if resolved_primal_objective is None or best_polished_feasible_vertex_objective > resolved_primal_objective + 1e-12:
            resolved_primal_x = np.asarray(polished_vertex.x, dtype=float)
            resolved_primal_objective = best_polished_feasible_vertex_objective
            primal_feasible = True
            primal_violation = float(polished_vertex.primal_violation)

    certified_gap = None
    certified_relative_gap = None
    has_valid_certificate = False
    if primal_feasible and dual_feasible and resolved_primal_objective is not None and resolved_dual_objective is not None:
        candidate_gap = float(resolved_dual_objective - resolved_primal_objective)
        # Tolerance-feasible vectors can give slightly inverted bounds. Record
        # those vectors, but do not call a negative or non-finite gap certified.
        if math.isfinite(candidate_gap) and candidate_gap >= 0.0:
            certified_gap = candidate_gap
            certified_relative_gap = _relative_gap(resolved_primal_objective, resolved_dual_objective)
            has_valid_certificate = True

    scipy_status = None
    if scipy_result is not None:
        scipy_status = "success" if scipy_result.success else f"status_{scipy_result.status}"

    return BoundCertificate(
        raw_primal_objective=resolved_raw_primal_objective,
        raw_dual_objective=resolved_raw_dual_objective,
        raw_gap=raw_gap,
        raw_sampled_gap=raw_gap,
        raw_primal_violation=raw_primal_violation,
        raw_dual_violation=raw_dual_violation,
        raw_gap_is_diagnostic_only=raw_gap_is_diagnostic_only,
        best_feasible_primal_lower_bound=resolved_primal_objective,
        best_feasible_primal_x=resolved_primal_x,
        primal_feasible=primal_feasible,
        primal_violation=primal_violation,
        best_feasible_dual_upper_bound=resolved_dual_objective,
        best_feasible_dual_y=resolved_dual_y,
        dual_feasible=dual_feasible,
        dual_violation=dual_violation,
        certified_gap=certified_gap,
        certified_relative_gap=certified_relative_gap,
        has_valid_certificate=has_valid_certificate,
        polished_vertex_objective=polished_vertex_objective,
        polished_vertex_x=polished_vertex_x,
        polished_vertex_feasible=polished_vertex_feasible,
        best_polished_feasible_vertex_objective=best_polished_feasible_vertex_objective,
        best_polished_feasible_vertex_violation=best_polished_feasible_vertex_violation,
        scipy_objective=None if scipy_result is None else scipy_result.objective,
        scipy_certified_objective=None if scipy_result is None else scipy_result.objective,
        scipy_x=None if scipy_result is None or scipy_result.x is None else np.asarray(scipy_result.x, dtype=float),
        scipy_status=scipy_status,
    )
