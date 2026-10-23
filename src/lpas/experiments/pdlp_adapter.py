"""Optional PDLP bridge and independent checks for sparse, general-bound LPs.

This benchmark model deliberately does not widen the dense research solver's
LPProblem contract. All objectives here use minimization, in original units.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SparseLP:
    A: sparse.csr_matrix
    c: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    offset: float = 0.0
    name: str = "lp"

    def __post_init__(self) -> None:
        A = sparse.csr_matrix(self.A, dtype=np.float64, copy=True)
        A.sum_duplicates()
        A.eliminate_zeros()
        A.sort_indices()
        object.__setattr__(self, "A", A)
        m, n = A.shape
        for field, size in (("c", n), ("row_lower", m), ("row_upper", m),
                            ("variable_lower", n), ("variable_upper", n)):
            value = np.array(getattr(self, field), dtype=np.float64, copy=True)
            if value.shape != (size,) or np.any(np.isnan(value)):
                raise ValueError(f"Invalid {field} dimensions or NaNs")
            object.__setattr__(self, field, value)
        if n == 0 or not np.all(np.isfinite(A.data)) or not np.all(np.isfinite(self.c)) or not np.isfinite(self.offset):
            raise ValueError("LP requires variables and finite matrix/objective data")
        for lower, upper in ((self.row_lower, self.row_upper), (self.variable_lower, self.variable_upper)):
            if np.any(lower > upper) or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
                raise ValueError("Invalid lower/upper bounds")

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.A.shape, dtype="<i8").tobytes())
        for a in (self.A.indptr, self.A.indices):
            digest.update(np.asarray(a, dtype="<i8").tobytes())
        for a in (self.A.data, self.c, self.row_lower, self.row_upper,
                  self.variable_lower, self.variable_upper, [self.offset]):
            digest.update(np.asarray(a, dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        np.savez_compressed(path, data=self.A.data, indices=self.A.indices,
                            indptr=self.A.indptr, shape=self.A.shape, c=self.c,
                            row_lower=self.row_lower, row_upper=self.row_upper,
                            variable_lower=self.variable_lower, variable_upper=self.variable_upper,
                            offset=self.offset, name=self.name)

    @classmethod
    def load(cls, path: Path) -> "SparseLP":
        with np.load(path, allow_pickle=False) as d:
            A = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
            return cls(A, d["c"], d["row_lower"], d["row_upper"], d["variable_lower"],
                       d["variable_upper"], float(d["offset"]), str(d["name"]))


def _highs():
    try:
        import highspy
    except ImportError as exc:
        raise ImportError("Install the optional pdlp extra: pip install '.[pdlp]'") from exc
    return highspy


def read_mps(path: Path) -> SparseLP:
    """Read safely through HiGHS; reject integer and quadratic models."""
    highspy = _highs()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"MPS read failed or warned: {path} ({status})")
    model = h.getModel()
    lp = model.lp_
    if model.hessian_.dim_ or any(t != highspy.HighsVarType.kContinuous for t in lp.integrality_):
        raise ValueError("Benchmark accepts continuous linear programs only")
    matrix = lp.a_matrix_
    if matrix.format_ == highspy.MatrixFormat.kColwise:
        A = sparse.csc_matrix((matrix.value_, matrix.index_, matrix.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()
    elif matrix.format_ == highspy.MatrixFormat.kRowwise:
        A = sparse.csr_matrix((matrix.value_, matrix.index_, matrix.start_), shape=(lp.num_row_, lp.num_col_))
    else:
        raise ValueError(f"Unsupported HiGHS matrix format: {matrix.format_}")
    multiplier = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    return SparseLP(A, multiplier * np.asarray(lp.col_cost_), lp.row_lower_, lp.row_upper_,
                    lp.col_lower_, lp.col_upper_, multiplier * lp.offset_, path.stem)


def _bound_errors(values, lower, upper) -> tuple[float, float]:
    absolute = relative = 0.0
    for bound, residual in ((lower, lower - values), (upper, values - upper)):
        mask = np.isfinite(bound)
        violation = np.maximum(residual[mask], 0.0)
        absolute = max(absolute, float(np.max(violation, initial=0.0)))
        relative = max(relative, float(np.max(violation / (1.0 + np.abs(bound[mask])), initial=0.0)))
    return absolute, relative


def _project_dual(values, lower, upper):
    """Coefficients of lower bounds are positive; upper bounds negative."""
    result = np.asarray(values, dtype=float).copy()
    result[~np.isfinite(lower)] = np.minimum(result[~np.isfinite(lower)], 0.0)
    result[~np.isfinite(upper)] = np.maximum(result[~np.isfinite(upper)], 0.0)
    return result


def _bound_objective(multiplier, lower, upper) -> float:
    positive, negative = multiplier > 0.0, multiplier < 0.0
    return float(multiplier[positive] @ lower[positive] + multiplier[negative] @ upper[negative])


def check_pair(lp: SparseLP, x, y, *, primal_tolerance=1e-7,
               dual_tolerance=1e-7, gap_tolerance=1e-4) -> dict:
    """Recompute numerical feasibility/gap, without trusting solver status.

    A tolerance-feasible dual is not an exact lower-bound certificate. Reduced
    costs are derived from c-A.T@y and projected onto admissible bound signs.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != (lp.n,) or y.shape != (lp.m,):
        raise ValueError("Primal/dual dimensions do not match LP")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return dict(finite=False, primal_feasible=False, dual_feasible=False, verified_success=False,
                    primal_objective=None, dual_objective_estimate=None, relative_gap=None,
                    primal_violation=None, primal_residual=None, dual_residual=None, dual_sign_violation=None)
    row_abs, row_rel = _bound_errors(lp.A @ x, lp.row_lower, lp.row_upper)
    bound_abs, bound_rel = _bound_errors(x, lp.variable_lower, lp.variable_upper)
    projected_y = _project_dual(y, lp.row_lower, lp.row_upper)
    gradient = lp.c - lp.A.T @ projected_y
    reduced_cost = _project_dual(gradient, lp.variable_lower, lp.variable_upper)
    sign_error = float(np.max(np.abs(y - projected_y), initial=0.0))
    stationarity = float(np.max(np.abs(gradient - reduced_cost), initial=0.0))
    dual_residual = max(stationarity / (1.0 + float(np.max(np.abs(lp.c), initial=0.0))),
                        sign_error / (1.0 + float(np.max(np.abs(y), initial=0.0))))
    primal_residual = max(row_rel, bound_rel)
    primal = float(lp.c @ x + lp.offset)
    dual = (_bound_objective(projected_y, lp.row_lower, lp.row_upper)
            + _bound_objective(reduced_cost, lp.variable_lower, lp.variable_upper) + lp.offset)
    gap = abs(primal - dual) / (1.0 + abs(primal) + abs(dual))
    finite = all(np.isfinite(v) for v in (primal, dual, gap, primal_residual, dual_residual))
    if not finite:
        raise ValueError("Numerical overflow while checking candidate")
    primal_ok, dual_ok = primal_residual <= primal_tolerance, dual_residual <= dual_tolerance
    return dict(finite=True, primal_feasible=primal_ok, dual_feasible=dual_ok,
                verified_success=primal_ok and dual_ok and gap <= gap_tolerance,
                primal_objective=primal, dual_objective_estimate=dual, relative_gap=gap,
                primal_violation=max(row_abs, bound_abs), primal_residual=primal_residual,
                dual_residual=dual_residual, dual_sign_violation=sign_error)


def project_primal(lp: SparseLP, x, *, max_steps=256, tolerance=1e-8, deadline=None) -> tuple[np.ndarray, int]:
    """General-bound version of the prototype's greedy halfspace recovery.

    Equalities use their exact hyperplane. Sparse rows are never densified.
    """
    if max_steps <= 0 or not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Positive steps and finite nonnegative tolerance required")
    point = np.asarray(x, dtype=float).copy()
    if point.shape != (lp.n,) or not np.all(np.isfinite(point)):
        raise ValueError("Invalid recovery point")
    if max(_bound_errors(lp.A @ point, lp.row_lower, lp.row_upper)[1],
           _bound_errors(point, lp.variable_lower, lp.variable_upper)[1]) <= tolerance:
        return point, 0
    point = np.clip(point, lp.variable_lower, lp.variable_upper)
    squared_norms = np.asarray(lp.A.multiply(lp.A).sum(axis=1)).ravel()
    norms = np.sqrt(np.maximum(squared_norms, np.finfo(float).tiny))
    lower_scale = 1.0 + np.abs(np.where(np.isfinite(lp.row_lower), lp.row_lower, 0.0))
    upper_scale = 1.0 + np.abs(np.where(np.isfinite(lp.row_upper), lp.row_upper, 0.0))
    for step in range(max_steps):
        if deadline is not None and time.perf_counter() >= deadline:
            return point, step
        activity = lp.A @ point
        lower_excess, upper_excess = lp.row_lower - activity, activity - lp.row_upper
        lower_bad = lower_excess > tolerance * lower_scale
        upper_bad = upper_excess > tolerance * upper_scale
        distance = np.maximum(np.where(lower_bad, lower_excess, 0.0), np.where(upper_bad, upper_excess, 0.0)) / norms
        if not np.any(distance > 0.0):
            return point, step
        row = int(np.argmax(distance))
        if squared_norms[row] == 0.0:
            return point, step  # Contradictory constant row; no infeasibility claim.
        is_lower = lower_excess[row] > upper_excess[row]
        target = lp.row_lower[row] if is_lower else lp.row_upper[row]
        if lp.row_lower[row] != lp.row_upper[row]:
            # Move slightly inside an inequality, capped by its interval width.
            margin = min(tolerance * 0.01 * (1.0 + abs(target)),
                         (lp.row_upper[row] - lp.row_lower[row]) * 0.25)
            target += margin if is_lower else -margin
        start, end = lp.A.indptr[row:row + 2]
        columns, values = lp.A.indices[start:end], lp.A.data[start:end]
        point[columns] += ((target - activity[row]) / squared_norms[row]) * values
        point = np.clip(point, lp.variable_lower, lp.variable_upper)
    return point, max_steps


def to_pdlp(lp: SparseLP, *, feasibility_objective=False):
    try:
        from ortools.pdlp.python import pdlp
    except ImportError as exc:
        raise ImportError("Install the optional pdlp extra: pip install '.[pdlp]'") from exc
    qp = pdlp.QuadraticProgram()
    qp.objective_vector = np.zeros(lp.n) if feasibility_objective else lp.c
    qp.objective_offset = 0.0 if feasibility_objective else lp.offset
    qp.constraint_matrix = lp.A
    qp.constraint_lower_bounds, qp.constraint_upper_bounds = lp.row_lower, lp.row_upper
    qp.variable_lower_bounds, qp.variable_upper_bounds = lp.variable_lower, lp.variable_upper
    qp.problem_name = lp.name
    pdlp.validate_quadratic_program_dimensions(qp)
    return qp


def pdlp_parameters(*, time_limit, polishing=False, iteration_limit=None,
                    residual_tolerance=1e-8, gap_tolerance=1e-6, check_frequency=32):
    from ortools.pdlp import solvers_pb2
    if not np.isfinite(time_limit) or time_limit <= 0.0:
        raise ValueError("PDLP needs a finite positive time limit")
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.num_threads = 1
    params.verbosity_level = 0
    params.presolve_options.use_glop = False
    params.handle_some_primal_gradients_on_finite_bounds_as_residuals = False
    params.use_feasibility_polishing = polishing
    params.apply_feasibility_polishing_after_limits_reached = False
    params.apply_feasibility_polishing_if_solver_is_interrupted = False
    params.termination_check_frequency = check_frequency
    criteria = params.termination_criteria
    criteria.time_sec_limit = time_limit
    criteria.optimality_norm = solvers_pb2.OPTIMALITY_NORM_L_INF_COMPONENTWISE
    if iteration_limit is not None:
        if iteration_limit < 0:
            raise ValueError("Iteration limit must be nonnegative")
        criteria.iteration_limit = iteration_limit
    detailed = criteria.detailed_optimality_criteria
    detailed.eps_optimal_primal_residual_absolute = residual_tolerance
    detailed.eps_optimal_primal_residual_relative = residual_tolerance
    detailed.eps_optimal_dual_residual_absolute = residual_tolerance
    detailed.eps_optimal_dual_residual_relative = residual_tolerance
    detailed.eps_optimal_objective_gap_absolute = gap_tolerance
    detailed.eps_optimal_objective_gap_relative = gap_tolerance
    return params


def solve_pdlp(lp: SparseLP, *, time_limit, polishing=False, iteration_limit=None,
               initial=None, feasibility_objective=False, **parameter_options) -> dict:
    from google.protobuf.json_format import MessageToDict
    from ortools.pdlp import solve_log_pb2
    from ortools.pdlp.python import pdlp
    start = time.perf_counter()
    qp = to_pdlp(lp, feasibility_objective=feasibility_objective)
    params = pdlp_parameters(time_limit=time_limit, polishing=polishing,
                             iteration_limit=iteration_limit, **parameter_options)
    warm = None
    if initial is not None:
        x, y = (np.asarray(v, dtype=float) for v in initial)
        if x.shape != (lp.n,) or y.shape != (lp.m,) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Invalid PDLP initial solution")
        warm = pdlp.PrimalAndDualSolution()
        warm.primal_solution, warm.dual_solution = x, y
    setup_seconds = time.perf_counter() - start
    remaining = time_limit - setup_seconds
    if remaining <= 0.0:
        # No unbudgeted solve. The normal default start is zero, box-projected.
        x = np.clip(np.zeros(lp.n), lp.variable_lower, lp.variable_upper) if initial is None else np.asarray(initial[0]).copy()
        y = np.zeros(lp.m) if initial is None else np.asarray(initial[1]).copy()
        return dict(x=x, y=y, status="SETUP_TIME_LIMIT", iterations=0, solver_seconds=0.0,
                    setup_seconds=setup_seconds, elapsed_seconds=time.perf_counter() - start,
                    parameters=MessageToDict(params, preserving_proto_field_name=True), log={})
    params.termination_criteria.time_sec_limit = remaining
    result = pdlp.primal_dual_hybrid_gradient(qp, params, initial_solution=warm)
    x, y = np.array(result.primal_solution), np.array(result.dual_solution)
    elapsed = time.perf_counter() - start
    log = MessageToDict(result.solve_log, preserving_proto_field_name=True)
    status = solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason)
    if status in {"TERMINATION_REASON_INVALID_PARAMETER", "TERMINATION_REASON_INVALID_PROBLEM"}:
        raise RuntimeError(f"PDLP failed: {status}: {log}")
    return dict(x=x, y=y, status=status, iterations=int(result.solve_log.iteration_count),
                solver_seconds=float(result.solve_log.solve_time_sec), setup_seconds=setup_seconds,
                elapsed_seconds=elapsed, parameters=MessageToDict(params, preserving_proto_field_name=True), log=log)


def solve_highs(lp: SparseLP, *, time_limit=30.0) -> dict:
    """External context/reference, never used to guide recovery."""
    highspy = _highs()
    start = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", time_limit)
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = lp.n, lp.m
    model.col_cost_, model.col_lower_, model.col_upper_ = lp.c, lp.variable_lower, lp.variable_upper
    model.row_lower_, model.row_upper_, model.offset_ = lp.row_lower, lp.row_upper, lp.offset
    A = lp.A.tocsc()
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = A.indptr, A.indices, A.data
    if h.passModel(model) != highspy.HighsStatus.kOk:
        raise ValueError("HiGHS rejected LP")
    h.run()
    solution = h.getSolution()
    return dict(x=np.asarray(solution.col_value), y=np.asarray(solution.row_dual),
                status=h.modelStatusToString(h.getModelStatus()), elapsed_seconds=time.perf_counter() - start,
                primal_valid=bool(solution.value_valid), dual_valid=bool(solution.dual_valid))
