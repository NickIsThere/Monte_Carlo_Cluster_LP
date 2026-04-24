from __future__ import annotations

from dataclasses import dataclass

from lpas.experiments.benchmark_runner import (
    GEOMETRY_AWARE_METHOD,
    GEOMETRY_AWARE_POLISHED_METHOD,
    NAIVE_MONTE_CARLO_METHOD,
    MethodExperimentResult,
    run_random_dense_suite,
)
from lpas.experiments.metrics import active_set_jaccard, objective_gap_to_reference
from lpas.solver.hints import ActiveSetHintEvaluation, evaluate_archive_hint
from lpas.utils.config import SolverConfig


@dataclass(frozen=True)
class SolverHintRecord:
    problem_name: str
    family: str
    method: str
    seed: int
    n_variables: int
    n_constraints: int
    hint_active_set_jaccard: float
    hint_corner_feasible: bool
    hint_corner_objective: float | None
    objective_gap_to_highs: float | None
    constraints_in_top_k_support: float
    reconstruction_success: bool
    wall_clock_seconds: float
    top_k: int
    reference_objective: float | None


def evaluate_method_hint(problem, result: MethodExperimentResult) -> tuple[SolverHintRecord, ActiveSetHintEvaluation]:
    reference_active_mask = result.reference_result.primal_active_mask
    if reference_active_mask is None:
        raise ValueError("solver-hint experiments require a reference active set")
    evaluation = evaluate_archive_hint(
        problem,
        result.elite_archive,
        reference_active_mask,
    )
    hint_active_set_jaccard = evaluation.hint_active_set_jaccard
    hint_corner_feasible = evaluation.reconstruction.feasible
    objective = evaluation.reconstruction.objective
    reconstruction_success = evaluation.reconstruction.feasible
    if result.method == GEOMETRY_AWARE_POLISHED_METHOD and result.polishing_result is not None and result.polishing_result.best_vertex is not None:
        polished_best = result.polishing_result.best_vertex
        hint_active_set_jaccard = active_set_jaccard(polished_best.original_active_mask, reference_active_mask)
        hint_corner_feasible = polished_best.feasible
        objective = polished_best.objective
        reconstruction_success = polished_best.feasible
    return (
        SolverHintRecord(
            problem_name=result.problem_name,
            family=result.family,
            method=result.method,
            seed=result.seed,
            n_variables=result.n_variables,
            n_constraints=result.n_constraints,
            hint_active_set_jaccard=hint_active_set_jaccard,
            hint_corner_feasible=hint_corner_feasible,
            hint_corner_objective=objective,
            objective_gap_to_highs=objective_gap_to_reference(objective, result.reference_result.objective),
            constraints_in_top_k_support=evaluation.constraints_in_top_k_support,
            reconstruction_success=reconstruction_success,
            wall_clock_seconds=result.wall_clock_seconds,
            top_k=evaluation.top_k,
            reference_objective=result.reference_result.objective,
        ),
        evaluation,
    )


def run_solver_hint_suite(instances, *, config: SolverConfig) -> list[SolverHintRecord]:
    records: list[SolverHintRecord] = []
    for instance in instances:
        method_results = run_random_dense_suite([instance], config=config, capture_samples=False)
        for result in method_results:
            record, _ = evaluate_method_hint(instance.problem, result)
            records.append(record)
    return records


__all__ = [
    "GEOMETRY_AWARE_METHOD",
    "GEOMETRY_AWARE_POLISHED_METHOD",
    "NAIVE_MONTE_CARLO_METHOD",
    "SolverHintRecord",
    "evaluate_method_hint",
    "run_solver_hint_suite",
]
