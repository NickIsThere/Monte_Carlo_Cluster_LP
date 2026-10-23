from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.experiments.benchmark_runner import GEOMETRY_AWARE_METHOD, run_problem_comparison
from lpas.experiments.metrics import relative_objective_error
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.utils.config import SolverConfig


OUTCOME_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CornerDiscoveryRecord:
    problem_name: str
    n: int
    m: int
    seed: int
    num_unique_reconstructed_vertices: int
    num_feasible_reconstructed_vertices: int
    num_infeasible_reconstructed_vertices: int
    fraction_feasible_reconstructed_vertices: float
    has_feasible_reconstructed_vertex: bool
    best_reconstructed_objective: float | None
    best_reconstructed_objective_any: float | None
    best_feasible_reconstructed_objective: float | None
    best_infeasible_reconstructed_objective: float | None
    best_feasible_reconstructed_x: np.ndarray | None
    best_infeasible_reconstructed_x: np.ndarray | None
    scipy_objective: float | None
    relative_error_best_feasible: float | None
    relative_error_best_any: float | None
    active_set_jaccard_to_scipy_optimum: float | None
    true_active_containment: float | None
    reconstruction_success_rate: float
    frequency_of_best_feasible_vertex: int
    frequency_of_best_any_vertex: int
    failure_reason: str | None
    outcome_status: str
    num_unique_vertices_found: int
    num_unique_feasible_vertices_found: int
    num_unique_infeasible_vertices_found: int
    top_1_feasible_objective: float | None
    top_5_feasible_objectives: list[float] = field(default_factory=list)
    top_10_feasible_objectives: list[float] = field(default_factory=list)
    objective_diversity_feasible: float | None = None
    active_set_diversity_feasible: float | None = None
    best_vertex_frequency: int | None = None
    median_vertex_frequency: float | None = None
    max_vertex_frequency: int | None = None
    num_vertices_matching_scipy_active_set: int = 0
    num_vertices_containing_true_active_set: int = 0


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _containment(result, candidate) -> float | None:
    if candidate is None or result.reference_result.augmented_primal_active_mask is None:
        return None
    truth = np.asarray(result.reference_result.augmented_primal_active_mask, dtype=bool)
    if not np.any(truth):
        return 1.0
    predicted = np.asarray(candidate.active_set, dtype=bool)
    overlap = float(np.sum(predicted & truth))
    return float(overlap / np.sum(truth))


def _failure_reason(outcome_status: str) -> str | None:
    if outcome_status.startswith("SUCCESS"):
        return None
    if outcome_status == "INFEASIBLE_HIGH_OBJECTIVE_ARTIFACT":
        return "NO_FEASIBLE_RECONSTRUCTED_VERTEX"
    return outcome_status


def _outcome_status(
    *,
    scipy_success: bool,
    scipy_objective: float | None,
    relative_error_best_feasible: float | None,
    has_feasible_vertex: bool,
    best_infeasible_objective: float | None,
) -> str:
    if not scipy_success:
        return "SCIPY_FAILED"
    if has_feasible_vertex:
        if relative_error_best_feasible is not None and relative_error_best_feasible <= OUTCOME_MATCH_TOLERANCE:
            return "SUCCESS_FEASIBLE_VERTEX_MATCH"
        return "SUCCESS_FEASIBLE_VERTEX_SUBOPTIMAL"
    if (
        best_infeasible_objective is not None
        and scipy_objective is not None
        and best_infeasible_objective > scipy_objective + 1e-12
    ):
        return "INFEASIBLE_HIGH_OBJECTIVE_ARTIFACT"
    return "NO_FEASIBLE_RECONSTRUCTED_VERTEX"


def _record_from_result(result) -> CornerDiscoveryRecord:
    corner = result.corner_discovery
    feasible_best = None if corner is None else corner.best_feasible_vertex
    infeasible_best = None if corner is None else corner.best_infeasible_vertex
    any_best = None if corner is None or not corner.unique_vertices else corner.unique_vertices[0]
    num_unique = 0 if corner is None else corner.num_unique_vertices
    num_feasible = 0 if corner is None else corner.num_feasible_vertices
    num_infeasible = 0 if corner is None else corner.num_infeasible_vertices
    scipy_objective = result.reference_result.objective
    best_feasible_objective = None if feasible_best is None else float(feasible_best.objective)
    best_infeasible_objective = None if infeasible_best is None else float(infeasible_best.objective)
    best_any_objective = None if any_best is None else float(any_best.objective)
    relative_error_best_feasible = relative_objective_error(best_feasible_objective, scipy_objective)
    relative_error_best_any = relative_objective_error(best_any_objective, scipy_objective)
    outcome_status = _outcome_status(
        scipy_success=bool(result.reference_result.success),
        scipy_objective=scipy_objective,
        relative_error_best_feasible=relative_error_best_feasible,
        has_feasible_vertex=bool(feasible_best is not None),
        best_infeasible_objective=best_infeasible_objective,
    )
    top_feasible_objectives = [] if corner is None else [float(value) for value in corner.top_feasible_objectives[:10]]
    return CornerDiscoveryRecord(
        problem_name=result.problem_name,
        n=result.n_variables,
        m=result.n_constraints,
        seed=result.seed,
        num_unique_reconstructed_vertices=num_unique,
        num_feasible_reconstructed_vertices=num_feasible,
        num_infeasible_reconstructed_vertices=num_infeasible,
        fraction_feasible_reconstructed_vertices=0.0 if num_unique == 0 else float(num_feasible / num_unique),
        has_feasible_reconstructed_vertex=bool(feasible_best is not None),
        best_reconstructed_objective=best_feasible_objective,
        best_reconstructed_objective_any=best_any_objective,
        best_feasible_reconstructed_objective=best_feasible_objective,
        best_infeasible_reconstructed_objective=best_infeasible_objective,
        best_feasible_reconstructed_x=None if feasible_best is None else np.asarray(feasible_best.x, dtype=float),
        best_infeasible_reconstructed_x=None if infeasible_best is None else np.asarray(infeasible_best.x, dtype=float),
        scipy_objective=scipy_objective,
        relative_error_best_feasible=relative_error_best_feasible,
        relative_error_best_any=relative_error_best_any,
        active_set_jaccard_to_scipy_optimum=None if corner is None else corner.optimum_jaccard,
        true_active_containment=_containment(result, feasible_best),
        reconstruction_success_rate=1.0 if feasible_best is not None else 0.0,
        frequency_of_best_feasible_vertex=0 if feasible_best is None else int(feasible_best.frequency),
        frequency_of_best_any_vertex=0 if any_best is None else int(any_best.frequency),
        failure_reason=_failure_reason(outcome_status),
        outcome_status=outcome_status,
        num_unique_vertices_found=num_unique,
        num_unique_feasible_vertices_found=num_feasible,
        num_unique_infeasible_vertices_found=num_infeasible,
        top_1_feasible_objective=None if not top_feasible_objectives else top_feasible_objectives[0],
        top_5_feasible_objectives=top_feasible_objectives[:5],
        top_10_feasible_objectives=top_feasible_objectives,
        objective_diversity_feasible=None if corner is None else corner.objective_diversity_feasible,
        active_set_diversity_feasible=None if corner is None else corner.active_set_diversity_feasible,
        best_vertex_frequency=None if corner is None else corner.best_vertex_frequency,
        median_vertex_frequency=None if corner is None else corner.median_vertex_frequency,
        max_vertex_frequency=None if corner is None else corner.max_vertex_frequency,
        num_vertices_matching_scipy_active_set=0 if corner is None else corner.num_vertices_matching_reference_active_set,
        num_vertices_containing_true_active_set=0 if corner is None else corner.num_vertices_containing_true_active_set,
    )


def run_corner_discovery_experiment(
    *,
    config: SolverConfig,
    dimensions: Iterable[int] = (2, 5, 10, 12, 15, 18, 20, 30, 50),
    ratios: Iterable[int] = (2, 5, 10),
    seeds: Iterable[int] = range(5),
) -> list[CornerDiscoveryRecord]:
    records: list[CornerDiscoveryRecord] = []
    for n in dimensions:
        for ratio in ratios:
            m = int(ratio) * int(n)
            for seed in seeds:
                generated = generate_random_feasible_bounded_lp(n, m, int(seed))
                results = run_problem_comparison(
                    problem_name=generated.name,
                    family="corner_discovery",
                    problem=generated.problem,
                    config=config,
                    capture_samples=False,
                )
                geometry_result = next(result for result in results if result.method == GEOMETRY_AWARE_METHOD)
                records.append(_record_from_result(geometry_result))
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_directory(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_directory(path.parent)
    path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")


def _plot_line(rows: list[dict[str, Any]], *, y_key: str, output_path: Path, title: str, ylabel: str) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = row.get(y_key)
        if value is None or not math.isfinite(float(value)):
            continue
        grouped.setdefault(int(row["n"]), []).append(float(value))
    if not grouped:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_hist(rows: list[dict[str, Any]], *, key: str, output_path: Path, title: str) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    if values:
        ax.hist(values, bins=min(12, max(3, len(values))))
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_corner_discovery_plots(records: list[CornerDiscoveryRecord], output_dir: Path) -> list[Path]:
    rows = [_jsonify(record) for record in records]
    outputs = [
        output_dir / "num_unique_vertices_vs_dimension.png",
        output_dir / "best_vertex_objective_error.png",
        output_dir / "active_set_jaccard_vs_dimension.png",
        output_dir / "true_active_containment_vs_dimension.png",
        output_dir / "reconstruction_success_rate.png",
        output_dir / "top_k_vertex_objectives.png",
    ]
    _plot_line(rows, y_key="num_unique_reconstructed_vertices", output_path=outputs[0], title="Unique vertices vs dimension", ylabel="count")
    _plot_hist(rows, key="best_feasible_reconstructed_objective", output_path=outputs[1], title="Best feasible reconstructed objective")
    _plot_line(rows, y_key="active_set_jaccard_to_scipy_optimum", output_path=outputs[2], title="Active-set Jaccard vs dimension", ylabel="jaccard")
    _plot_line(rows, y_key="true_active_containment", output_path=outputs[3], title="True active containment vs dimension", ylabel="containment")
    _plot_line(rows, y_key="reconstruction_success_rate", output_path=outputs[4], title="Reconstruction success rate", ylabel="rate")
    _plot_hist(rows, key="frequency_of_best_any_vertex", output_path=outputs[5], title="Top-k vertex frequencies")
    return outputs


def write_corner_discovery_outputs(
    records: list[CornerDiscoveryRecord],
    output_root: Path,
    *,
    config_payload: dict[str, Any],
) -> dict[str, Path]:
    results_dir = output_root / "results" / "corner_discovery"
    figures_dir = output_root / "figures" / "corner_discovery"
    rows = [_jsonify(record) for record in records]
    results_csv = results_dir / "corner_discovery_results.csv"
    results_json = results_dir / "corner_discovery_results.json"
    _write_csv(results_csv, rows)
    _write_json(results_json, {"config": config_payload, "records": rows})
    make_corner_discovery_plots(records, figures_dir)
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "figure_dir": figures_dir,
    }
