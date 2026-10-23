from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.experiments.benchmark_runner import MethodExperimentResult, run_problem_comparison
from lpas.experiments.metrics import objective_gap_to_reference, relative_objective_error
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.utils.config import SolverConfig


@dataclass(frozen=True)
class ScalingRecord:
    problem_name: str
    method: str
    n: int
    m: int
    m_over_n: float
    seed: int
    backend: str
    samples_per_iteration: int
    iterations: int
    chunk_size: int | None
    success: bool
    relative_error: float | None
    absolute_error: float | None
    best_raw_objective: float | None
    best_feasible_primal_lower_bound: float | None
    best_feasible_dual_upper_bound: float | None
    best_polished_vertex_objective: float | None
    scipy_objective: float | None
    raw_gap: float | None
    certified_gap: float | None
    primal_violation: float | None
    dual_violation: float | None
    active_set_jaccard: float | None
    true_active_containment: float | None
    reconstruction_success: bool
    fraction_feasible_vertices: float | None
    outcome_status: str | None
    failure_reason: str | None
    num_unique_vertices: int
    time_to_good_bound_1e_2: float | None
    time_to_good_bound_1e_3: float | None
    time_to_good_bound_1e_4: float | None
    time_to_first_feasible_reconstructed_vertex: float | None
    time_to_first_optimal_reconstructed_vertex: float | None
    total_runtime_seconds: float
    samples_per_second: float
    approx_success_1e_2: bool
    approx_success_1e_3: bool
    approx_success_1e_4: bool
    active_set_success: bool
    vertex_recovery_success: bool
    certificate_success: bool


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


def _active_set_containment(result: MethodExperimentResult) -> float | None:
    if (
        result.corner_discovery is None
        or result.corner_discovery.best_vertex is None
        or result.reference_result.augmented_primal_active_mask is None
    ):
        return None
    predicted = np.asarray(result.corner_discovery.best_vertex.active_set, dtype=bool)
    truth = np.asarray(result.reference_result.augmented_primal_active_mask, dtype=bool)
    truth_count = int(np.sum(truth))
    if truth_count == 0:
        return 1.0
    overlap = int(np.sum(predicted & truth))
    return float(overlap / truth_count)


def _best_active_set_jaccard(result: MethodExperimentResult) -> float | None:
    candidates = [
        result.raw_active_set_similarity,
        result.polished_active_set_similarity,
        None if result.corner_discovery is None else result.corner_discovery.optimum_jaccard,
    ]
    numeric = [float(value) for value in candidates if value is not None]
    if not numeric:
        return None
    return float(max(numeric))


def _time_to_good_bound(history, reference_objective: float | None, tolerance: float) -> float | None:
    if reference_objective is None:
        return None
    for entry in history:
        objective = entry.best_feasible_objective
        if objective is None:
            continue
        error = relative_objective_error(objective, reference_objective)
        if error is not None and error <= tolerance:
            return float(entry.elapsed_seconds)
    return None


def _record_from_result(result: MethodExperimentResult, *, backend: str) -> ScalingRecord:
    reference_objective = result.reference_result.objective
    final_objective = result.best_feasible_objective
    relative_error = relative_objective_error(final_objective, reference_objective)
    absolute_error = objective_gap_to_reference(final_objective, reference_objective)
    containment = _active_set_containment(result)
    corner = result.corner_discovery
    reconstruction_success = bool(
        corner is not None
        and corner.best_vertex is not None
        and corner.best_vertex.is_feasible
    )
    fraction_feasible_vertices = None
    outcome_status = None
    failure_reason = None
    time_to_first_feasible_reconstructed_vertex = None
    time_to_first_optimal_reconstructed_vertex = None
    if corner is not None:
        if corner.num_unique_vertices > 0:
            fraction_feasible_vertices = float(corner.num_feasible_vertices / corner.num_unique_vertices)
        best_infeasible_objective = None if corner.best_infeasible_vertex is None else corner.best_infeasible_vertex.objective
        if reconstruction_success:
            if relative_error is not None and relative_error <= 1e-6:
                outcome_status = "SUCCESS_FEASIBLE_VERTEX_MATCH"
                time_to_first_optimal_reconstructed_vertex = float(result.wall_clock_seconds)
            else:
                outcome_status = "SUCCESS_FEASIBLE_VERTEX_SUBOPTIMAL"
            time_to_first_feasible_reconstructed_vertex = float(result.wall_clock_seconds)
        elif (
            best_infeasible_objective is not None
            and reference_objective is not None
            and best_infeasible_objective > reference_objective + 1e-12
        ):
            outcome_status = "INFEASIBLE_HIGH_OBJECTIVE_ARTIFACT"
            failure_reason = outcome_status
        else:
            outcome_status = "NO_FEASIBLE_RECONSTRUCTED_VERTEX"
            failure_reason = outcome_status
    vertex_recovery_error = relative_objective_error(result.polished_objective, reference_objective)
    certificate_success = bool(
        result.bound_certificate is not None
        and result.bound_certificate.has_valid_certificate
        and result.bound_certificate.certified_gap is not None
        and result.bound_certificate.certified_gap <= 1e-6
    )
    samples_per_second = (
        float(result.n_samples_total / result.wall_clock_seconds)
        if result.wall_clock_seconds > 0.0
        else math.inf
    )
    return ScalingRecord(
        problem_name=result.problem_name,
        method=result.method,
        n=result.n_variables,
        m=result.n_constraints,
        m_over_n=float(result.n_constraints / max(result.n_variables, 1)),
        seed=result.seed,
        backend=backend,
        samples_per_iteration=int(result.n_samples_total / max(len(result.history), 1)),
        iterations=len(result.history),
        chunk_size=None,
        success=bool(relative_error is not None),
        relative_error=relative_error,
        absolute_error=absolute_error,
        best_raw_objective=result.raw_best_objective,
        best_feasible_primal_lower_bound=result.best_feasible_primal_lower_bound,
        best_feasible_dual_upper_bound=result.best_feasible_dual_upper_bound,
        best_polished_vertex_objective=result.polished_objective,
        scipy_objective=reference_objective,
        raw_gap=result.best_raw_gap,
        certified_gap=result.best_certified_gap,
        primal_violation=result.best_primal_violation,
        dual_violation=result.best_dual_violation,
        active_set_jaccard=_best_active_set_jaccard(result),
        true_active_containment=containment,
        reconstruction_success=reconstruction_success,
        fraction_feasible_vertices=fraction_feasible_vertices,
        outcome_status=outcome_status,
        failure_reason=failure_reason,
        num_unique_vertices=0 if corner is None else corner.num_unique_vertices,
        time_to_good_bound_1e_2=_time_to_good_bound(result.history, reference_objective, 1e-2),
        time_to_good_bound_1e_3=_time_to_good_bound(result.history, reference_objective, 1e-3),
        time_to_good_bound_1e_4=_time_to_good_bound(result.history, reference_objective, 1e-4),
        time_to_first_feasible_reconstructed_vertex=time_to_first_feasible_reconstructed_vertex,
        time_to_first_optimal_reconstructed_vertex=time_to_first_optimal_reconstructed_vertex,
        total_runtime_seconds=float(result.wall_clock_seconds),
        samples_per_second=samples_per_second,
        approx_success_1e_2=bool(relative_error is not None and relative_error <= 1e-2),
        approx_success_1e_3=bool(relative_error is not None and relative_error <= 1e-3),
        approx_success_1e_4=bool(relative_error is not None and relative_error <= 1e-4),
        active_set_success=bool(containment is not None and containment >= 0.8),
        vertex_recovery_success=bool(
            result.polished_certified_feasible
            and vertex_recovery_error is not None
            and vertex_recovery_error <= 1e-6
        ),
        certificate_success=certificate_success,
    )


def run_scaling_experiment(
    *,
    config: SolverConfig,
    dimensions: Iterable[int] = (2, 5, 10, 12, 15, 18, 20, 30, 50),
    ratios: Iterable[int] = (2, 5, 10),
    seeds: Iterable[int] = range(5),
    backend: str = "numpy_cpu",
) -> list[ScalingRecord]:
    if backend != "numpy_cpu":
        raise ValueError("The scaling experiment currently supports only the numpy_cpu adaptive sampler path.")

    records: list[ScalingRecord] = []
    for n in dimensions:
        for ratio in ratios:
            m = int(ratio) * int(n)
            for seed in seeds:
                generated = generate_random_feasible_bounded_lp(n, m, int(seed))
                results = run_problem_comparison(
                    problem_name=generated.name,
                    family="scaling",
                    problem=generated.problem,
                    config=config,
                    capture_samples=False,
                )
                records.extend(_record_from_result(result, backend=backend) for result in results)
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


def _safe_mean(values: Iterable[float | None]) -> float:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return math.nan
    return float(np.mean(numeric))


def _safe_median(values: Iterable[float | None]) -> float:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return math.nan
    return float(np.median(numeric))


def _group_summary(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), row[key])].append(row)
    summary: list[dict[str, Any]] = []
    metrics = [
        "relative_error",
        "active_set_jaccard",
        "true_active_containment",
        "time_to_good_bound_1e_2",
        "samples_per_second",
        "num_unique_vertices",
        "approx_success_1e_2",
        "approx_success_1e_3",
        "approx_success_1e_4",
        "active_set_success",
        "reconstruction_success",
        "vertex_recovery_success",
        "certificate_success",
    ]
    for (method, group_key), group_rows in grouped.items():
        entry: dict[str, Any] = {"method": method, key: group_key, "count": len(group_rows)}
        for metric in metrics:
            entry[f"{metric}_mean"] = _safe_mean(row.get(metric) for row in group_rows)
        summary.append(entry)
    summary.sort(key=lambda row: (str(row["method"]), float(row[key])))
    return summary


def _aggregate_scaling_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["n"]),
                float(row["m_over_n"]),
                int(row["m"]),
                str(row["backend"]),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (n, m_ratio, m, backend), group_rows in sorted(grouped.items()):
        success_values = [1.0 if bool(row.get("reconstruction_success")) else 0.0 for row in group_rows]
        no_feasible = [1.0 if row.get("outcome_status") == "NO_FEASIBLE_RECONSTRUCTED_VERTEX" else 0.0 for row in group_rows]
        infeasible_artifact = [
            1.0 if row.get("outcome_status") == "INFEASIBLE_HIGH_OBJECTIVE_ARTIFACT" else 0.0
            for row in group_rows
        ]
        entry: dict[str, Any] = {
            "n": n,
            "m_ratio": m_ratio,
            "m": m,
            "backend": backend,
            "count": len(group_rows),
            "mean_success_rate": _safe_mean(success_values),
            "median_success_rate": _safe_median(success_values),
            "mean_fraction_feasible_vertices": _safe_mean(row.get("fraction_feasible_vertices") for row in group_rows),
            "median_fraction_feasible_vertices": _safe_median(row.get("fraction_feasible_vertices") for row in group_rows),
            "mean_active_set_jaccard": _safe_mean(row.get("active_set_jaccard") for row in group_rows),
            "median_active_set_jaccard": _safe_median(row.get("active_set_jaccard") for row in group_rows),
            "mean_true_active_containment": _safe_mean(row.get("true_active_containment") for row in group_rows),
            "median_true_active_containment": _safe_median(row.get("true_active_containment") for row in group_rows),
            "mean_relative_error_best_feasible": _safe_mean(row.get("relative_error") for row in group_rows),
            "median_relative_error_best_feasible": _safe_median(row.get("relative_error") for row in group_rows),
            "failure_rate_no_feasible_vertex": _safe_mean(no_feasible),
            "failure_rate_infeasible_high_objective": _safe_mean(infeasible_artifact),
            "mean_time_to_first_feasible_vertex": _safe_mean(
                row.get("time_to_first_feasible_reconstructed_vertex") for row in group_rows
            ),
            "mean_time_to_first_optimal_vertex": _safe_mean(
                row.get("time_to_first_optimal_reconstructed_vertex") for row in group_rows
            ),
        }
        summary.append(entry)
    return summary


def _plot_lines(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(y_key)
        if value is None or not math.isfinite(float(value)):
            continue
        grouped[str(row["method"])][float(row[x_key])].append(float(value))
    if not grouped:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        for method, points in sorted(grouped.items()):
            xs = sorted(points)
            ys = [float(np.mean(points[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=method)
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_box(rows: list[dict[str, Any]], *, metric: str, output_path: Path, title: str) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = []
    labels = []
    for method in sorted({str(row["method"]) for row in rows}):
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None and math.isfinite(float(row[metric])) and str(row["method"]) == method
        ]
        if values:
            data.append(values)
            labels.append(method)
    if data:
        ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_scaling_plots(records: list[ScalingRecord], output_dir: Path) -> list[Path]:
    rows = [_jsonify(record) for record in records]
    outputs = [
        output_dir / "success_rate_vs_dimension.png",
        output_dir / "relative_error_vs_dimension.png",
        output_dir / "active_set_recovery_vs_dimension.png",
        output_dir / "time_to_good_bound_vs_dimension.png",
        output_dir / "samples_per_second_vs_dimension.png",
        output_dir / "reconstruction_success_vs_dimension.png",
        output_dir / "num_unique_vertices_vs_dimension.png",
        output_dir / "error_distribution_boxplot.png",
    ]
    _plot_lines(rows, x_key="n", y_key="approx_success_1e_2", output_path=outputs[0], title="Success rate vs dimension", ylabel="rate")
    _plot_lines(rows, x_key="n", y_key="relative_error", output_path=outputs[1], title="Relative error vs dimension", ylabel="relative error")
    _plot_lines(rows, x_key="n", y_key="true_active_containment", output_path=outputs[2], title="Active-set recovery vs dimension", ylabel="containment")
    _plot_lines(rows, x_key="n", y_key="time_to_good_bound_1e_2", output_path=outputs[3], title="Time to good bound vs dimension", ylabel="seconds")
    _plot_lines(rows, x_key="n", y_key="samples_per_second", output_path=outputs[4], title="Samples per second vs dimension", ylabel="samples/s")
    _plot_lines(rows, x_key="n", y_key="reconstruction_success", output_path=outputs[5], title="Reconstruction success vs dimension", ylabel="rate")
    _plot_lines(rows, x_key="n", y_key="num_unique_vertices", output_path=outputs[6], title="Unique vertices vs dimension", ylabel="count")
    _plot_box(rows, metric="relative_error", output_path=outputs[7], title="Relative error distribution")
    return outputs


def write_scaling_outputs(
    records: list[ScalingRecord],
    output_root: Path,
    *,
    config_payload: dict[str, Any],
) -> dict[str, Path]:
    results_dir = output_root / "results" / "scaling"
    figures_dir = output_root / "figures" / "scaling"
    rows = [_jsonify(record) for record in records]
    payload = {"config": config_payload, "records": rows}
    summary_by_ratio = _group_summary(rows, key="m_over_n")
    summary_aggregate = _aggregate_scaling_summary(rows)
    results_csv = results_dir / "scaling_results.csv"
    results_json = results_dir / "scaling_results.json"
    summary_dim_csv = results_dir / "scaling_summary_by_dimension.csv"
    summary_dim_json = results_dir / "scaling_summary_by_dimension.json"
    summary_ratio_csv = results_dir / "scaling_summary_by_constraint_ratio.csv"
    _write_csv(results_csv, rows)
    _write_json(results_json, payload)
    _write_csv(summary_dim_csv, summary_aggregate)
    _write_json(summary_dim_json, {"config": config_payload, "records": summary_aggregate})
    _write_csv(summary_ratio_csv, summary_by_ratio)
    make_scaling_plots(records, figures_dir)
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "summary_by_dimension_csv": summary_dim_csv,
        "summary_by_dimension_json": summary_dim_json,
        "summary_by_constraint_ratio_csv": summary_ratio_csv,
        "figure_dir": figures_dir,
    }
