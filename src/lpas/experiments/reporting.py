from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.experiments.benchmark_runner import (
    GEOMETRY_AWARE_METHOD,
    GEOMETRY_AWARE_POLISHED_METHOD,
    MethodExperimentResult,
    NAIVE_MONTE_CARLO_METHOD,
)
from lpas.experiments.metrics import safe_mean
from lpas.experiments.solver_hint_experiment import SolverHintRecord


SCIENTIFIC_FRAMING = (
    "This project investigates whether geometry-aware adaptive sampling can recover useful active-set structure in "
    "dense linear programs faster than naive Monte Carlo sampling. The method is evaluated as a diagnostic and "
    "solver-hint mechanism, not as a replacement for mature LP solvers such as HiGHS."
)

METHOD_ORDER = [NAIVE_MONTE_CARLO_METHOD, GEOMETRY_AWARE_METHOD, GEOMETRY_AWARE_POLISHED_METHOD]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def benchmark_result_row(result: MethodExperimentResult) -> dict[str, object]:
    reference_active_set_size = None
    if result.reference_result.primal_active_mask is not None:
        reference_active_set_size = int(np.sum(np.asarray(result.reference_result.primal_active_mask, dtype=bool)))
    return {
        "problem_name": result.problem_name,
        "family": result.family,
        "method": result.method,
        "seed": result.seed,
        "n_variables": result.n_variables,
        "n_constraints": result.n_constraints,
        "reference_success": result.reference_result.success,
        "reference_objective": result.reference_result.objective,
        "reference_active_set_size": reference_active_set_size,
        "best_feasible_objective": result.best_feasible_objective,
        "best_objective_any_candidate": result.best_objective_any_candidate,
        "best_primal_violation": result.best_primal_violation,
        "best_dual_violation": result.best_dual_violation,
        "best_gap": result.best_gap,
        "best_complementarity_error": result.best_complementarity_error,
        "active_set_recovery_accuracy": result.active_set_recovery_accuracy,
        "exact_active_set_match": result.exact_active_set_match,
        "first_recovery_iteration": result.first_recovery_iteration,
        "time_to_identify_optimal_active_constraints": result.time_to_identify_optimal_active_constraints,
        "wall_clock_seconds": result.wall_clock_seconds,
        "n_samples_total": result.n_samples_total,
        "objective_gap_to_highs": result.objective_gap_to_highs,
        "raw_best_objective": result.raw_best_objective,
        "raw_best_primal_violation": result.raw_best_primal_violation,
        "raw_active_set_similarity": result.raw_active_set_similarity,
        "polished_objective": result.polished_objective,
        "polished_primal_violation": result.polished_primal_violation,
        "polished_active_set_similarity": result.polished_active_set_similarity,
        "polishing_improved_solution": result.polishing_improved_solution,
        "polished_certified_feasible": result.polished_certified_feasible,
        "polishing_wall_clock_seconds": result.polishing_wall_clock_seconds,
        "polishing_candidates_generated": result.polishing_candidates_generated,
        "vertices_reconstructed": result.vertices_reconstructed,
        "vertices_feasible": result.vertices_feasible,
        "solution_source": result.solution_source,
    }


def benchmark_result_payload(result: MethodExperimentResult) -> dict[str, object]:
    return {
        "summary": benchmark_result_row(result),
        "reference": {
            "success": result.reference_result.success,
            "status": result.reference_result.status,
            "message": result.reference_result.message,
            "objective": result.reference_result.objective,
            "primal_active_mask": _jsonify(result.reference_result.primal_active_mask),
        },
        "history": [_jsonify(entry) for entry in result.history],
        "best_scored_entry": None
        if result.best_scored_entry is None
        else {
            "score": result.best_scored_entry.score,
            "primal_objective": result.best_scored_entry.primal_objective,
            "primal_feasible": result.best_scored_entry.primal_feasible,
            "primal_active_mask": _jsonify(result.best_scored_entry.primal_active_mask),
        },
        "polishing": None
        if result.polishing_result is None
        else {
            "best_vertex": None
            if result.polishing_result.best_vertex is None
            else {
                "x": _jsonify(result.polishing_result.best_vertex.x),
                "objective": result.polishing_result.best_vertex.objective,
                "feasible": result.polishing_result.best_vertex.feasible,
                "active_indices": _jsonify(result.polishing_result.best_vertex.active_indices),
                "original_active_mask": _jsonify(result.polishing_result.best_vertex.original_active_mask),
            },
            "candidates_tried": result.polishing_result.candidates_tried,
            "candidates_feasible": result.polishing_result.candidates_feasible,
            "diagnostics": _jsonify(result.polishing_result.diagnostics),
        },
    }


def solver_hint_row(record: SolverHintRecord) -> dict[str, object]:
    return {
        "problem_name": record.problem_name,
        "family": record.family,
        "method": record.method,
        "seed": record.seed,
        "n_variables": record.n_variables,
        "n_constraints": record.n_constraints,
        "hint_active_set_jaccard": record.hint_active_set_jaccard,
        "hint_corner_feasible": record.hint_corner_feasible,
        "hint_corner_objective": record.hint_corner_objective,
        "objective_gap_to_highs": record.objective_gap_to_highs,
        "constraints_in_top_k_support": record.constraints_in_top_k_support,
        "reconstruction_success": record.reconstruction_success,
        "wall_clock_seconds": record.wall_clock_seconds,
        "top_k": record.top_k,
        "reference_objective": record.reference_objective,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")


def _group_rows(rows: list[dict[str, object]], *keys: str) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _aggregate_numeric_rows(rows: list[dict[str, object]], *keys: str, metrics: list[str]) -> list[dict[str, object]]:
    aggregated: list[dict[str, object]] = []
    for group_key, group_rows in _group_rows(rows, *keys).items():
        record = {key: value for key, value in zip(keys, group_key, strict=True)}
        record["count"] = len(group_rows)
        for metric in metrics:
            values = [row.get(metric) for row in group_rows]
            record[f"{metric}_mean"] = safe_mean(values)
        aggregated.append(record)
    aggregated.sort(key=lambda row: tuple(str(row[key]) for key in keys))
    return aggregated


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _method_label(method: str) -> str:
    if method == GEOMETRY_AWARE_METHOD:
        return "Geometry-aware (raw)"
    if method == GEOMETRY_AWARE_POLISHED_METHOD:
        return "Geometry-aware + polishing"
    if method == NAIVE_MONTE_CARLO_METHOD:
        return "Naive Monte Carlo"
    return method


def _present_methods(rows: list[dict[str, object]]) -> list[str]:
    methods = {str(row["method"]) for row in rows}
    ordered = [method for method in METHOD_ORDER if method in methods]
    extras = sorted(methods.difference(METHOD_ORDER))
    return ordered + extras


def _history_payloads(results_or_payload: list[MethodExperimentResult] | dict[str, object]) -> list[dict[str, object]]:
    if isinstance(results_or_payload, dict):
        return list(results_or_payload["results"])
    return [benchmark_result_payload(result) for result in results_or_payload]


def _method_color(method: str) -> str:
    if method == NAIVE_MONTE_CARLO_METHOD:
        return "#9a3412"
    if method == GEOMETRY_AWARE_METHOD:
        return "#0f766e"
    if method == GEOMETRY_AWARE_POLISHED_METHOD:
        return "#2563eb"
    return "#4b5563"


def _plot_average_history(payloads: list[dict[str, object]], metric: str, ylabel: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for payload in payloads:
        method = str(payload["summary"]["method"])
        history = payload["history"]
        grouped[method].append([float(entry[metric]) for entry in history])
    if not grouped:
        ax.text(0.5, 0.5, "No benchmark histories available", ha="center", va="center")
    else:
        ordered_methods = [method for method in METHOD_ORDER if method in grouped] + sorted(set(grouped).difference(METHOD_ORDER))
        for method in ordered_methods:
            curves = grouped[method]
            max_len = max(len(curve) for curve in curves)
            matrix = np.full((len(curves), max_len), np.nan, dtype=float)
            for row_index, curve in enumerate(curves):
                matrix[row_index, : len(curve)] = curve
            mean_curve = np.nanmean(matrix, axis=0)
            ax.plot(
                np.arange(1, max_len + 1),
                mean_curve,
                label=_method_label(method),
                color=_method_color(method),
                linestyle="--" if method == GEOMETRY_AWARE_POLISHED_METHOD else "-",
            )
        ax.legend(loc="best")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_histogram(rows: list[dict[str, object]], metric: str, title: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False
    for method in _present_methods(rows):
        values = [
            float(row[metric])
            for row in rows
            if row["method"] == method and row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        if values:
            any_data = True
            ax.hist(values, bins=min(10, max(3, len(values))), alpha=0.55, label=_method_label(method), color=_method_color(method))
    if not any_data:
        ax.text(0.5, 0.5, "No successful recoveries in this run", ha="center", va="center")
    else:
        ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_runtime_bar(rows: list[dict[str, object]], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    methods = _present_methods(rows)
    means = [
        safe_mean(row["wall_clock_seconds"] for row in rows if row["method"] == method)
        for method in methods
    ]
    ax.bar([_method_label(method) for method in methods], means, color=[_method_color(method) for method in methods])
    ax.set_ylabel("seconds")
    ax.set_title("Runtime comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_random_dense_plots(results_or_payload: list[MethodExperimentResult] | dict[str, object], output_dir: Path) -> list[Path]:
    payloads = _history_payloads(results_or_payload)
    rows = [payload["summary"] for payload in payloads]
    ensure_directory(output_dir)
    outputs = [
        output_dir / "best_primal_violation_vs_iteration.png",
        output_dir / "best_dual_violation_vs_iteration.png",
        output_dir / "best_complementarity_error_vs_iteration.png",
        output_dir / "active_set_recovery_vs_iteration.png",
        output_dir / "time_to_active_set_recovery.png",
        output_dir / "runtime_comparison.png",
        output_dir / "objective_gap_to_highs.png",
    ]
    _plot_average_history(payloads, "best_primal_violation", "Best primal violation", outputs[0])
    _plot_average_history(payloads, "best_dual_violation", "Best dual violation", outputs[1])
    _plot_average_history(payloads, "best_complementarity_error", "Best complementarity error", outputs[2])
    _plot_average_history(payloads, "active_set_recovery_accuracy", "Active-set recovery accuracy", outputs[3])
    _plot_histogram(rows, "time_to_identify_optimal_active_constraints", "Time to active-set recovery", outputs[4])
    _plot_runtime_bar(rows, outputs[5])
    _plot_histogram(rows, "objective_gap_to_highs", "Objective gap to HiGHS", outputs[6])
    return outputs


def _random_dense_summary_text(rows: list[dict[str, object]], summary_rows: list[dict[str, object]], config_payload: dict[str, object]) -> str:
    overall_columns = [
        "method",
        "count",
        "raw_best_objective_mean",
        "polished_objective_mean",
        "active_set_recovery_accuracy_mean",
        "polishing_wall_clock_seconds_mean",
        "time_to_identify_optimal_active_constraints_mean",
        "objective_gap_to_highs_mean",
        "wall_clock_seconds_mean",
    ]
    display_rows = [{**row, "method": _method_label(str(row["method"]))} for row in summary_rows]
    family_rows = _aggregate_numeric_rows(
        rows,
        "method",
        "family",
        metrics=[
            "raw_best_objective",
            "polished_objective",
            "active_set_recovery_accuracy",
            "objective_gap_to_highs",
            "wall_clock_seconds",
        ],
    )
    family_display = [{**row, "method": _method_label(str(row["method"]))} for row in family_rows]

    interpretation = "The benchmark did not show a clear advantage from the polishing step on this run."
    if len(summary_rows) >= 2:
        by_method = {str(row["method"]): row for row in summary_rows}
        geometry_raw = by_method.get(GEOMETRY_AWARE_METHOD)
        geometry_polished = by_method.get(GEOMETRY_AWARE_POLISHED_METHOD)
        naive = by_method.get(NAIVE_MONTE_CARLO_METHOD)
        if geometry_raw and geometry_polished:
            raw_gap = geometry_raw.get("objective_gap_to_highs_mean")
            polished_gap = geometry_polished.get("objective_gap_to_highs_mean")
            if isinstance(raw_gap, float) and isinstance(polished_gap, float):
                if polished_gap < raw_gap:
                    interpretation = (
                        "Vertex polishing reduced the mean objective gap after geometry-aware sampling while keeping the scientific framing as a solver-hint refinement step."
                    )
                elif polished_gap > raw_gap:
                    interpretation = (
                        "The polishing pass did not improve objective quality on this run; the raw geometry-aware archive remained the stronger diagnostic signal."
                    )
        elif geometry_raw and naive:
            geometry_jaccard = geometry_raw.get("active_set_recovery_accuracy_mean")
            naive_jaccard = naive.get("active_set_recovery_accuracy_mean")
            if isinstance(geometry_jaccard, float) and isinstance(naive_jaccard, float) and geometry_jaccard > naive_jaccard:
                interpretation = (
                    "Geometry-aware sampling recovered the optimal active set earlier or more reliably on average under the same sample budget."
                )

    return "\n".join(
        [
            "# Random Dense Benchmark",
            "",
            SCIENTIFIC_FRAMING,
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(_jsonify(config_payload), indent=2),
            "```",
            "",
            "## Overall Method Comparison",
            "",
            _markdown_table(display_rows, overall_columns),
            "",
            "## Family Breakdown",
            "",
            _markdown_table(
                family_display,
                [
                    "method",
                    "family",
                    "count",
                    "raw_best_objective_mean",
                    "polished_objective_mean",
                    "active_set_recovery_accuracy_mean",
                    "objective_gap_to_highs_mean",
                    "wall_clock_seconds_mean",
                ],
            ),
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )


def write_random_dense_outputs(results: list[MethodExperimentResult], output_root: Path, *, config_payload: dict[str, object]) -> dict[str, Path]:
    benchmark_dir = output_root / "benchmarks" / "random_dense"
    figure_dir = output_root / "figures" / "benchmarks"
    rows = [benchmark_result_row(result) for result in results]
    payload = {
        "framing": SCIENTIFIC_FRAMING,
        "config": config_payload,
        "results": [benchmark_result_payload(result) for result in results],
    }
    summary_rows = _aggregate_numeric_rows(
        rows,
        "method",
        metrics=[
            "raw_best_objective",
            "polished_objective",
            "active_set_recovery_accuracy",
            "polishing_wall_clock_seconds",
            "time_to_identify_optimal_active_constraints",
            "objective_gap_to_highs",
            "wall_clock_seconds",
        ],
    )
    csv_path = benchmark_dir / "results.csv"
    json_path = benchmark_dir / "results.json"
    summary_csv_path = benchmark_dir / "summary.csv"
    summary_md_path = benchmark_dir / "summary.md"
    _write_csv(csv_path, rows)
    _write_json(json_path, payload)
    _write_csv(summary_csv_path, summary_rows)
    summary_md_path.write_text(_random_dense_summary_text(rows, summary_rows, config_payload), encoding="utf-8")
    make_random_dense_plots(results, figure_dir)
    return {
        "results_csv": csv_path,
        "results_json": json_path,
        "summary_csv": summary_csv_path,
        "summary_md": summary_md_path,
        "figure_dir": figure_dir,
    }


def load_report_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_solver_hint_jaccard(rows: list[dict[str, object]], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = []
    labels = []
    for method in _present_methods(rows):
        values = [float(row["hint_active_set_jaccard"]) for row in rows if row["method"] == method]
        if values:
            data.append(values)
            labels.append(_method_label(method))
    if data:
        ax.boxplot(data, labels=labels)
    else:
        ax.text(0.5, 0.5, "No hint data available", ha="center", va="center")
    ax.set_title("Hint active-set Jaccard distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_solver_hint_objective_gap(rows: list[dict[str, object]], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False
    for method in _present_methods(rows):
        values = [
            float(row["objective_gap_to_highs"])
            for row in rows
            if row["method"] == method and row["objective_gap_to_highs"] is not None
        ]
        if values:
            any_data = True
            ax.hist(values, bins=min(10, max(3, len(values))), alpha=0.55, label=_method_label(method), color=_method_color(method))
    if any_data:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No feasible reconstructed corners", ha="center", va="center")
    ax.set_title("Objective gap to HiGHS")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_solver_hint_success(rows: list[dict[str, object]], metric: str, title: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped = _aggregate_numeric_rows(rows, "method", "n_variables", metrics=[metric])
    if not grouped:
        ax.text(0.5, 0.5, "No hint records available", ha="center", va="center")
    else:
        methods = [_method_label(str(row["method"])) for row in grouped]
        labels = [f"{method}\nn={row['n_variables']}" for method, row in zip(methods, grouped, strict=True)]
        values = [float(row[f"{metric}_mean"]) for row in grouped]
        ax.bar(labels, values, color=[_method_color(str(row["method"])) for row in grouped])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_solver_hint_plots(records: list[SolverHintRecord], output_dir: Path) -> list[Path]:
    rows = [solver_hint_row(record) for record in records]
    ensure_directory(output_dir)
    outputs = [
        output_dir / "hint_active_set_jaccard_distribution.png",
        output_dir / "objective_gap_to_highs.png",
        output_dir / "reconstruction_success_rate_by_dimension.png",
        output_dir / "top_k_active_constraint_containment.png",
    ]
    _plot_solver_hint_jaccard(rows, outputs[0])
    _plot_solver_hint_objective_gap(rows, outputs[1])
    _plot_solver_hint_success(rows, "reconstruction_success", "Reconstruction success rate by dimension", outputs[2])
    _plot_solver_hint_success(rows, "constraints_in_top_k_support", "Top-k active-constraint containment", outputs[3])
    return outputs


def _solver_hint_summary_text(rows: list[dict[str, object]], summary_rows: list[dict[str, object]], config_payload: dict[str, object]) -> str:
    display_rows = [{**row, "method": _method_label(str(row["method"]))} for row in summary_rows]
    interpretation = "The solver-hint layer behaved as a diagnostic experiment rather than a direct solver replacement in this run."
    if len(summary_rows) >= 2:
        by_method = {str(row["method"]): row for row in summary_rows}
        geometry = by_method.get(GEOMETRY_AWARE_METHOD)
        polished = by_method.get(GEOMETRY_AWARE_POLISHED_METHOD)
        naive = by_method.get(NAIVE_MONTE_CARLO_METHOD)
        if polished and geometry:
            polished_success = polished.get("reconstruction_success_mean")
            geometry_success = geometry.get("reconstruction_success_mean")
            if isinstance(polished_success, float) and isinstance(geometry_success, float) and polished_success > geometry_success:
                interpretation = "The explicit polishing stage converted more geometry-aware hints into feasible corners than the raw archive alone on this run."
        elif geometry and naive:
            geo_success = geometry.get("reconstruction_success_mean")
            naive_success = naive.get("reconstruction_success_mean")
            if isinstance(geo_success, float) and isinstance(naive_success, float):
                if geo_success > naive_success:
                    interpretation = "Geometry-aware sampling produced more feasible reconstructed corners than naive Monte Carlo on this run."
                elif geo_success < naive_success:
                    interpretation = "Naive Monte Carlo produced more feasible reconstructed corners on this run; geometry-aware sampling was still useful as an active-set diagnostic."
    return "\n".join(
        [
            "# Solver Hint Experiment",
            "",
            SCIENTIFIC_FRAMING,
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(_jsonify(config_payload), indent=2),
            "```",
            "",
            "## Overall Method Comparison",
            "",
            _markdown_table(
                display_rows,
                [
                    "method",
                    "count",
                    "hint_active_set_jaccard_mean",
                    "constraints_in_top_k_support_mean",
                    "reconstruction_success_mean",
                    "objective_gap_to_highs_mean",
                    "wall_clock_seconds_mean",
                ],
            ),
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )


def write_solver_hint_outputs(records: list[SolverHintRecord], output_root: Path, *, config_payload: dict[str, object]) -> dict[str, Path]:
    benchmark_dir = output_root / "benchmarks" / "solver_hints"
    figure_dir = output_root / "figures" / "solver_hints"
    rows = [solver_hint_row(record) for record in records]
    summary_rows = _aggregate_numeric_rows(
        rows,
        "method",
        metrics=[
            "hint_active_set_jaccard",
            "constraints_in_top_k_support",
            "reconstruction_success",
            "objective_gap_to_highs",
            "wall_clock_seconds",
        ],
    )
    csv_path = benchmark_dir / "results.csv"
    json_path = benchmark_dir / "results.json"
    summary_csv_path = benchmark_dir / "summary.csv"
    summary_md_path = benchmark_dir / "summary.md"
    _write_csv(csv_path, rows)
    _write_json(json_path, {"framing": SCIENTIFIC_FRAMING, "config": config_payload, "results": [_jsonify(record) for record in records]})
    _write_csv(summary_csv_path, summary_rows)
    summary_md_path.write_text(_solver_hint_summary_text(rows, summary_rows, config_payload), encoding="utf-8")
    make_solver_hint_plots(records, figure_dir)
    return {
        "results_csv": csv_path,
        "results_json": json_path,
        "summary_csv": summary_csv_path,
        "summary_md": summary_md_path,
        "figure_dir": figure_dir,
    }
