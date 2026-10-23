from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.corners.multi_corner_discovery import recover_soft_active_set
from lpas.experiments.benchmark_runner import GEOMETRY_AWARE_POLISHED_METHOD, GEOMETRY_AWARE_METHOD, run_problem_comparison
from lpas.experiments.metrics import relative_objective_error
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.solver.educational_simplex import origin_basis, solve_educational_simplex
from lpas.solver.scipy_handoff import solve_with_scipy
from lpas.solver.vertex_polishing import reconstruct_vertex_from_active_set
from lpas.utils.config import SolverConfig

SCIPY_HIGHS_ONLY_METHOD = "scipy_highs_only"
SCIPY_DIAGNOSTICS_METHOD = "sampler_seed_diagnostics_plus_scipy_highs"
SAMPLER_RAW_ONLY_METHOD = "sampler_raw_only"
SAMPLER_PLUS_POLISHING_METHOD = "sampler_plus_polishing"
SAMPLER_PLUS_VERTEX_POLISHING_METHOD = "sampler_plus_vertex_polishing"
UNSEEDED_SIMPLEX_METHOD = "unseeded_educational_simplex"
SEEDED_FROM_SAMPLED_METHOD = "seeded_educational_simplex_from_sampled_active_set"
SEEDED_FROM_POLISHED_METHOD = "seeded_educational_simplex_from_polished_vertex"


@dataclass(frozen=True)
class SolverSeedingRecord:
    problem_id: str
    method_name: str
    backend: str
    n: int
    m: int
    seed: int
    sampling_time_seconds: float
    polishing_time_seconds: float
    active_set_recovery_time_seconds: float | None
    seed_build_time_seconds: float
    solver_time_seconds: float
    total_time_seconds: float
    unseeded_solver_time_seconds: float
    speedup_solver_only: float | None
    speedup_total: float | None
    objective: float | None
    relative_error: float | None
    raw_gap: float | None
    certified_gap: float | None
    primal_violation: float | None
    dual_violation: float | None
    status: str


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


def _speedup(reference_time: float, candidate_time: float) -> float | None:
    if candidate_time <= 0.0:
        return None
    return float(reference_time / candidate_time)


def _single_sample_vertex(problem, sample: np.ndarray | None):
    if sample is None:
        return None
    basis = recover_soft_active_set(problem, sample)
    if len(basis) < problem.n:
        return None
    return reconstruct_vertex_from_active_set(problem, basis)


def _record(
    *,
    problem_id: str,
    method_name: str,
    backend: str,
    n: int,
    m: int,
    seed: int,
    sampling_time_seconds: float,
    polishing_time_seconds: float,
    active_set_recovery_time_seconds: float | None,
    seed_build_time_seconds: float,
    solver_time_seconds: float,
    total_time_seconds: float,
    unseeded_solver_time_seconds: float,
    objective: float | None,
    reference_objective: float | None,
    raw_gap: float | None,
    certified_gap: float | None,
    primal_violation: float | None,
    dual_violation: float | None,
    status: str,
) -> SolverSeedingRecord:
    return SolverSeedingRecord(
        problem_id=problem_id,
        method_name=method_name,
        backend=backend,
        n=n,
        m=m,
        seed=seed,
        sampling_time_seconds=float(sampling_time_seconds),
        polishing_time_seconds=float(polishing_time_seconds),
        active_set_recovery_time_seconds=active_set_recovery_time_seconds,
        seed_build_time_seconds=float(seed_build_time_seconds),
        solver_time_seconds=float(solver_time_seconds),
        total_time_seconds=float(total_time_seconds),
        unseeded_solver_time_seconds=float(unseeded_solver_time_seconds),
        speedup_solver_only=_speedup(unseeded_solver_time_seconds, solver_time_seconds) if solver_time_seconds > 0.0 else None,
        speedup_total=_speedup(unseeded_solver_time_seconds, total_time_seconds) if total_time_seconds > 0.0 else None,
        objective=objective,
        relative_error=relative_objective_error(objective, reference_objective),
        raw_gap=raw_gap,
        certified_gap=certified_gap,
        primal_violation=primal_violation,
        dual_violation=dual_violation,
        status=status,
    )


def benchmark_solver_seeding_problem(
    *,
    problem_id: str,
    problem,
    config: SolverConfig,
    backend: str = "numpy_cpu",
) -> list[SolverSeedingRecord]:
    reference_start = time.perf_counter()
    scipy_reference = solve_with_scipy(problem)
    scipy_runtime = time.perf_counter() - reference_start

    method_results = run_problem_comparison(
        problem_name=problem_id,
        family="solver_seeding",
        problem=problem,
        config=config,
        capture_samples=False,
    )
    geometry_result = next(result for result in method_results if result.method == GEOMETRY_AWARE_METHOD)
    polished_result = next(result for result in method_results if result.method == GEOMETRY_AWARE_POLISHED_METHOD)

    unseeded_simplex = solve_educational_simplex(problem, initial_basis=origin_basis(problem))
    unseeded_runtime = float(unseeded_simplex.runtime_seconds)

    single_vertex_start = time.perf_counter()
    single_vertex = _single_sample_vertex(problem.to_maximization(), geometry_result.final_x)
    single_vertex_seconds = time.perf_counter() - single_vertex_start

    raw_seed_start = time.perf_counter()
    raw_seed_basis = None if geometry_result.final_x is None else recover_soft_active_set(problem.to_maximization(), geometry_result.final_x)
    raw_seed_seconds = time.perf_counter() - raw_seed_start

    seeded_from_raw = None
    if raw_seed_basis is not None and len(raw_seed_basis) == problem.n:
        seeded_from_raw = solve_educational_simplex(problem, initial_basis=raw_seed_basis)

    polished_seed_start = time.perf_counter()
    polished_seed_basis = None
    if polished_result.polishing_result is not None and polished_result.polishing_result.best_vertex is not None:
        polished_seed_basis = polished_result.polishing_result.best_vertex.active_indices
    polished_seed_seconds = time.perf_counter() - polished_seed_start

    seeded_from_polished = None
    if polished_seed_basis is not None:
        seeded_from_polished = solve_educational_simplex(problem, initial_basis=polished_seed_basis)

    records = [
        _record(
            problem_id=problem_id,
            method_name=SCIPY_HIGHS_ONLY_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=0.0,
            polishing_time_seconds=0.0,
            active_set_recovery_time_seconds=None,
            seed_build_time_seconds=0.0,
            solver_time_seconds=scipy_runtime,
            total_time_seconds=scipy_runtime,
            unseeded_solver_time_seconds=scipy_runtime,
            objective=scipy_reference.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=None,
            certified_gap=0.0 if scipy_reference.success else None,
            primal_violation=0.0 if scipy_reference.success else None,
            dual_violation=0.0 if scipy_reference.success else None,
            status="success" if scipy_reference.success else f"status_{scipy_reference.status}",
        ),
        _record(
            problem_id=problem_id,
            method_name=SAMPLER_RAW_ONLY_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=0.0,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=0.0,
            solver_time_seconds=0.0,
            total_time_seconds=geometry_result.wall_clock_seconds,
            unseeded_solver_time_seconds=scipy_runtime,
            objective=geometry_result.best_feasible_objective,
            reference_objective=scipy_reference.objective,
            raw_gap=geometry_result.best_raw_gap,
            certified_gap=geometry_result.best_certified_gap,
            primal_violation=geometry_result.best_primal_violation,
            dual_violation=geometry_result.best_dual_violation,
            status="success" if geometry_result.best_feasible_objective is not None else "no_feasible_sample",
        ),
        _record(
            problem_id=problem_id,
            method_name=SAMPLER_PLUS_POLISHING_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=single_vertex_seconds,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=0.0,
            solver_time_seconds=0.0,
            total_time_seconds=geometry_result.wall_clock_seconds + single_vertex_seconds,
            unseeded_solver_time_seconds=scipy_runtime,
            objective=None if single_vertex is None else single_vertex.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=geometry_result.best_raw_gap,
            certified_gap=None
            if single_vertex is None or not single_vertex.feasible or geometry_result.best_feasible_dual_upper_bound is None
            else geometry_result.best_feasible_dual_upper_bound - single_vertex.objective,
            primal_violation=None if single_vertex is None else single_vertex.primal_violation,
            dual_violation=geometry_result.best_dual_violation,
            status="success" if single_vertex is not None else "no_polished_vertex",
        ),
        _record(
            problem_id=problem_id,
            method_name=SAMPLER_PLUS_VERTEX_POLISHING_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=polished_result.polishing_wall_clock_seconds or 0.0,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=0.0,
            solver_time_seconds=0.0,
            total_time_seconds=polished_result.wall_clock_seconds,
            unseeded_solver_time_seconds=scipy_runtime,
            objective=polished_result.best_feasible_objective,
            reference_objective=scipy_reference.objective,
            raw_gap=polished_result.best_raw_gap,
            certified_gap=polished_result.best_certified_gap,
            primal_violation=polished_result.best_primal_violation,
            dual_violation=polished_result.best_dual_violation,
            status="success" if polished_result.best_feasible_objective is not None else "no_polished_vertex",
        ),
        _record(
            problem_id=problem_id,
            method_name=UNSEEDED_SIMPLEX_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=0.0,
            polishing_time_seconds=0.0,
            active_set_recovery_time_seconds=None,
            seed_build_time_seconds=0.0,
            solver_time_seconds=unseeded_runtime,
            total_time_seconds=unseeded_runtime,
            unseeded_solver_time_seconds=unseeded_runtime,
            objective=unseeded_simplex.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=None,
            certified_gap=None,
            primal_violation=0.0 if unseeded_simplex.x is not None else None,
            dual_violation=None,
            status=unseeded_simplex.status,
        ),
        _record(
            problem_id=problem_id,
            method_name=SCIPY_DIAGNOSTICS_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=0.0,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=raw_seed_seconds,
            solver_time_seconds=scipy_runtime,
            total_time_seconds=geometry_result.wall_clock_seconds + raw_seed_seconds + scipy_runtime,
            unseeded_solver_time_seconds=scipy_runtime,
            objective=scipy_reference.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=geometry_result.best_raw_gap,
            certified_gap=0.0 if scipy_reference.success else None,
            primal_violation=0.0 if scipy_reference.success else None,
            dual_violation=0.0 if scipy_reference.success else None,
            status="success" if scipy_reference.success else f"status_{scipy_reference.status}",
        ),
        _record(
            problem_id=problem_id,
            method_name=SEEDED_FROM_SAMPLED_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=0.0,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=raw_seed_seconds,
            solver_time_seconds=0.0 if seeded_from_raw is None else seeded_from_raw.runtime_seconds,
            total_time_seconds=geometry_result.wall_clock_seconds + raw_seed_seconds + (0.0 if seeded_from_raw is None else seeded_from_raw.runtime_seconds),
            unseeded_solver_time_seconds=unseeded_runtime,
            objective=None if seeded_from_raw is None else seeded_from_raw.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=geometry_result.best_raw_gap,
            certified_gap=geometry_result.best_certified_gap,
            primal_violation=geometry_result.best_primal_violation,
            dual_violation=geometry_result.best_dual_violation,
            status="missing_seed" if seeded_from_raw is None else seeded_from_raw.status,
        ),
        _record(
            problem_id=problem_id,
            method_name=SEEDED_FROM_POLISHED_METHOD,
            backend=backend,
            n=problem.n,
            m=problem.m,
            seed=config.seed or 0,
            sampling_time_seconds=geometry_result.wall_clock_seconds,
            polishing_time_seconds=polished_result.polishing_wall_clock_seconds or 0.0,
            active_set_recovery_time_seconds=geometry_result.time_to_identify_optimal_active_constraints,
            seed_build_time_seconds=polished_seed_seconds,
            solver_time_seconds=0.0 if seeded_from_polished is None else seeded_from_polished.runtime_seconds,
            total_time_seconds=geometry_result.wall_clock_seconds + (polished_result.polishing_wall_clock_seconds or 0.0) + polished_seed_seconds + (0.0 if seeded_from_polished is None else seeded_from_polished.runtime_seconds),
            unseeded_solver_time_seconds=unseeded_runtime,
            objective=None if seeded_from_polished is None else seeded_from_polished.objective,
            reference_objective=scipy_reference.objective,
            raw_gap=polished_result.best_raw_gap,
            certified_gap=polished_result.best_certified_gap,
            primal_violation=polished_result.best_primal_violation,
            dual_violation=polished_result.best_dual_violation,
            status="missing_seed" if seeded_from_polished is None else seeded_from_polished.status,
        ),
    ]
    return records


def run_solver_seeding_total_time_benchmark(
    *,
    config: SolverConfig,
    dimensions: Iterable[int] = (2, 3, 4),
    ratios: Iterable[int] = (2, 4),
    seeds: Iterable[int] = range(3),
    backend: str = "numpy_cpu",
) -> list[SolverSeedingRecord]:
    if backend != "numpy_cpu":
        raise ValueError("The solver-seeding total-time benchmark currently supports only the numpy_cpu path.")
    records: list[SolverSeedingRecord] = []
    for n in dimensions:
        for ratio in ratios:
            m = int(ratio) * int(n)
            for seed in seeds:
                generated = generate_random_feasible_bounded_lp(n, m, int(seed))
                records.extend(
                    benchmark_solver_seeding_problem(
                        problem_id=generated.name,
                        problem=generated.problem,
                        config=config,
                        backend=backend,
                    )
                )
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


def _method_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        SCIPY_HIGHS_ONLY_METHOD,
        SCIPY_DIAGNOSTICS_METHOD,
        SAMPLER_RAW_ONLY_METHOD,
        SAMPLER_PLUS_POLISHING_METHOD,
        SAMPLER_PLUS_VERTEX_POLISHING_METHOD,
        UNSEEDED_SIMPLEX_METHOD,
        SEEDED_FROM_SAMPLED_METHOD,
        SEEDED_FROM_POLISHED_METHOD,
    ]
    present = {str(row["method_name"]) for row in rows}
    return [method for method in preferred if method in present] + sorted(present - set(preferred))


def _mean_by_method(rows: list[dict[str, Any]], metric: str) -> tuple[list[str], list[float]]:
    means: list[float] = []
    methods = _method_order(rows)
    filtered_methods: list[str] = []
    for method in methods:
        values = [
            float(row[metric])
            for row in rows
            if str(row["method_name"]) == method and row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        if not values:
            continue
        filtered_methods.append(method)
        means.append(float(np.mean(values)))
    return filtered_methods, means


def _plot_bars(rows: list[dict[str, Any]], *, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods, means = _mean_by_method(rows, metric)
    if not methods:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        ax.bar(methods, means)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_metric_vs_dimension(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    _ensure_directory(output_path.parent)
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        method = str(row["method_name"])
        n = int(row["n"])
        grouped.setdefault(method, {}).setdefault(n, []).append(float(value))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for method in _method_order(rows):
        points = grouped.get(method)
        if not points:
            continue
        xs = sorted(points)
        ys = [float(np.mean(points[x])) for x in xs]
        ax.plot(xs, ys, marker="o", label=method)
        plotted = True
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_time_breakdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    _ensure_directory(output_path.parent)
    components = [
        ("sampling_time_seconds", "sampling"),
        ("polishing_time_seconds", "polishing"),
        ("seed_build_time_seconds", "seed build"),
        ("solver_time_seconds", "solver"),
    ]
    methods = _method_order(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    bottoms = np.zeros(len(methods), dtype=float)
    for metric, label in components:
        values = []
        for method in methods:
            component_values = [
                float(row[metric])
                for row in rows
                if str(row["method_name"]) == method
                and row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ]
            values.append(float(np.mean(component_values)) if component_values else 0.0)
        if any(value > 0.0 for value in values):
            ax.bar(methods, values, bottom=bottoms, label=label)
            bottoms = bottoms + np.asarray(values, dtype=float)
            plotted = True
    if plotted:
        ax.legend(loc="best")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title("Time breakdown by method")
    ax.set_ylabel("seconds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_objective_error_vs_time(rows: list[dict[str, Any]], output_path: Path) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for method in _method_order(rows):
        points = [
            (float(row["total_time_seconds"]), float(row["relative_error"]))
            for row in rows
            if str(row["method_name"]) == method
            and row.get("total_time_seconds") is not None
            and row.get("relative_error") is not None
            and math.isfinite(float(row["total_time_seconds"]))
            and math.isfinite(float(row["relative_error"]))
        ]
        if not points:
            continue
        plotted = True
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(xs, ys, label=method, alpha=0.7)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title("Objective error vs total time")
    ax.set_xlabel("total time (s)")
    ax.set_ylabel("relative error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_solver_seeding_plots(records: list[SolverSeedingRecord], output_dir: Path) -> list[Path]:
    rows = [_jsonify(record) for record in records]
    outputs = [
        output_dir / "solver_time_with_without_seed.png",
        output_dir / "total_time_with_without_seed.png",
        output_dir / "speedup_solver_only_vs_dimension.png",
        output_dir / "speedup_total_vs_dimension.png",
        output_dir / "time_breakdown_stacked_bar.png",
        output_dir / "objective_error_vs_time.png",
    ]
    _plot_bars(rows, metric="solver_time_seconds", output_path=outputs[0], title="Solver time with and without seed", ylabel="seconds")
    _plot_bars(rows, metric="total_time_seconds", output_path=outputs[1], title="Total time with and without seed", ylabel="seconds")
    _plot_metric_vs_dimension(rows, metric="speedup_solver_only", output_path=outputs[2], title="Solver-only speedup vs dimension", ylabel="speedup")
    _plot_metric_vs_dimension(rows, metric="speedup_total", output_path=outputs[3], title="Total speedup vs dimension", ylabel="speedup")
    _plot_time_breakdown(rows, output_path=outputs[4])
    _plot_objective_error_vs_time(rows, output_path=outputs[5])
    return outputs


def write_solver_seeding_outputs(
    records: list[SolverSeedingRecord],
    output_root: Path,
    *,
    config_payload: dict[str, Any],
) -> dict[str, Path]:
    results_dir = output_root / "results" / "solver_seeding"
    figures_dir = output_root / "figures" / "solver_seeding"
    rows = [_jsonify(record) for record in records]
    results_csv = results_dir / "solver_seeding_total_time_results.csv"
    results_json = results_dir / "solver_seeding_total_time_results.json"
    _write_csv(results_csv, rows)
    _write_json(
        results_json,
        {
            "config": config_payload,
            "interpretation": (
                "HiGHS is used as a reference solver and diagnostic baseline. "
                "These measurements do not claim a verified HiGHS warm-start interface."
            ),
            "records": rows,
        },
    )
    make_solver_seeding_plots(records, figures_dir)
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "figure_dir": figures_dir,
    }
