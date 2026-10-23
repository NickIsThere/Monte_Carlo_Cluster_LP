"""Matched-time sampling / PDHG / hybrid comparison, without vertex polishing."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy

from lpas.core.active_set import augmented_active_mask
from lpas.core.primal_dual import evaluate_primal_dual_pairs, normalized_kkt_residual
from lpas.experiments.metrics import active_set_jaccard, relative_objective_error
from lpas.experiments.random_dense import (
    generate_bounded_dense_lp, generate_controlled_optimum_lp, generate_narrow_region_lp,
)
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.solver.scipy_handoff import solve_with_scipy
from lpas.utils.config import PDHGConfig, PrimalRecoveryConfig, SamplerConfig, SolverConfig, VertexPolishingConfig, WarmStartConfig


METHODS = ("sampling", "pdhg", "hybrid")


def compare_problem(
    problem, *, problem_id: str, family: str, seed: int, time_budget: float,
    batch_size: int = 64, pdhg_steps: int = 5, repeat: int = 0,
    recovery: bool = False, recovery_steps: int = 256, recovery_candidates: int = 4,
    recovery_fraction: float = 0.1, history_limit: int | None = None,
) -> list[dict]:
    reference = solve_with_scipy(problem)
    if not reference.success or reference.x is None:
        raise ValueError("Comparison requires a feasible, bounded reference problem")
    config = SolverConfig(
        batch_size=batch_size, max_iter=1_000_000, patience=1_000_000,
        seed=seed, time_limit_seconds=time_budget, active_tol=1e-4,
        sampler=SamplerConfig(seed=seed, sigma_init=1.35, primal_init_mean=0.8, dual_init_mean=0.8),
        pdhg=PDHGConfig(steps_per_iteration=pdhg_steps),
        vertex_polishing=VertexPolishingConfig(enabled=False),
        warm_start=WarmStartConfig(max_combinations=0),
        stop_on_variance_collapse=False,
        recovery=PrimalRecoveryConfig(enabled=recovery, max_steps=recovery_steps, max_candidates=recovery_candidates, time_fraction=recovery_fraction),
    )
    canonical = problem.to_maximization()
    reference_mask = augmented_active_mask(canonical, reference.x, config.active_tol)
    records = []
    # Shuffle execution order, retaining a deterministic order for each repeat.
    order = np.random.default_rng(seed + 10_007 * repeat).permutation(METHODS)
    for method in order:
        start = time.perf_counter()
        result = AdaptiveLPSolver(replace(config, search_mode=str(method))).solve(problem)
        total_seconds = time.perf_counter() - start
        history = result.history
        if history_limit is not None and len(history) > history_limit:
            if history_limit < 2:
                raise ValueError("history_limit must be at least two")
            history = [history[int(index)] for index in np.linspace(0, len(history) - 1, history_limit)]
        certificate = result.bound_certificate
        feasible = bool(certificate is not None and certificate.primal_feasible)
        objective = None if not feasible else certificate.best_feasible_primal_lower_bound
        reference_objective = float(canonical.c @ reference.x)
        selected_kkt = None
        if result.best_x is not None and result.best_y is not None:
            selected_kkt = float(normalized_kkt_residual(canonical, evaluate_primal_dual_pairs(canonical, result.best_x, result.best_y))[0])
        active_jaccard = None if result.best_x is None else active_set_jaccard(
            augmented_active_mask(canonical, result.best_x, config.active_tol), reference_mask,
        )
        records.append({
            "problem_id": problem_id, "family": family, "n": problem.n, "m": problem.m,
            "seed": seed, "repeat": repeat, "method": str(method),
            "time_budget_seconds": time_budget, "total_time_seconds": total_seconds,
            "status": result.status.value, "feasible": feasible,
            "canonical_objective": objective, "reference_canonical_objective": reference_objective,
            "relative_objective_error": relative_objective_error(objective, reference_objective),
            "selected_active_set_jaccard": active_jaccard, "selected_kkt_residual": selected_kkt,
            "certified_gap": result.best_certified_gap,
            "certified_relative_gap": result.best_certified_relative_gap,
            "iterations": result.iterations, **result.diagnostics,
            "recovery_max_steps": recovery_steps, "recovery_max_candidates": recovery_candidates,
            "recovery_time_fraction": recovery_fraction,
            "best_x": None if result.best_x is None else result.best_x.tolist(),
            "best_y": None if result.best_y is None else result.best_y.tolist(),
            "history": [{
                "elapsed_seconds": h.elapsed_seconds, "best_kkt_residual": h.best_kkt_residual,
                "primal_lower_bound": h.best_feasible_primal_objective,
                "dual_upper_bound": h.best_feasible_dual_objective,
                "certified_gap": h.best_certified_gap,
            } for h in history],
        })
    return records


def run_comparison(
    *, dimensions: Iterable[int] = (5, 10, 20), ratios: Iterable[int] = (2, 5),
    seeds: Iterable[int] = range(3), time_budget: float = 0.1, batch_size: int = 64,
    pdhg_steps: int = 5, repeats: int = 1,
    recovery: bool = False,
) -> list[dict]:
    dimensions, ratios, seeds = tuple(dimensions), tuple(ratios), tuple(seeds)
    if repeats <= 0 or not dimensions or not ratios or not seeds:
        raise ValueError("Nonempty dimensions, ratios, seeds and positive repeats are required")
    generators = {
        "bounded_dense": generate_bounded_dense_lp,
        "controlled_optimum": generate_controlled_optimum_lp,
        "narrow_region": generate_narrow_region_lp,
        "mixed_sign": generate_random_feasible_bounded_lp,
    }
    # Warm the shared Python / NumPy path outside the measured runs.
    warm = generate_bounded_dense_lp(2, 4, 0).problem
    warm_config = SolverConfig(batch_size=8, max_iter=1, vertex_polishing=VertexPolishingConfig(enabled=False))
    for method in METHODS:
        AdaptiveLPSolver(replace(warm_config, search_mode=method)).solve(warm)
    records = []
    for n in dimensions:
        for ratio in ratios:
            for seed in seeds:
                for family, generator in generators.items():
                    instance = generator(int(n), int(n * ratio), int(seed))
                    for repeat in range(repeats):
                        records.extend(compare_problem(
                            instance.problem, problem_id=f"{family}_n{n}_r{ratio}_s{seed}",
                            family=family, seed=int(seed), time_budget=time_budget,
                            batch_size=batch_size, pdhg_steps=pdhg_steps, repeat=repeat,
                            recovery=recovery,
                        ))
    return records


def summarize(records: list[dict]) -> list[dict]:
    summaries = []
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        errors = [r["relative_objective_error"] for r in rows if r["relative_objective_error"] is not None]
        residuals = [r["selected_kkt_residual"] for r in rows if r["selected_kkt_residual"] is not None]
        observed_residuals = [r["best_kkt_residual"] for r in rows if r["best_kkt_residual"] is not None]
        summaries.append({
            "method": method, "runs": len(rows),
            "feasible_rate": sum(r["feasible"] for r in rows) / len(rows),
            "objective_success_rate_1e-4": sum(r["feasible"] and r["relative_objective_error"] <= 1e-4 for r in rows) / len(rows),
            "certified_rate": sum(r["status"] == "OPTIMAL_CERTIFIED" for r in rows) / len(rows),
            "median_feasible_relative_error": None if not errors else float(np.median(errors)),
            "median_selected_kkt": None if not residuals else float(np.median(residuals)),
            "median_best_observed_kkt": None if not observed_residuals else float(np.median(observed_residuals)),
            "median_total_seconds": float(np.median([r["total_time_seconds"] for r in rows])),
        })
    return summaries


def comparison_environment() -> dict:
    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted(str(path.relative_to(source_root)) for path in source_root.rglob("*.py"))
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_files},
        "feasibility_tolerance": 1e-7, "absolute_gap_tolerance": 1e-6,
        "pdhg_step_scale": 0.9, "polishing_enabled": False,
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


def write_comparison(records: list[dict], output_dir: Path, *, config: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "summary.md"
    metadata = comparison_environment()
    json_path.write_text(json.dumps({"config": config, "environment": metadata, "summary": summary, "records": records}, indent=2, allow_nan=False))
    flat = [{k: v for k, v in row.items() if k not in {"history", "best_x", "best_y"}} for row in records]
    if flat:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = [
        "# Sampling with directed primal-dual updates", "",
        "All methods receive the same wall-clock limit per problem. Runs may stop early on a verified gap. "
        "Timing includes solver setup, metrics, archive management and final result construction. "
        "The limit is cooperative; an in-flight operation and final bookkeeping can overshoot it.", "",
        "Sampling uses the existing adaptive Gaussian and geometry rewards. PDHG follows one persistent zero-start pair. "
        "The hybrid refines sampled pairs with PDHG, trains its distribution on refined elites and preserves the same zero-start trajectory. "
        "PDHG uses safe diagonal row/column-sum steps. No vertex polishing, reference solution, or generated feasible point guides search.", "",
        "Active-set scores describe the selected point, include variable bounds, and use row-normalized boundary distances. "
        "In degenerate LPs, a different optimal active set can still solve the problem correctly.", "",
        f"This matrix contains {len({r['problem_id'] for r in records})} distinct LPs. Repeated timings of the same LP are not independent problem instances. "
        "Primal/dual feasibility tolerance is 1e-7 and the stopping gap tolerance is 1e-6.", "",
        "| Method | Runs | Feasible | Objective error ≤1e-4 | Certified | Median error, feasible runs | Best observed KKT | Selected KKT | Median seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if config.get("recovery"):
        lines[2:2] = ["Primal recovery is enabled: 10% of each time limit is reserved for bounded halfspace projections, shared by all three modes. Feasibility is independently checked at the original tolerance.", ""]
    for row in summary:
        error = "—" if row["median_feasible_relative_error"] is None else f"{row['median_feasible_relative_error']:.3g}"
        kkt = "—" if row["median_selected_kkt"] is None else f"{row['median_selected_kkt']:.3g}"
        observed = "—" if row["median_best_observed_kkt"] is None else f"{row['median_best_observed_kkt']:.3g}"
        lines.append(f"| {row['method']} | {row['runs']} | {row['feasible_rate']:.1%} | {row['objective_success_rate_1e-4']:.1%} | {row['certified_rate']:.1%} | {error} | {observed} | {kkt} | {row['median_total_seconds']:.4f} |")
    lines.extend(["", "KKT columns are medians across runs. Best observed KKT may come from an infeasible pair, while selected KKT describes the returned pair. "
                  "A method can approach optimality closely yet retain a poor feasible incumbent under the strict feasibility tolerance. "
                  "Lower feasible objective error therefore does not imply better primal-dual convergence.", ""])
    paired = defaultdict(dict)
    for r in records:
        paired[(r["problem_id"], r["repeat"])][r["method"]] = r
    def quality(row):
        error = row["relative_objective_error"]
        return float("inf") if error is None else error
    lines.extend(["", "Paired comparisons use final feasible objective error; missing feasible points lose. Differences ≤1e-6 count as ties.", ""])
    for baseline in ("sampling", "pdhg"):
        wins = ties = losses = 0
        for group in paired.values():
            if "hybrid" not in group or baseline not in group:
                continue
            a, b = quality(group["hybrid"]), quality(group[baseline])
            if a == b or abs(a - b) <= 1e-6:
                ties += 1
            elif a < b:
                wins += 1
            else:
                losses += 1
        lines.append(f"- Hybrid versus {baseline}: {wins} wins, {ties} ties, {losses} losses.")
    lines.extend(["", "![Comparison by problem dimension](comparison.png)", "", "This is a small synthetic CPU study, not evidence of general solver acceleration. "
                  "Problem dimensions, families, initial distributions and time budgets limit the conclusion. "
                  "Use repeated timings and larger / application LPs before selecting a default method.",
                  "", "## Configuration", "", "```json", json.dumps(config, indent=2), "```", ""])
    report_path.write_text("\n".join(lines))
    figure_path = output_dir / "comparison.png"
    _plot_comparison(records, figure_path)
    return {"results_json": json_path, "results_csv": csv_path, "summary": report_path, "figure": figure_path}


def _plot_comparison(records: list[dict], path: Path) -> None:
    # Import the repository's headless plotting setup only when exporting.
    from lpas.experiments.reporting import plt

    colors = {"sampling": "#a86432", "pdhg": "#386da8", "hybrid": "#178477"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    dimensions = sorted({r["n"] for r in records})
    panels = [
        ("feasible_rate", "Runs with a feasible incumbent", False),
        ("objective_success_rate_1e-4", "Feasible runs with objective error ≤ 1e-4", False),
        ("median_feasible_relative_error", "Median objective error (feasible runs)", True),
        ("median_best_observed_kkt", "Best observed KKT (pair may be infeasible)", True),
    ]
    for ax, (metric, title, log) in zip(axes.flat, panels, strict=True):
        for method in METHODS:
            values = []
            for n in dimensions:
                summary = summarize([r for r in records if r["n"] == n and r["method"] == method])[0]
                value = summary[metric]
                values.append(np.nan if value is None else max(value, 1e-12) if log else value)
            ax.plot(dimensions, values, marker="o", color=colors[method], label=method)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Variables")
        ax.set_xticks(dimensions)
        ax.grid(alpha=0.2)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Sampling and directed primal-dual search", x=0.06, ha="left", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
