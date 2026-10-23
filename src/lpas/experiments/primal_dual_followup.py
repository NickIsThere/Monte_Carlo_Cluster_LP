"""Predeclared recovery ablation with separate screening and held-out seeds."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lpas.experiments.primal_dual_comparison import METHODS, compare_problem, comparison_environment
from lpas.experiments.random_dense import generate_bounded_dense_lp, generate_controlled_optimum_lp, generate_narrow_region_lp
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver
from lpas.utils.config import PrimalRecoveryConfig, SolverConfig, VertexPolishingConfig


PROFILES = {
    "raw": {"recovery": False, "batch_size": 64, "pdhg_steps": 5},
    "recovered": {"recovery": True, "batch_size": 64, "pdhg_steps": 5},
    "recovered_small": {"recovery": True, "batch_size": 16, "pdhg_steps": 5},
    "recovered_deep": {"recovery": True, "batch_size": 16, "pdhg_steps": 20},
}
GENERATORS = {
    "bounded_dense": generate_bounded_dense_lp,
    "controlled_optimum": generate_controlled_optimum_lp,
    "narrow_region": generate_narrow_region_lp,
    "mixed_sign": generate_random_feasible_bounded_lp,
}


def run_followup(
    *, dimensions=(10, 20, 40), ratios=(2, 5), seeds=(3, 4, 5),
    budgets=(0.1, 0.3), repeats=2, profiles=("raw", "recovered"),
    progress: bool = False,
) -> list[dict]:
    if repeats <= 0 or not all((dimensions, ratios, seeds, budgets, profiles)):
        raise ValueError("Follow-up requires nonempty settings and positive repeats")
    unknown = set(profiles).difference(PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    warm = generate_bounded_dense_lp(2, 4, 0).problem
    for method in METHODS:
        for recovery in (False, True):
            AdaptiveLPSolver(SolverConfig(search_mode=method, batch_size=8, max_iter=2,
                recovery=PrimalRecoveryConfig(enabled=recovery), vertex_polishing=VertexPolishingConfig(enabled=False))).solve(warm)
    records = []
    total_problems = len(dimensions) * len(ratios) * len(seeds) * len(GENERATORS)
    completed = 0
    for n in dimensions:
        for ratio in ratios:
            for split_seed in seeds:
                for family_index, (family, generator) in enumerate(GENERATORS.items()):
                    # Independent problem seeds per design cell; sampler seeds
                    # and problem seeds are recorded, never selected by outcomes.
                    seed = int(np.random.SeedSequence([split_seed, n, ratio, family_index, 261022]).generate_state(1)[0])
                    instance = generator(n, n * ratio, seed)
                    problem_id = f"{family}_n{n}_r{ratio}_s{split_seed}"
                    for budget in budgets:
                        for repeat in range(repeats):
                            order = np.random.default_rng(seed + repeat).permutation(profiles)
                            for profile in order:
                                rows = compare_problem(instance.problem, problem_id=problem_id,
                                    family=family, seed=seed, time_budget=budget, repeat=repeat,
                                    history_limit=64, **PROFILES[str(profile)])
                                for row in rows:
                                    row.update(profile=str(profile), split_seed=split_seed)
                                records.extend(rows)
                    completed += 1
                    if progress and (completed == 1 or completed % max(1, total_problems // 12) == 0 or completed == total_problems):
                        print(f"Completed {completed}/{total_problems} LPs ({len(records)} timed runs)", flush=True)
    return records


def followup_summary(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["time_budget_seconds"], row["profile"], row["method"])].append(row)
    summaries = []
    for (budget, profile, method), rows in sorted(grouped.items()):
        errors = [r["relative_objective_error"] for r in rows if r["feasible"]]
        summary = {"budget_seconds": budget, "profile": profile, "method": method,
            "runs": len(rows), "feasible_rate": float(np.mean([r["feasible"] for r in rows])),
            "median_feasible_error": None if not errors else float(np.median(errors)),
            "median_best_kkt": float(np.median([r["best_kkt_residual"] for r in rows])),
            "median_seconds": float(np.median([r["total_time_seconds"] for r in rows])),
            "median_recovery_seconds": float(np.median([r["recovery_time_seconds"] for r in rows])),
            "recovery_improved_rate": float(np.mean([r["recovery_improved_incumbent"] for r in rows])),
            "certified_rate": float(np.mean([r["status"] == "OPTIMAL_CERTIFIED" for r in rows])),
        }
        for label, threshold in (("1pct", 1e-2), ("01pct", 1e-3), ("001pct", 1e-4)):
            summary[f"success_{label}"] = float(np.mean([r["feasible"] and r["relative_objective_error"] <= threshold for r in rows]))
        summaries.append(summary)
    return summaries


def paired_success_interval(records: list[dict], *, budget: float, left: tuple[str, str], right: tuple[str, str]) -> dict:
    """Bootstrap distinct LPs, averaging timing repeats within each LP first."""
    values = defaultdict(lambda: defaultdict(list))
    for row in records:
        key = (row["profile"], row["method"])
        if row["time_budget_seconds"] != budget or key not in {left, right}:
            continue
        success = row["feasible"] and row["relative_objective_error"] <= 1e-2
        values[row["problem_id"]][key].append(float(success))
    deltas = np.array([np.mean(v[left]) - np.mean(v[right]) for v in values.values() if left in v and right in v])
    if deltas.size == 0:
        return {"problems": 0, "difference": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(261022)
    boot = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
    return {"problems": len(deltas), "difference": float(deltas.mean()),
            "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def write_followup(records: list[dict], output_dir: Path, *, config: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = followup_summary(records)
    paths = {"results": output_dir / "results.json", "csv": output_dir / "results.csv",
             "report": output_dir / "summary.md", "figure": output_dir / "comparison.png"}
    contrasts = []
    budgets = sorted({r["time_budget_seconds"] for r in records})
    for budget in budgets:
        for method in METHODS:
            contrasts.append({"budget": budget, "comparison": f"recovery − raw, {method}",
                **paired_success_interval(records, budget=budget, left=("recovered", method), right=("raw", method))})
        for other in ("sampling", "pdhg"):
            contrasts.append({"budget": budget, "comparison": f"recovered hybrid − recovered {other}",
                **paired_success_interval(records, budget=budget, left=("recovered", "hybrid"), right=("recovered", other))})
        contrasts.append({"budget": budget, "comparison": "deep recovered hybrid − deep recovered pdhg",
            **paired_success_interval(records, budget=budget, left=("recovered_deep", "hybrid"), right=("recovered_deep", "pdhg"))})
    paths["results"].write_text(json.dumps({"config": config, "environment": comparison_environment(),
        "summary": summaries, "paired_success_contrasts": contrasts, "records": records}, indent=2, allow_nan=False))
    flat = [{k: v for k, v in row.items() if k not in {"history", "best_x", "best_y"}} for row in records]
    if flat:
        with paths["csv"].open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    lines = ["# Follow-up: feasibility recovery and longer budgets", "",
        f"{len({r['problem_id'] for r in records})} distinct synthetic LPs; {len(records)} timed runs. "
        "The raw and recovered profiles use the same batch size and directed steps. "
        "Recovery reserves 10% of each time limit for at most 256 projection rounds on up to four candidate points (at most 1024 halfspace projections). "
        "It runs after search, never trains the sampler, and uses no reference point or planted feasible point. "
        "All methods use the same recovery policy; all recovery work counts toward the time limit.", "",
        "Feasibility tolerance remains 1e-7. The projection target is 1e-8; tolerances were not relaxed. "
        "Recovery is bounded and can fail. Timings are cooperative and include setup and bookkeeping. "
        "Histories retain up to 64 evenly spaced checkpoints, including the first and last; final metrics include recovery.", "",
        "Screening profiles additionally vary batch size / PDHG steps. The primary validation compares raw and recovered profiles "
        "on new independent seeds, so that comparison changes one factor. A separate recovered_deep arm checks the setting chosen from screening (batch 16, 20 steps). "
        "The 1%, 0.1% and 0.01% accuracy thresholds and study grid were specified before the timed runs.", "",
        "| Budget | Profile | Method | Runs | Feasible | Error ≤1% | ≤0.1% | ≤0.01% | Median error, feasible | Median KKT | Seconds |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        error = "—" if row["median_feasible_error"] is None else f"{row['median_feasible_error']:.3g}"
        lines.append(f"| {row['budget_seconds']:.2f} | {row['profile']} | {row['method']} | {row['runs']} | {row['feasible_rate']:.1%} | {row['success_1pct']:.1%} | {row['success_01pct']:.1%} | {row['success_001pct']:.1%} | {error} | {row['median_best_kkt']:.3g} | {row['median_seconds']:.4f} |")
    lines.extend(["", "Median errors are conditional on feasibility; the success-rate columns count every run. KKT can come from an infeasible pair.",
        "", "## Paired success-rate differences", "",
        "Differences below are percentage points for feasible solutions with objective error ≤1%. "
        "Timing repeats are averaged within each LP; approximate 95% bootstrap intervals resample distinct LPs. "
        "These intervals describe this synthetic design, not arbitrary LP populations.", "",
        "| Budget | Comparison | LPs | Difference | 95% interval |", "| ---: | --- | ---: | ---: | --- |"])
    for row in contrasts:
        if row["problems"]:
            lines.append(f"| {row['budget']:.2f} | {row['comparison']} | {row['problems']} | {100*row['difference']:+.1f} | [{100*row['ci_low']:+.1f}, {100*row['ci_high']:+.1f}] |")
    lines.extend(["", "![Recovery comparison](comparison.png)", "", "## By dimension", "",
        "| Variables | Budget | Profile | Method | Feasible | Error ≤1% |", "| ---: | ---: | --- | --- | ---: | ---: |"])
    for n in sorted({r["n"] for r in records}):
        for row in followup_summary([r for r in records if r["n"] == n]):
            lines.append(f"| {n} | {row['budget_seconds']:.2f} | {row['profile']} | {row['method']} | {row['feasible_rate']:.1%} | {row['success_1pct']:.1%} |")
    lines.extend(["", "## Limitations", "", "Basic PDHG has diagonal preconditioning but no adaptive restart, line search or full PDLP implementation. "
        "The synthetic families and subsecond CPU limits constrain the conclusion. No comparison here establishes acceleration over a production LP solver. "
        "Recovery is optional and does not establish infeasibility when unsuccessful.", "",
        "Projection background: [De Loera, Haddock and Needell, 2016](https://arxiv.org/abs/1605.01418). "
        "This implementation uses the full most-violated-row rule, not their randomized sampling algorithm.", "",
        "## Configuration", "", "```json", json.dumps(config, indent=2), "```", ""])
    paths["report"].write_text("\n".join(lines))
    _plot_followup(summaries, paths["figure"])
    return paths


def _plot_followup(summary: list[dict], path: Path) -> None:
    from lpas.experiments.reporting import plt
    from matplotlib.ticker import PercentFormatter

    budgets = sorted({r["budget_seconds"] for r in summary})
    profiles = [p for p in PROFILES if any(r["profile"] == p for r in summary)]
    colors = {"raw": "#aa703f", "recovered": "#178477", "recovered_small": "#386da8", "recovered_deep": "#73588b"}
    fig, axes = plt.subplots(len(budgets), 3, figsize=(12, 3.7 * len(budgets)), squeeze=False)
    for row_index, budget in enumerate(budgets):
        for column, (metric, title, log) in enumerate([
            ("feasible_rate", "Feasible incumbent", False), ("success_1pct", "Feasible with objective error ≤1%", False),
            ("median_feasible_error", "Median error among feasible runs", True),
        ]):
            ax = axes[row_index, column]; width = 0.8 / len(profiles)
            for i, profile in enumerate(profiles):
                values = []
                for method in METHODS:
                    cell = next(r for r in summary if r["budget_seconds"] == budget and r["profile"] == profile and r["method"] == method)
                    value = cell[metric]; values.append(np.nan if value is None else max(value, 1e-10) if log else value)
                ax.bar(np.arange(3) + (i - (len(profiles)-1)/2)*width, values, width=width,
                       label=profile, color=colors[profile])
            ax.set_xticks(range(3), METHODS); ax.set_title(f"{budget:.2f}s · {title}", loc="left", fontsize=10)
            ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
            if log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0,0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recovering feasible points within the search budget", x=.06, ha="left", fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(path, dpi=160); plt.close(fig)
