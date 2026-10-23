"""Protocol-driven comparison with production PDLP and native polishing."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from lpas.experiments.pdlp_adapter import (
    SparseLP, check_pair, project_primal, read_mps, solve_highs, solve_pdlp,
)

SOLVE_METHODS = ("pdlp", "pdlp_native", "pdlp_projection")
RECOVERY_METHODS = ("projection", "pdlp_primal_feasibility", "pdlp_native_resume")


def pair_hash(x, y) -> str:
    digest = hashlib.sha256()
    for v in (x, y):
        digest.update(np.asarray(v, dtype="<f8").tobytes())
    return digest.hexdigest()


def _options(settings):
    return dict(residual_tolerance=settings["solver_residual_tolerance"],
                gap_tolerance=settings["solver_gap_tolerance"],
                check_frequency=settings["termination_check_frequency"])


def _check(lp, x, y, settings):
    return check_pair(lp, x, y, primal_tolerance=settings["primal_tolerance"],
                      dual_tolerance=settings["dual_tolerance"],
                      gap_tolerance=settings["gap_tolerances"][0])


def _accept_primal(lp, original, candidate, y, settings):
    old, new = _check(lp, original, y, settings), _check(lp, candidate, y, settings)
    def key(r):
        return (r["verified_success"], r["primal_feasible"],
                -r["primal_objective"] if r["primal_feasible"] else -r["primal_residual"])
    return candidate if new["finite"] and key(new) > key(old) else original


def _record(lp, result, *, start, budget, settings, **fields):
    x, y = result["x"], result["y"]
    metrics = _check(lp, x, y, settings)
    secondary = {str(tol): bool(metrics["primal_feasible"] and metrics["dual_feasible"]
                               and metrics["relative_gap"] is not None and metrics["relative_gap"] <= tol)
                 for tol in settings["gap_tolerances"]}
    elapsed = time.perf_counter() - start
    # Vector serialization, hashing, and writing artifacts are not solve work.
    return dict(problem_id=lp.name, problem_sha256=lp.fingerprint(), n=lp.n, m=lp.m,
                nonzeros=lp.A.nnz, budget_seconds=budget, total_seconds=elapsed,
                overrun_seconds=max(0.0, elapsed - budget),
                budget_success=metrics["verified_success"] and elapsed <= budget,
                verified_return_seconds=elapsed if metrics["verified_success"] else None,
                status=result["status"], iterations=result.get("iterations", 0),
                solver_seconds=result.get("solver_seconds", 0.0),
                parameters=result.get("parameters", {}), log=result.get("log", {}),
                **metrics, success_by_gap=secondary, x=np.asarray(x).tolist(), y=np.asarray(y).tolist(), **fields)


def run_recovery(lp, seed, method, budget, settings):
    """Every branch sees an immutable copy of the same saved primal/dual pair."""
    start = time.perf_counter()
    x, y = np.array(seed["x"], copy=True), np.array(seed["y"], copy=True)
    initial_sha = pair_hash(x, y)
    original_metrics = _check(lp, x, y, settings)
    remaining = budget - (time.perf_counter() - start)
    steps, accepted = 0, False
    result = dict(x=x, y=y, status="UNCHANGED")
    if remaining > 0.0 and method == "projection":
        candidate, steps = project_primal(lp, x, max_steps=settings["projection_max_steps"],
                                          tolerance=settings["projection_target"], deadline=start + budget)
        result["x"] = _accept_primal(lp, x, candidate, y, settings)
        accepted = not np.array_equal(result["x"], x)
        result["status"] = "PROJECTION_ACCEPTED" if accepted else "PROJECTION_UNCHANGED"
    elif remaining > 0.0 and method in {"pdlp_primal_feasibility", "pdlp_native_resume"}:
        primal_only = method == "pdlp_primal_feasibility"
        result = solve_pdlp(lp, time_limit=remaining, polishing=not primal_only,
                            initial=(x, np.zeros(lp.m) if primal_only else y),
                            feasibility_objective=primal_only, **_options(settings))
        if primal_only:
            result["x"] = _accept_primal(lp, x, result["x"], y, settings)
            result["y"] = y  # Feasibility-subproblem duals cannot certify the original objective.
        accepted = not np.array_equal(result["x"], x) or not np.array_equal(result["y"], y)
    elif method not in RECOVERY_METHODS:
        raise ValueError(f"Unknown recovery method: {method}")
    return _record(lp, result, start=start, budget=budget, settings=settings,
                   stage="recovery", method=method, initial_pair_sha256=initial_sha,
                   initial_metrics=original_metrics, projection_steps=steps, recovery_accepted=accepted)


def run_complete(loader, method, budget, settings):
    """Timed from MPS parsing through independent final verification."""
    if method not in SOLVE_METHODS:
        raise ValueError(f"Unknown solve method: {method}")
    start = time.perf_counter()
    lp = loader()
    ingestion_seconds = time.perf_counter() - start
    remaining = budget - ingestion_seconds
    fraction = settings["main_search_fraction"] if method == "pdlp_projection" else 1.0
    if remaining > 0.0:
        result = solve_pdlp(lp, time_limit=remaining * fraction, polishing=method == "pdlp_native", **_options(settings))
    else:
        result = dict(x=np.clip(np.zeros(lp.n), lp.variable_lower, lp.variable_upper),
                      y=np.zeros(lp.m), status="INGESTION_TIME_LIMIT")
    main_seconds = time.perf_counter() - start - ingestion_seconds
    recovery_start = time.perf_counter()
    steps, accepted = 0, False
    if method == "pdlp_projection" and time.perf_counter() < start + budget:
        metrics = _check(lp, result["x"], result["y"], settings)
        # Do not spend time repairing an already adequate primal point.
        if metrics["finite"] and not metrics["primal_feasible"] and result["status"] not in {
            "TERMINATION_REASON_PRIMAL_INFEASIBLE", "TERMINATION_REASON_DUAL_INFEASIBLE", "TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE",
        }:
            candidate, steps = project_primal(lp, result["x"], max_steps=settings["projection_max_steps"],
                                              tolerance=settings["projection_target"], deadline=start + budget)
            updated = _accept_primal(lp, result["x"], candidate, result["y"], settings)
            accepted = not np.array_equal(updated, result["x"])
            result["x"] = updated
    recovery_seconds = time.perf_counter() - recovery_start
    return _record(lp, result, start=start, budget=budget, settings=settings,
                   stage="complete", method=method, ingestion_seconds=ingestion_seconds,
                   main_seconds=main_seconds, recovery_seconds=recovery_seconds,
                   projection_steps=steps, recovery_accepted=accepted)


def smoke_problem() -> SparseLP:
    # Free x0, upper-only x1, lower-only x2, fixed x3; equality and ranged row.
    return SparseLP(sparse.csr_matrix([[1., 1., 0., 0.], [0., 1., 1., 1.]]),
                    np.array([1., -1., 2., 0.]), [1., 0.], [1., 3.],
                    [-np.inf, -np.inf, 0., 1.], [np.inf, 2., np.inf, 1.], -2., "smoke_bounds")


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol["version"] != 1:
        raise ValueError("Unsupported protocol version")
    settings = protocol["settings"]
    if settings["num_threads"] != 1 or settings["presolve"]:
        raise ValueError("This comparison requires one thread and shared presolve disabled")
    if not 0.0 < settings["main_search_fraction"] <= 1.0:
        raise ValueError("Invalid main-search fraction")
    if settings["repeats"] <= 0 or settings["projection_max_steps"] <= 0:
        raise ValueError("Positive repeats and projection steps required")
    for key in ("seed_time_limit_seconds", "projection_target", "primal_tolerance", "dual_tolerance",
                "solver_residual_tolerance", "solver_gap_tolerance"):
        if not np.isfinite(settings[key]) or settings[key] <= 0.0:
            raise ValueError(f"Invalid protocol setting: {key}")
    for key in ("recovery_budgets_seconds", "solve_budgets_seconds", "gap_tolerances"):
        if not settings[key] or any(not np.isfinite(v) or v <= 0.0 for v in settings[key]):
            raise ValueError(f"Invalid protocol grid: {key}")
    if not settings["seed_iterations"] or any(not isinstance(v, int) or v < 0 for v in settings["seed_iterations"]):
        raise ValueError("Invalid checkpoint iteration grid")
    if protocol["decision"]["bootstrap_resamples"] <= 0:
        raise ValueError("Positive bootstrap count required")
    return protocol


def run_benchmark(protocol, *, root: Path, output_dir: Path, stage="pilot", quick=False, progress=True):
    if stage not in {"pilot", "development", "heldout"}:
        raise ValueError("Unknown benchmark stage")
    settings = copy.deepcopy(protocol["settings"])
    if stage == "pilot":
        settings.update(protocol["pilot_overrides"])
    if quick:
        settings.update(repeats=1, seed_iterations=[2], seed_time_limit_seconds=0.1,
                        recovery_budgets_seconds=[0.02], solve_budgets_seconds=[0.05])
        loaders = [("smoke_bounds", smoke_problem)]
    else:
        split = "heldout" if stage == "heldout" else "development"
        loaders = []
        for item in protocol["instances"]:
            if item["split"] != split:
                continue
            path = root / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Input checksum mismatch: {path}")
            loaders.append((item["name"], lambda path=path: read_mps(path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    source_names = [Path(__file__), Path(__file__).with_name("pdlp_adapter.py")]
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       versions={p: importlib.metadata.version(p) for p in ("ortools", "highspy", "numpy", "scipy")},
                       thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                       sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_names})
    frozen = dict(stage=stage, quick=quick, settings=settings, protocol=protocol, environment=environment)
    # Persist the exact grid and input hashes before the first measured run.
    frozen_bytes = (json.dumps(frozen, indent=2, allow_nan=False) + "\n").encode()
    frozen_path = output_dir / "run_protocol.json"
    if frozen_path.exists() and frozen_path.read_bytes() != frozen_bytes:
        raise ValueError("Output directory contains a different protocol; choose a new directory")
    frozen_path.write_bytes(frozen_bytes)
    # Warm optional imports, native library setup, and shared paths outside timing.
    warm_lp = smoke_problem()
    solve_pdlp(warm_lp, time_limit=0.05)
    solve_pdlp(warm_lp, time_limit=0.05, polishing=True)
    solve_pdlp(warm_lp, time_limit=0.05, initial=(np.zeros(warm_lp.n), np.zeros(warm_lp.m)), feasibility_objective=True)
    solve_highs(warm_lp)
    rng = np.random.default_rng(settings["randomization_seed"])
    rng.shuffle(loaders)
    records, model_metadata = [], []
    journal = output_dir / "records.jsonl"
    with journal.open("w") as stream:
        def append(row):
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        for name, loader in loaders:
            lp = loader()
            lp.save(models_dir / f"{name}.npz")
            model_metadata.append(dict(name=name, model=f"models/{name}.npz", sha256=lp.fingerprint(), n=lp.n, m=lp.m, nonzeros=lp.A.nnz))
            reference_start = time.perf_counter()
            reference = solve_highs(lp)
            append(_record(lp, reference, start=reference_start, budget=30.0, settings=settings,
                           stage="reference", method="highs", repeat=0))
            for repeat in range(settings["repeats"]):
                for count in settings["seed_iterations"]:
                    seed_start = time.perf_counter()
                    seed = solve_pdlp(lp, time_limit=settings["seed_time_limit_seconds"],
                                      iteration_limit=count, **_options(settings))
                    seed_id = f"{name}:repeat{repeat}:iterations{count}"
                    seed_row = _record(lp, seed, start=seed_start, budget=settings["seed_time_limit_seconds"],
                                       settings=settings, stage="checkpoint", method="pdlp_seed", repeat=repeat,
                                       seed_id=seed_id, requested_seed_iterations=count, pair_sha256=pair_hash(seed["x"], seed["y"]))
                    append(seed_row)
                    for budget in settings["recovery_budgets_seconds"]:
                        for method in rng.permutation(RECOVERY_METHODS):
                            row = run_recovery(lp, seed, str(method), budget, settings)
                            row.update(repeat=repeat, seed_id=seed_id, requested_seed_iterations=count,
                                       seed_seconds=seed_row["total_seconds"])
                            append(row)
                for budget in settings["solve_budgets_seconds"]:
                    for method in rng.permutation(SOLVE_METHODS):
                        row = run_complete(loader, str(method), budget, settings)
                        row["repeat"] = repeat
                        append(row)
            if progress:
                print(f"Completed {name}: {lp.n} variables, {lp.m} rows, {lp.A.nnz} nonzeros", flush=True)
    payload = dict(**frozen, models=model_metadata, records=records,
                   run_protocol_sha256=hashlib.sha256(frozen_bytes).hexdigest())
    write_benchmark(payload, output_dir)
    return payload


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        if r["stage"] in {"complete", "recovery"}:
            groups[(r["stage"], r["budget_seconds"], r["method"])].append(r)
    rows = []
    for (stage, budget, method), group in sorted(groups.items()):
        rows.append(dict(stage=stage, budget_seconds=budget, method=method, runs=len(group),
                         problems=len({r["problem_id"] for r in group}),
                         feasible_rate=float(np.mean([r["primal_feasible"] for r in group])),
                         verified_success_rate=float(np.mean([r["verified_success"] for r in group])),
                         budget_success_rate=float(np.mean([r["budget_success"] for r in group])),
                         median_seconds=float(np.median([r["total_seconds"] for r in group])),
                         projection_acceptance_rate=float(np.mean([r.get("recovery_accepted", False) for r in group]))))
    return rows


def paired_comparisons(records, settings, resamples=2000):
    """Average repeats/checkpoints within an LP before resampling LPs."""
    output = []
    for stage, candidate, baseline in (("complete", "pdlp_projection", "pdlp_native"),
                                       ("recovery", "projection", "pdlp_primal_feasibility"),
                                       ("recovery", "projection", "pdlp_native_resume")):
        budgets = sorted({r["budget_seconds"] for r in records if r["stage"] == stage})
        for budget in budgets:
            groups = defaultdict(lambda: defaultdict(list))
            for row in records:
                if row["stage"] == stage and row["budget_seconds"] == budget:
                    groups[row["problem_id"]][row["method"]].append(row)
            differences, log_ratios = [], []
            for group in groups.values():
                if candidate not in group or baseline not in group:
                    continue
                arms = [group[candidate], group[baseline]]
                rates = [np.mean([r["budget_success"] for r in arm]) for arm in arms]
                # PAR2: a failure costs twice its budget, so early inaccurate
                # returns cannot create an apparent speed advantage.
                times = [np.mean([r["total_seconds"] if r["budget_success"] else 2 * budget for r in arm]) for arm in arms]
                differences.append(rates[0] - rates[1])
                log_ratios.append(np.log(max(times[0], 1e-12) / max(times[1], 1e-12)))
            if not differences:
                continue
            differences, log_ratios = np.asarray(differences), np.asarray(log_ratios)
            rng = np.random.default_rng(settings["randomization_seed"])
            indices = rng.integers(len(differences), size=(resamples, len(differences)))
            rate_ci = np.quantile(np.mean(differences[indices], axis=1), [0.025, 0.975])
            time_ci = np.exp(np.quantile(np.mean(log_ratios[indices], axis=1), [0.025, 0.975]))
            output.append(dict(stage=stage, budget_seconds=budget, candidate=candidate, baseline=baseline,
                               distinct_problems=len(differences),
                               success_difference_pp=100 * float(np.mean(differences)),
                               success_difference_ci_pp=(100 * rate_ci).tolist(),
                               par2_geometric_time_ratio=float(np.exp(np.mean(log_ratios))),
                               par2_time_ratio_ci=time_ci.tolist()))
    return output


def write_benchmark(payload, output_dir):
    payload["summary"] = summarize(payload["records"])
    payload["paired_comparisons"] = paired_comparisons(
        payload["records"], payload["settings"], payload["protocol"]["decision"]["bootstrap_resamples"])
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    excluded = {"x", "y", "log", "parameters", "initial_metrics", "success_by_gap"}
    flat = [{k: v for k, v in row.items() if k not in excluded} for row in payload["records"]]
    columns = sorted({k for row in flat for k in row})
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# PDLP recovery comparison", "",
             f"Stage: **{payload['stage']}**. Distinct problems: {len(payload['models'])}. "
             f"Timing repeats per problem: {payload['settings']['repeats']}.", "",
             "Pilot and development results are exploratory. Native PDLP polishing is the established comparator; "
             "projection is an alternative implementation of known feasibility methods, with no novelty claim.", "",
             f"Success requires independently checked primal residual ≤{payload['settings']['primal_tolerance']:g}, "
             f"dual residual ≤{payload['settings']['dual_tolerance']:g}, and relative gap ≤{payload['settings']['gap_tolerances'][0]:g}. "
             "These are numerical tolerances, not exact certificates. Budget success also requires measured return time within the limit.", "",
             "## Complete solves", "",
             "Timing includes MPS parsing, adapter conversion, native solve, optional recovery and independent final checks. "
             "Imports are warmed. Artifact serialization is excluded. PDLP projection reserves up to 10% for recovery; unused reserved time is not reclaimed. "
             "Presolve is disabled in all PDLP arms for compatibility with native polishing. HiGHS is separate reference context.", ""]
    def table(stage):
        lines.extend(["| Method | Budget (s) | Runs | Feasible | Verified success | Within-budget success | Median return (s) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in payload["summary"]:
            if row["stage"] == stage:
                lines.append(f"| {row['method']} | {row['budget_seconds']:g} | {row['runs']} | {row['feasible_rate']:.1%} | "
                             f"{row['verified_success_rate']:.1%} | {row['budget_success_rate']:.1%} | {row['median_seconds']:.6f} |")
    table("complete")
    lines.extend(["", "## Identical starting pairs", "",
                  "Each recovery comparison uses the same saved PDLP primal/dual pair, verified by its hash. "
                  "Projection and the explicit PDLP primal-feasibility subproblem preserve the original dual. "
                  "The latter uses zero objective and starts its own dual at zero, following PDLP's primal-polishing construction. "
                  "It is a controlled reconstruction, not the native internal polishing scheduler. "
                  "`pdlp_native_resume` resumes the original optimization problem with native polishing and can improve both vectors. "
                  "Checkpoint generation time is recorded separately and is common to these comparisons.", ""])
    table("recovery")
    lines.extend(["", "## Paired comparisons", "",
                  "Differences favor projection when positive; time ratios favor it below one. PAR2 assigns failures twice the budget. "
                  "Intervals resample distinct LPs after averaging repeats and checkpoints within each LP. "
                  "With very few LPs, these intervals are descriptive only.", "",
                  "| Stage | Budget (s) | Projection compared with | LPs | Within-budget success difference (pp), 95% interval | PAR2 time ratio, 95% interval |",
                  "| --- | ---: | --- | ---: | ---: | ---: |"])
    for row in payload["paired_comparisons"]:
        lo, hi = row["success_difference_ci_pp"]
        time_lo, time_hi = row["par2_time_ratio_ci"]
        lines.append(f"| {row['stage']} | {row['budget_seconds']:g} | {row['baseline']} | {row['distinct_problems']} | "
                     f"{row['success_difference_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}] | "
                     f"{row['par2_geometric_time_ratio']:.3f} [{time_lo:.3f}, {time_hi:.3f}] |")
    lines.extend(["", "## Interpretation", "",
                  "Compare success rates before conditional timing averages. A method returning early without meeting accuracy is not a speed win. "
                  "Returned-point timing is measured; first-hitting time is unavailable through the current Python binding. "
                  "A feasible primal point alone does not establish optimality. Repeated timings are not independent problems. "
                  "No statistical superiority is claimed from this small pilot.", "",
                  "See `run_protocol.json` for the settings frozen before timing, `results.json` for vectors and solver parameters, "
                  "and `validation.json` for independently recomputed checks.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines))


def validate_benchmark(path: Path, *, require_current_source=False) -> dict:
    """Rebuild checks from saved sparse models and raw vectors, including seed identity."""
    payload = json.loads(path.read_text())
    settings, directory = payload["settings"], path.parent
    failures = []
    models = {}
    for item in payload["models"]:
        lp = SparseLP.load(directory / item["model"])
        models[item["name"]] = lp
        if lp.fingerprint() != item["sha256"]:
            failures.append(f"Model hash mismatch: {item['name']}")
    frozen = (directory / "run_protocol.json").read_bytes()
    if hashlib.sha256(frozen).hexdigest() != payload["run_protocol_sha256"]:
        failures.append("Frozen protocol hash mismatch")
    frozen_payload = json.loads(frozen)
    for field in ("settings", "protocol", "environment", "stage", "quick"):
        if payload[field] != frozen_payload[field]:
            failures.append(f"Changed frozen field: {field}")
    seeds = {r["seed_id"]: r for r in payload["records"] if r["stage"] == "checkpoint"}
    expected = set()
    for item in payload["models"]:
        name = item["name"]
        expected.add((name, "reference", "highs", 0, None, 30.0))
        for repeat in range(settings["repeats"]):
            for count in settings["seed_iterations"]:
                expected.add((name, "checkpoint", "pdlp_seed", repeat, count, settings["seed_time_limit_seconds"]))
                for budget in settings["recovery_budgets_seconds"]:
                    for method in RECOVERY_METHODS:
                        expected.add((name, "recovery", method, repeat, count, budget))
            for budget in settings["solve_budgets_seconds"]:
                for method in SOLVE_METHODS:
                    expected.add((name, "complete", method, repeat, None, budget))
    observed = [(r["problem_id"], r["stage"], r["method"], r["repeat"],
                 r.get("requested_seed_iterations"), r["budget_seconds"]) for r in payload["records"]]
    if set(observed) != expected or len(observed) != len(expected):
        failures.append("Missing, duplicate or unexpected experiment cells")
    for i, row in enumerate(payload["records"]):
        if row["problem_id"] not in models:
            failures.append(f"Record {i}: unknown model")
            continue
        lp = models[row["problem_id"]]
        if row["problem_sha256"] != lp.fingerprint():
            failures.append(f"Record {i}: problem hash mismatch")
        actual = _check(lp, row["x"], row["y"], settings)
        for tolerance, saved in row["success_by_gap"].items():
            expected_success = (actual["primal_feasible"] and actual["dual_feasible"]
                                and actual["relative_gap"] is not None and actual["relative_gap"] <= float(tolerance))
            if saved != expected_success:
                failures.append(f"Record {i}: secondary success mismatch")
        if not np.isfinite(row["total_seconds"]) or row["total_seconds"] < 0:
            failures.append(f"Record {i}: invalid measured time")
        for key, value in actual.items():
            saved = row[key]
            if isinstance(value, (bool, type(None))):
                valid = saved == value
            else:
                valid = saved is not None and np.isclose(saved, value, rtol=1e-11, atol=1e-13)
            if not valid:
                failures.append(f"Record {i}: invalid {key}")
        expected_budget_success = actual["verified_success"] and row["total_seconds"] <= row["budget_seconds"]
        if row["budget_success"] != expected_budget_success:
            failures.append(f"Record {i}: invalid budget success")
        if row["stage"] == "checkpoint" and row["pair_sha256"] != pair_hash(row["x"], row["y"]):
            failures.append(f"Record {i}: checkpoint pair hash mismatch")
        if row["stage"] == "recovery":
            if row["seed_id"] not in seeds:
                failures.append(f"Record {i}: missing checkpoint")
                continue
            seed = seeds[row["seed_id"]]
            if row["initial_pair_sha256"] != pair_hash(seed["x"], seed["y"]):
                failures.append(f"Record {i}: different starting pair")
            if row["initial_metrics"] != _check(lp, seed["x"], seed["y"], settings):
                failures.append(f"Record {i}: initial metrics mismatch")
            if row["method"] in {"projection", "pdlp_primal_feasibility"} and row["y"] != seed["y"]:
                failures.append(f"Record {i}: primal recovery changed original dual")
    if payload["summary"] != summarize(payload["records"]):
        failures.append("Summary differs from raw records")
    if payload["paired_comparisons"] != paired_comparisons(
        payload["records"], settings, payload["protocol"]["decision"]["bootstrap_resamples"]):
        failures.append("Paired comparisons differ from raw records")
    root = Path(__file__).resolve().parents[3]
    sources_match = all((root / name).exists() and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                        for name, digest in payload["environment"]["sources"].items())
    if require_current_source and not sources_match:
        failures.append("Execution sources differ from current source")
    result = dict(records_checked=len(payload["records"]), models_checked=len(models),
                  source_matches=sources_match, passed=not failures, failures=failures)
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
