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

from lpas.backends import create_backend
from lpas.gpu.memory import estimate_batch_memory_bytes
from lpas.problem_generation.random_feasible_lps import generate_random_feasible_bounded_lp
from lpas.utils.config import BackendConfig, ParallelSolverConfig, SamplerConfig


BENCHMARK_K_VALUES = (100_000, 500_000, 1_000_000)
DEFAULT_BENCHMARK_CHUNK_SIZE = 100_000
BENCHMARK_DIMENSIONS = {
    "small": {"n": 10, "m": 20, "K": BENCHMARK_K_VALUES},
    "medium": {"n": 50, "m": 100, "K": BENCHMARK_K_VALUES},
    "large": {"n": 100, "m": 200, "K": BENCHMARK_K_VALUES},
    "wide": {"n": 100, "m": 500, "K": BENCHMARK_K_VALUES},
}


@dataclass(frozen=True)
class GPUThroughputRecord:
    backend: str
    backend_requested: str
    backend_used: str | None
    backend_available: bool
    backend_unavailability_reason: str | None
    device: str | None
    dtype: str
    n: int
    m: int
    K: int
    chunk_size: int
    num_chunks: int
    elapsed_seconds: float | None
    samples_per_second: float | None
    matmul_time_seconds: float | None
    scoring_time_seconds: float | None
    elite_selection_time_seconds: float | None
    distribution_update_time_seconds: float | None
    memory_estimate_mb: float
    peak_memory_mb: float | None
    best_score: float | None
    best_primal_objective: float | None
    best_dual_objective: float | None
    best_raw_gap: float | None
    best_primal_violation: float | None
    best_dual_violation: float | None
    best_complementarity: float | None
    unavailability_reason: str | None = None


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _chunk_sequence(total: int, chunk_size: int) -> list[int]:
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def _peak_memory_mb(backend_name: str) -> float | None:
    if not backend_name.startswith("torch_"):
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if backend_name == "torch_cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0**2))
    if backend_name == "torch_mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory() / (1024.0**2))
    return None


def _resolve_chunk_size(K: int, chunk_size: int | None) -> int:
    resolved_chunk_size = int(chunk_size) if chunk_size is not None else min(int(K), DEFAULT_BENCHMARK_CHUNK_SIZE)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return min(int(K), resolved_chunk_size)


def _memory_estimate_mb(*, K: int, chunk_size: int, n: int, m: int, dtype: str) -> float:
    return float(
        estimate_batch_memory_bytes(
            min(int(K), int(chunk_size)),
            int(n),
            int(m),
            dtype_bytes=4 if dtype == "float32" else 8,
        )
        / (1024.0**2)
    )


def _unavailable_record(
    *,
    backend: str,
    n: int,
    m: int,
    K: int,
    chunk_size: int,
    num_chunks: int,
    dtype: str,
    memory_estimate_mb: float,
    reason: str,
) -> GPUThroughputRecord:
    return GPUThroughputRecord(
        backend=backend,
        backend_requested=backend,
        backend_used=None,
        backend_available=False,
        backend_unavailability_reason=reason,
        device=None,
        dtype=dtype,
        n=int(n),
        m=int(m),
        K=int(K),
        chunk_size=int(chunk_size),
        num_chunks=int(num_chunks),
        elapsed_seconds=math.nan,
        samples_per_second=math.nan,
        matmul_time_seconds=math.nan,
        scoring_time_seconds=math.nan,
        elite_selection_time_seconds=math.nan,
        distribution_update_time_seconds=math.nan,
        memory_estimate_mb=memory_estimate_mb,
        peak_memory_mb=None,
        best_score=None,
        best_primal_objective=None,
        best_dual_objective=None,
        best_raw_gap=None,
        best_primal_violation=None,
        best_dual_violation=None,
        best_complementarity=None,
        unavailability_reason=reason,
    )


def benchmark_candidate_throughput(
    *,
    backend: str,
    n: int,
    m: int,
    K: int,
    chunk_size: int | None = None,
    seed: int = 0,
    dtype: str = "float32",
) -> GPUThroughputRecord:
    resolved_chunk_size = _resolve_chunk_size(int(K), chunk_size)
    num_chunks = len(_chunk_sequence(int(K), resolved_chunk_size))
    memory_estimate_mb = _memory_estimate_mb(K=int(K), chunk_size=resolved_chunk_size, n=int(n), m=int(m), dtype=dtype)

    try:
        backend_impl = create_backend(backend, dtype=dtype)
    except RuntimeError as exc:
        return _unavailable_record(
            backend=backend,
            n=int(n),
            m=int(m),
            K=int(K),
            chunk_size=resolved_chunk_size,
            num_chunks=num_chunks,
            dtype=dtype,
            memory_estimate_mb=memory_estimate_mb,
            reason=str(exc),
        )
    problem = generate_random_feasible_bounded_lp(int(n), int(m), seed).problem
    config = ParallelSolverConfig(
        samples_per_iteration=int(K),
        chunk_size=resolved_chunk_size,
        iterations=1,
        backend=BackendConfig(backend=backend, dtype=dtype),
        sampler=SamplerConfig(
            seed=seed,
            alpha=0.5,
            sigma_init=1.5,
            sigma_min=1e-4,
            sigma_max=4.0,
            primal_init_mean=0.5,
            dual_init_mean=0.5,
        ),
    )
    problem_data = backend_impl.prepare_problem(problem)
    state = backend_impl.initialize_distribution(problem, config)

    if backend_impl.resolved_backend == "torch_cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()

    matmul_seconds = 0.0
    scoring_seconds = 0.0
    elite_selection_seconds = 0.0
    distribution_update_seconds = 0.0
    best_payload: dict[str, float] | None = None

    backend_impl.sync()
    overall_start = time.perf_counter()
    for current_chunk in _chunk_sequence(int(K), resolved_chunk_size):
        X_chunk, Y_chunk = backend_impl.sample_candidates(state, current_chunk)

        backend_impl.sync()
        start = time.perf_counter()
        metrics = backend_impl.evaluate_batch(
            problem_data,
            X_chunk,
            Y_chunk,
            active_epsilon=config.backend.active_epsilon,
        )
        backend_impl.sync()
        matmul_seconds += time.perf_counter() - start

        backend_impl.sync()
        start = time.perf_counter()
        scores = backend_impl.score_batch(
            problem_data,
            X_chunk,
            Y_chunk,
            metrics,
            config.scoring,
            active_frequency=None,
            active_epsilon=config.backend.active_epsilon,
            dual_positive_epsilon=config.backend.dual_positive_epsilon,
        )
        backend_impl.sync()
        scoring_seconds += time.perf_counter() - start

        keep_count = max(1, min(current_chunk, math.ceil(0.05 * current_chunk)))
        backend_impl.sync()
        start = time.perf_counter()
        elite_X, elite_Y, elite_scores, elite_indices = backend_impl.select_elites(X_chunk, Y_chunk, scores, keep_count)
        backend_impl.sync()
        elite_selection_seconds += time.perf_counter() - start

        best_index = int(backend_impl.to_numpy(elite_indices)[0])
        candidate = {
            "score": float(backend_impl.to_numpy(elite_scores)[0]),
            "primal_objective": float(backend_impl.to_numpy(metrics["primal_objective"][best_index])),
            "dual_objective": float(backend_impl.to_numpy(metrics["dual_objective"][best_index])),
            "raw_gap": float(backend_impl.to_numpy(metrics["gap"][best_index])),
            "primal_violation": float(backend_impl.to_numpy(metrics["primal_violation"][best_index])),
            "dual_violation": float(backend_impl.to_numpy(metrics["dual_violation"][best_index])),
            "complementarity": float(backend_impl.to_numpy(metrics["complementarity"][best_index])),
        }
        if best_payload is None or candidate["score"] > best_payload["score"]:
            best_payload = candidate

        backend_impl.sync()
        start = time.perf_counter()
        state = backend_impl.update_distribution(state, elite_X, elite_Y, config)
        backend_impl.sync()
        distribution_update_seconds += time.perf_counter() - start

    backend_impl.sync()
    elapsed_seconds = time.perf_counter() - overall_start
    if best_payload is None:
        raise RuntimeError("No benchmark candidates were evaluated.")

    return GPUThroughputRecord(
        backend=backend,
        backend_requested=backend,
        backend_used=backend_impl.resolved_backend,
        backend_available=True,
        backend_unavailability_reason=None,
        device=backend_impl.device_name,
        dtype=dtype,
        n=int(n),
        m=int(m),
        K=int(K),
        chunk_size=resolved_chunk_size,
        num_chunks=num_chunks,
        elapsed_seconds=float(elapsed_seconds),
        samples_per_second=float(K / elapsed_seconds) if elapsed_seconds > 0.0 else math.inf,
        matmul_time_seconds=float(matmul_seconds),
        scoring_time_seconds=float(scoring_seconds),
        elite_selection_time_seconds=float(elite_selection_seconds),
        distribution_update_time_seconds=float(distribution_update_seconds),
        memory_estimate_mb=memory_estimate_mb,
        peak_memory_mb=_peak_memory_mb(backend_impl.resolved_backend),
        best_score=float(best_payload["score"]),
        best_primal_objective=float(best_payload["primal_objective"]),
        best_dual_objective=float(best_payload["dual_objective"]),
        best_raw_gap=float(best_payload["raw_gap"]),
        best_primal_violation=float(best_payload["primal_violation"]),
        best_dual_violation=float(best_payload["dual_violation"]),
        best_complementarity=float(best_payload["complementarity"]),
        unavailability_reason=None,
    )


def run_gpu_throughput_suite(
    *,
    backends: Iterable[str],
    size_names: Iterable[str] = BENCHMARK_DIMENSIONS.keys(),
    seed: int = 0,
    dtype: str = "float32",
    chunk_size: int | None = None,
) -> list[GPUThroughputRecord]:
    results: list[GPUThroughputRecord] = []
    for size_name in size_names:
        size = BENCHMARK_DIMENSIONS[str(size_name)]
        for K in size["K"]:
            for backend in backends:
                results.append(
                    benchmark_candidate_throughput(
                        backend=str(backend),
                        n=int(size["n"]),
                        m=int(size["m"]),
                        K=int(K),
                        chunk_size=chunk_size,
                        seed=seed,
                        dtype=dtype,
                    )
                )
    return results


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


def _plot_series(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    _ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("backend_available") is False:
            continue
        value = row.get(y_key)
        if value is None or not math.isfinite(float(value)):
            continue
        backend_requested = str(row.get("backend_requested", row["backend"]))
        backend_used = row.get("backend_used")
        backend_label = backend_requested
        if backend_used and backend_used != backend_requested:
            backend_label = f"{backend_requested} ({backend_used})"
        grouped.setdefault(backend_label, []).append((float(row[x_key]), float(value)))
    if not grouped:
        ax.text(0.5, 0.5, "No benchmark data available", ha="center", va="center")
    else:
        for backend_name, points in sorted(grouped.items()):
            points.sort(key=lambda item: item[0])
            ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=backend_name)
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_gpu_throughput_plots(records: list[GPUThroughputRecord], output_dir: Path) -> list[Path]:
    rows = [_jsonify(record) for record in records]
    outputs = [
        output_dir / "samples_per_second_by_backend.png",
        output_dir / "samples_per_second_vs_K.png",
        output_dir / "evaluation_time_vs_dimension.png",
        output_dir / "memory_estimate_vs_K.png",
    ]
    _plot_series(rows, x_key="n", y_key="samples_per_second", output_path=outputs[0], title="Samples per second by backend", xlabel="n", ylabel="samples/s")
    _plot_series(rows, x_key="K", y_key="samples_per_second", output_path=outputs[1], title="Samples per second vs K", xlabel="K", ylabel="samples/s")
    _plot_series(rows, x_key="n", y_key="elapsed_seconds", output_path=outputs[2], title="Evaluation time vs dimension", xlabel="n", ylabel="seconds")
    _plot_series(rows, x_key="K", y_key="memory_estimate_mb", output_path=outputs[3], title="Memory estimate vs K", xlabel="K", ylabel="MB")
    return outputs


def write_gpu_throughput_outputs(
    records: list[GPUThroughputRecord],
    output_root: Path,
    *,
    config_payload: dict[str, Any],
) -> dict[str, Path]:
    results_dir = output_root / "results" / "gpu_benchmarks"
    figures_dir = output_root / "figures" / "gpu_benchmarks"
    rows = [_jsonify(record) for record in records]
    results_csv = results_dir / "gpu_throughput_results.csv"
    results_json = results_dir / "gpu_throughput_results.json"
    _write_csv(results_csv, rows)
    _write_json(
        results_json,
        {
            "config": config_payload,
            "interpretation": (
                "This benchmark measures candidate evaluation throughput, not optimization success. "
                "GPU backends are expected to become useful only for sufficiently large K, n, and m, "
                "and unavailable backends are recorded explicitly."
            ),
            "records": rows,
        },
    )
    make_gpu_throughput_plots(records, figures_dir)
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "figure_dir": figures_dir,
    }
