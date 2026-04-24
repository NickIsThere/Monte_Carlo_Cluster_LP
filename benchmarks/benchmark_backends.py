from __future__ import annotations

import argparse
import time
from pprint import pprint

from lpas.experiments.generators import random_feasible_lp
from lpas.gpu.memory import estimate_batch_memory_bytes
from lpas.solver.parallel_solver import ParallelLPSolver
from lpas.utils.config import BackendConfig, ParallelSolverConfig, SamplerConfig

PROBLEM_SIZES = {
    "small": {"n": 10, "m": 20, "samples": 10_000, "chunk_size": 10_000},
    "medium": {"n": 100, "m": 200, "samples": 100_000, "chunk_size": 25_000},
    "large": {"n": 500, "m": 1000, "samples": 100_000, "chunk_size": 10_000},
}
DEFAULT_BACKENDS = ["numpy_cpu", "torch_cpu", "torch_mps", "torch_cuda", "numba_cpu"]


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 8


def benchmark_backend(backend: str, size_name: str, *, iterations: int, seed: int, dtype: str) -> dict[str, object]:
    size = PROBLEM_SIZES[size_name]
    problem = random_feasible_lp(n=size["n"], m=size["m"], seed=seed)
    config = ParallelSolverConfig(
        samples_per_iteration=size["samples"],
        chunk_size=size["chunk_size"],
        elite_fraction=0.05,
        iterations=iterations,
        backend=BackendConfig(backend=backend, dtype=dtype),
        sampler=SamplerConfig(
            seed=seed,
            alpha=0.4,
            sigma_init=1.5,
            sigma_min=1e-3,
            sigma_max=4.0,
            primal_init_mean=0.5,
            dual_init_mean=0.5,
        ),
    )
    start = time.perf_counter()
    result = ParallelLPSolver(config).solve(problem)
    elapsed = time.perf_counter() - start
    return {
        "backend_name": backend,
        "device_name": result.device,
        "size": size_name,
        "n": size["n"],
        "m": size["m"],
        "samples_per_iteration": size["samples"],
        "iterations": iterations,
        "elapsed_time_seconds": elapsed,
        "samples_per_second": size["samples"] * iterations / elapsed if elapsed > 0.0 else float("inf"),
        "memory_estimate_bytes": estimate_batch_memory_bytes(
            min(size["samples"], size["chunk_size"]),
            size["n"],
            size["m"],
            dtype_bytes=_dtype_bytes(dtype),
        ),
        "best_score_improvement": result.history[-1].best_score - result.history[0].best_score,
        "best_score": result.best_score,
        "best_primal_objective": result.best_primal_objective,
        "best_gap": result.best_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark explicit NumPy, Torch, and Numba LP sampling backends.")
    parser.add_argument("--sizes", nargs="+", choices=sorted(PROBLEM_SIZES), default=["small", "medium", "large"])
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for size_name in args.sizes:
        for backend in args.backends:
            try:
                outcome = benchmark_backend(backend, size_name, iterations=args.iterations, seed=args.seed, dtype=args.dtype)
            except RuntimeError as exc:
                outcome = {"backend_name": backend, "size": size_name, "skipped": str(exc)}
            results.append(outcome)
            pprint(outcome)

    print("\nCompleted benchmark runs:")
    pprint(results)


if __name__ == "__main__":
    main()
