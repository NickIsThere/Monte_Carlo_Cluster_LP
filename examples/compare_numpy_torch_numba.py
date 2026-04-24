from __future__ import annotations

import argparse
from pprint import pprint

from lpas.experiments.generators import random_feasible_lp
from lpas.solver.parallel_solver import ParallelLPSolver
from lpas.utils.config import BackendConfig, ParallelSolverConfig, SamplerConfig

DEFAULT_BACKENDS = ["numpy_cpu", "torch_cpu", "torch_mps", "torch_cuda", "numba_cpu"]


def run_backend(backend: str, *, samples: int, chunk_size: int, iterations: int, seed: int, dtype: str) -> dict[str, object]:
    config = ParallelSolverConfig(
        samples_per_iteration=samples,
        chunk_size=chunk_size,
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
    result = ParallelLPSolver(config).solve(random_feasible_lp(n=20, m=40, seed=seed))
    return {
        "backend": result.backend,
        "device": result.device,
        "dtype": result.dtype,
        "best_score": result.best_score,
        "best_primal_objective": result.best_primal_objective,
        "best_gap": result.best_gap,
        "samples_per_second": result.history[-1].samples_per_second,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare explicit NumPy, Torch, and Numba LP sampling backends.")
    parser.add_argument("--backend", default="all", choices=["all"] + DEFAULT_BACKENDS)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    backends = DEFAULT_BACKENDS if args.backend == "all" else [args.backend]
    outcomes: list[dict[str, object]] = []
    for backend in backends:
        try:
            outcomes.append(
                run_backend(
                    backend,
                    samples=args.samples,
                    chunk_size=args.chunk_size,
                    iterations=args.iterations,
                    seed=args.seed,
                    dtype=args.dtype,
                )
            )
        except RuntimeError as exc:
            outcomes.append({"backend": backend, "skipped": str(exc)})
    pprint(outcomes)


if __name__ == "__main__":
    main()
