from __future__ import annotations

import argparse
from pprint import pprint

from lpas.experiments.generators import random_feasible_lp
from lpas.solver.parallel_solver import ParallelLPSolver
from lpas.utils.config import BackendConfig, ParallelSolverConfig, SamplerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parallel LP sampler on a random dense LP.")
    parser.add_argument("--backend", default="numpy_cpu", choices=["numpy_cpu", "torch_cpu", "torch_mps", "torch_cuda", "numba_cpu", "auto_torch"])
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    config = ParallelSolverConfig(
        samples_per_iteration=args.samples,
        chunk_size=args.chunk_size,
        elite_fraction=0.05,
        iterations=args.iterations,
        backend=BackendConfig(backend=args.backend, dtype=args.dtype),
        sampler=SamplerConfig(
            seed=args.seed,
            alpha=0.4,
            sigma_init=1.5,
            sigma_min=1e-3,
            sigma_max=4.0,
            primal_init_mean=0.5,
            dual_init_mean=0.5,
        ),
    )
    problem = random_feasible_lp(n=args.n, m=args.m, seed=args.seed)
    result = ParallelLPSolver(config).solve(problem)
    pprint(
        {
            "backend": result.backend,
            "device": result.device,
            "dtype": result.dtype,
            "best_score": result.best_score,
            "best_primal_objective": result.best_primal_objective,
            "best_gap": result.best_gap,
            "best_primal_violation": result.best_primal_violation,
            "best_dual_violation": result.best_dual_violation,
            "samples_per_second": result.history[-1].samples_per_second,
            "likely_active_constraints": result.likely_active_constraints.tolist(),
        }
    )


if __name__ == "__main__":
    main()
