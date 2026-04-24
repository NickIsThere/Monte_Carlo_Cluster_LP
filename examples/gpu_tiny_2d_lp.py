from __future__ import annotations

import argparse
from pprint import pprint

import numpy as np

from lpas.core.lp_problem import LPProblem
from lpas.solver.parallel_solver import ParallelLPSolver
from lpas.solver.scipy_handoff import solve_with_scipy
from lpas.utils.config import BackendConfig, ParallelSolverConfig, SamplerConfig


def tiny_gpu_lp() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 2.0], [4.0, 2.0]], dtype=float),
        b=np.array([4.0, 12.0], dtype=float),
        c=np.array([1.0, 1.0], dtype=float),
        sense="max",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parallel LP sampler on a tiny 2D LP.")
    parser.add_argument("--backend", default="numpy_cpu", choices=["numpy_cpu", "torch_cpu", "torch_mps", "torch_cuda", "numba_cpu", "auto_torch"])
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
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
    problem = tiny_gpu_lp()
    result = ParallelLPSolver(config).solve(problem)
    scipy_result = solve_with_scipy(problem)
    pprint(
        {
            "backend": result.backend,
            "device": result.device,
            "dtype": result.dtype,
            "best_primal_objective": result.best_primal_objective,
            "best_dual_objective": result.best_dual_objective,
            "gap": result.best_gap,
            "primal_violation": result.best_primal_violation,
            "dual_violation": result.best_dual_violation,
            "best_x": result.best_x.tolist(),
            "best_y": result.best_y.tolist(),
            "likely_active_constraints": result.likely_active_constraints.tolist(),
            "scipy_objective": scipy_result.objective,
        }
    )


if __name__ == "__main__":
    main()
