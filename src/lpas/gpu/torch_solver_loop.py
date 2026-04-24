from __future__ import annotations

from lpas.backends.torch_backend import TorchBackend
from lpas.core.lp_problem import LPProblem
from lpas.solver.parallel_solver import solve_with_backend
from lpas.solver.result import ParallelSolverResult
from lpas.utils.config import ParallelSolverConfig


def solve_primal_dual_parallel_torch(problem: LPProblem, config: ParallelSolverConfig) -> ParallelSolverResult:
    backend = TorchBackend(config.backend.backend, dtype=config.backend.dtype)
    return solve_with_backend(problem, config, backend)
