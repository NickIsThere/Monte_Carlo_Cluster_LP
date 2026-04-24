from __future__ import annotations

from typing import Any

from lpas.backends.base import CandidateBackend, BackendInfo
from lpas.backends.torch_device import (
    _require_torch,
    resolve_torch_device,
    synchronize_torch_device,
    torch_device_name,
    torch_dtype_from_name,
)
from lpas.core.lp_problem import LPProblem
from lpas.gpu.torch_active_sets import active_frequency_torch, active_support_reward_torch, dual_active_agreement_torch
from lpas.gpu.torch_sampler import make_torch_generator, sample_nonnegative_normal_torch, select_elites_torch, update_distribution_torch
from lpas.gpu.torch_scoring import evaluate_primal_dual_batch_torch, score_primal_dual_batch_torch
from lpas.utils.config import ParallelScoreConfig, ParallelSolverConfig


class TorchBackend(CandidateBackend):
    def __init__(self, requested_backend: str, *, dtype: str = "float32") -> None:
        self.torch = _require_torch()
        self.device = resolve_torch_device(requested_backend)
        self.dtype = torch_dtype_from_name(dtype, self.device)
        super().__init__(requested_backend, dtype=dtype)
        self.info = BackendInfo(
            requested_backend=requested_backend,
            resolved_backend=f"torch_{self.device.type}",
            device_name=torch_device_name(self.device),
            dtype=dtype,
        )

    def prepare_problem(self, problem: LPProblem) -> dict[str, Any]:
        return {
            "A": self.torch.as_tensor(problem.A, dtype=self.dtype, device=self.device),
            "b": self.torch.as_tensor(problem.b, dtype=self.dtype, device=self.device),
            "c": self.torch.as_tensor(problem.c, dtype=self.dtype, device=self.device),
        }

    def to_backend_array(self, value: Any) -> Any:
        return self.torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def initialize_distribution(self, problem: LPProblem, config: ParallelSolverConfig) -> dict[str, Any]:
        return {
            "mu_x": self.torch.full((problem.n,), config.sampler.primal_init_mean, dtype=self.dtype, device=self.device),
            "sigma_x": self.torch.full((problem.n,), config.sampler.sigma_init, dtype=self.dtype, device=self.device),
            "mu_y": self.torch.full((problem.m,), config.sampler.dual_init_mean, dtype=self.dtype, device=self.device),
            "sigma_y": self.torch.full((problem.m,), config.sampler.sigma_init, dtype=self.dtype, device=self.device),
            "generator": make_torch_generator(self.device, config.sampler.seed),
        }

    def sample_candidates(self, state: dict[str, Any], sample_size: int) -> tuple[Any, Any]:
        return sample_nonnegative_normal_torch(
            state["mu_x"],
            state["sigma_x"],
            state["mu_y"],
            state["sigma_y"],
            sample_size,
            generator=state["generator"],
        )

    def evaluate_batch(self, problem_data: dict[str, Any], X: Any, Y: Any, *, active_epsilon: float) -> dict[str, Any]:
        return evaluate_primal_dual_batch_torch(
            problem_data["A"],
            problem_data["b"],
            problem_data["c"],
            X,
            Y,
            active_epsilon=active_epsilon,
        )

    def score_batch(
        self,
        problem_data: dict[str, Any],
        X: Any,
        Y: Any,
        metrics: dict[str, Any],
        weights: ParallelScoreConfig,
        *,
        active_frequency: Any | None,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> Any:
        del problem_data, X, active_epsilon
        active_support = None
        if weights.active_support != 0.0 and active_frequency is not None:
            active_support = active_support_reward_torch(metrics["active_mask"], active_frequency)
        active_agreement = None
        active_conflict = None
        if weights.active_agreement != 0.0 or weights.active_conflict != 0.0:
            active_agreement, active_conflict = dual_active_agreement_torch(
                Y,
                metrics["active_mask"],
                dual_positive_epsilon=dual_positive_epsilon,
            )
        return score_primal_dual_batch_torch(
            metrics,
            weights,
            active_support=active_support,
            active_agreement=active_agreement,
            active_conflict=active_conflict,
        )

    def select_elites(self, X: Any, Y: Any, scores: Any, elite_count: int) -> tuple[Any, Any, Any, Any]:
        return select_elites_torch(X, Y, scores, elite_count)

    def update_distribution(self, state: dict[str, Any], elite_X: Any, elite_Y: Any, config: ParallelSolverConfig) -> dict[str, Any]:
        mu_x, sigma_x, mu_y, sigma_y = update_distribution_torch(
            state["mu_x"],
            state["sigma_x"],
            state["mu_y"],
            state["sigma_y"],
            elite_X,
            elite_Y,
            alpha=config.sampler.alpha,
            min_sigma=config.sampler.sigma_min,
            max_sigma=config.sampler.sigma_max,
        )
        state["mu_x"] = mu_x
        state["sigma_x"] = sigma_x
        state["mu_y"] = mu_y
        state["sigma_y"] = sigma_y
        return state

    def compute_active_statistics(
        self,
        problem_data: dict[str, Any],
        elite_X: Any,
        elite_Y: Any,
        *,
        active_epsilon: float,
        dual_positive_epsilon: float,
    ) -> dict[str, Any]:
        del elite_Y, dual_positive_epsilon
        primal_slack = problem_data["b"].unsqueeze(0) - elite_X @ problem_data["A"].transpose(0, 1)
        active_mask = primal_slack <= active_epsilon
        active_frequency = active_frequency_torch(active_mask)
        likely_active = self.torch.nonzero(active_frequency >= 0.5, as_tuple=False).flatten()
        return {
            "elite_active_mask": active_mask,
            "active_frequency": active_frequency,
            "likely_active_constraints": likely_active,
        }

    def concatenate(self, values: Any, *, axis: int = 0) -> Any:
        return self.torch.cat(list(values), dim=axis)

    def take(self, value: Any, indices: Any) -> Any:
        return value[indices]

    def to_numpy(self, value: Any):
        return value.detach().cpu().numpy()

    def to_scalar(self, value: Any) -> float:
        return float(value.detach().cpu().item())

    def sync(self) -> None:
        synchronize_torch_device(self.device)
