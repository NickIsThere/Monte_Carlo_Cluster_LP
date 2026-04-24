from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lpas.gpu.torch_scoring import evaluate_primal_dual_batch_torch
from tests.backend_helpers import make_optimal_pair, make_parallel_test_problem


def _problem_tensors():
    problem = make_parallel_test_problem()
    A = torch.as_tensor(problem.A, dtype=torch.float64)
    b = torch.as_tensor(problem.b, dtype=torch.float64)
    c = torch.as_tensor(problem.c, dtype=torch.float64)
    return A, b, c


def test_shapes_are_correct() -> None:
    A, b, c = _problem_tensors()
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=torch.float64)
    Y = torch.tensor([[0.0, 0.0], [0.25, 0.25], [0.5, 0.1]], dtype=torch.float64)
    metrics = evaluate_primal_dual_batch_torch(A, b, c, X, Y)
    assert metrics["primal_objective"].shape == (3,)
    assert metrics["dual_objective"].shape == (3,)
    assert metrics["gap"].shape == (3,)
    assert metrics["primal_violation"].shape == (3,)
    assert metrics["dual_violation"].shape == (3,)
    assert metrics["complementarity"].shape == (3,)
    assert metrics["primal_slack"].shape == (3, 2)
    assert metrics["dual_slack"].shape == (3, 2)
    assert metrics["active_mask"].shape == (3, 2)


def test_feasible_x_has_near_zero_primal_violation() -> None:
    A, b, c = _problem_tensors()
    x, y = make_optimal_pair()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.as_tensor(np.asarray([x]), dtype=torch.float64),
        torch.as_tensor(np.asarray([y]), dtype=torch.float64),
    )
    assert metrics["primal_violation"][0].item() <= 1e-8


def test_infeasible_x_has_positive_primal_violation() -> None:
    A, b, c = _problem_tensors()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.tensor([[4.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    )
    assert metrics["primal_violation"][0].item() > 0.0


def test_feasible_y_has_near_zero_dual_violation() -> None:
    A, b, c = _problem_tensors()
    x, y = make_optimal_pair()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.as_tensor(np.asarray([x]), dtype=torch.float64),
        torch.as_tensor(np.asarray([y]), dtype=torch.float64),
    )
    assert metrics["dual_violation"][0].item() <= 1e-8


def test_infeasible_y_has_positive_dual_violation() -> None:
    A, b, c = _problem_tensors()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    )
    assert metrics["dual_violation"][0].item() > 0.0


def test_feasible_pair_has_nonnegative_gap() -> None:
    A, b, c = _problem_tensors()
    x, y = make_optimal_pair()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.as_tensor(np.asarray([x]), dtype=torch.float64),
        torch.as_tensor(np.asarray([y]), dtype=torch.float64),
    )
    assert metrics["gap"][0].item() >= -1e-8


def test_complementarity_is_zero_for_known_optimal_pair() -> None:
    A, b, c = _problem_tensors()
    x, y = make_optimal_pair()
    metrics = evaluate_primal_dual_batch_torch(
        A,
        b,
        c,
        torch.as_tensor(np.asarray([x]), dtype=torch.float64),
        torch.as_tensor(np.asarray([y]), dtype=torch.float64),
    )
    assert metrics["complementarity"][0].item() <= 1e-8
