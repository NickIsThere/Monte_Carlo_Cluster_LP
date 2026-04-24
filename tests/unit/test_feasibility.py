from __future__ import annotations

import numpy as np

from lpas.core.feasibility import (
    dual_violation_norm,
    is_dual_feasible,
    is_primal_feasible,
    primal_violation_norm,
    project_nonnegative,
)
from lpas.core.lp_problem import LPProblem


def _problem() -> LPProblem:
    return LPProblem(
        A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        b=np.array([1.0, 0.8, 0.8]),
        c=np.array([1.0, 1.0]),
    )


def test_simple_feasible_point_returns_true() -> None:
    assert is_primal_feasible(_problem(), np.array([0.5, 0.5]))


def test_violated_constraints_return_false() -> None:
    assert not is_primal_feasible(_problem(), np.array([1.0, 1.0]))


def test_nonnegative_bounds_enforced() -> None:
    assert not is_primal_feasible(_problem(), np.array([-0.1, 0.5]))
    np.testing.assert_allclose(project_nonnegative(np.array([-1.0, 2.0])), np.array([0.0, 2.0]))


def test_tolerance_allows_tiny_violation() -> None:
    point = np.array([0.8 + 1e-8, 0.2])
    assert is_primal_feasible(_problem(), point, tol=1e-6)


def test_primal_violation_norm_increases_with_violation() -> None:
    problem = _problem()
    small = primal_violation_norm(problem, np.array([0.85, 0.2]))
    large = primal_violation_norm(problem, np.array([1.2, 1.2]))
    assert large > small > 0.0


def test_dual_feasibility_cases() -> None:
    problem = _problem()
    assert is_dual_feasible(problem, np.array([1.0, 0.0, 0.0]))
    assert not is_dual_feasible(problem, np.array([0.0, 0.0, 0.0]))
    assert not is_dual_feasible(problem, np.array([-1.0, 0.0, 0.0]))
    assert dual_violation_norm(problem, np.array([0.0, 0.0, 0.0])) > 0.0
