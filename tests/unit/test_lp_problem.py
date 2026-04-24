from __future__ import annotations

import numpy as np
import pytest

from lpas.core.lp_problem import LPProblem


def test_valid_lp_initializes_correctly() -> None:
    problem = LPProblem(A=[[1.0, 2.0], [3.0, 4.0]], b=[5.0, 6.0], c=[1.0, 1.0], sense="max")
    assert problem.m == 2
    assert problem.n == 2
    assert isinstance(problem.A, np.ndarray)
    assert isinstance(problem.b, np.ndarray)
    assert isinstance(problem.c, np.ndarray)


def test_invalid_shape_of_A_raises() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=[1.0, 2.0], b=[1.0], c=[1.0, 2.0])


def test_invalid_shape_of_b_raises() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=[[1.0, 2.0]], b=[1.0, 2.0], c=[1.0, 2.0])


def test_invalid_shape_of_c_raises() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=[[1.0, 2.0]], b=[1.0], c=[1.0])


def test_nan_values_rejected() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=[[np.nan]], b=[1.0], c=[1.0])


def test_infinite_values_rejected() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=[[1.0]], b=[np.inf], c=[1.0])


def test_zero_dimensional_lps_are_rejected() -> None:
    with pytest.raises(ValueError):
        LPProblem(A=np.empty((0, 0)), b=np.array([]), c=np.array([]))


def test_conversion_from_lists_and_min_to_max() -> None:
    problem = LPProblem(A=[[1.0, 0.0]], b=[2.0], c=[3.0, 4.0], sense="min")
    converted = problem.to_maximization()
    np.testing.assert_allclose(converted.c, np.array([-3.0, -4.0]))
    assert converted.sense == "max"


def test_negative_b_allowed() -> None:
    problem = LPProblem(A=[[1.0]], b=[-1.0], c=[1.0])
    assert problem.b[0] == -1.0
