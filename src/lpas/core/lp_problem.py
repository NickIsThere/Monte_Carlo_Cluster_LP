from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpas.utils.validation import as_matrix, as_vector, validate_lp_dimensions


@dataclass(frozen=True)
class LPProblem:
    """Dense LP in the form Ax <= b, x >= 0 with max/min objective sense."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    sense: str = "max"
    variable_lower_bounds: str = "nonnegative"
    constraint_type: str = "le"

    def __post_init__(self) -> None:
        A = as_matrix(self.A, name="A")
        b = as_vector(self.b, name="b")
        c = as_vector(self.c, name="c")
        validate_lp_dimensions(A, b, c)
        if self.sense not in {"max", "min"}:
            raise ValueError("sense must be 'max' or 'min'")
        if self.variable_lower_bounds != "nonnegative":
            raise ValueError("Only x >= 0 is supported in the prototype")
        if self.constraint_type != "le":
            raise ValueError("Only Ax <= b constraints are supported in the prototype")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    def to_maximization(self) -> "LPProblem":
        if self.sense == "max":
            return self
        return LPProblem(A=self.A, b=self.b, c=-self.c, sense="max")

    def objective_value(self, x: np.ndarray) -> float:
        value = float(np.dot(self.c, np.asarray(x, dtype=float)))
        if self.sense == "max":
            return value
        return value

    def maximization_objective_value(self, x: np.ndarray) -> float:
        max_problem = self.to_maximization()
        return float(np.dot(max_problem.c, np.asarray(x, dtype=float)))

    def to_scipy_linprog(self) -> dict[str, object]:
        bounds = [(0.0, None)] * self.n
        if self.sense == "max":
            c = -self.c
            objective_multiplier = -1.0
        else:
            c = self.c
            objective_multiplier = 1.0
        return {
            "c": c,
            "A_ub": self.A,
            "b_ub": self.b,
            "bounds": bounds,
            "method": "highs",
            "objective_multiplier": objective_multiplier,
        }
