"""Lab 2 two-dimensional objective functions."""

from __future__ import annotations

import math

import numpy as np

from optimlib.core.base import FloatArray
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.registry import register_function
from optimlib.utils.validation import as_float_vector, ensure_gradient


class WellConditionedQuadratic(MultivariateFunction):
    """``0.5 * (x^2 + y^2)`` with condition number 1."""

    name = "WellConditionedQuadratic"
    x0 = np.array([-2.0, -2.0], dtype=np.float64)
    global_minimizers = (np.array([0.0, 0.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        if x0 is not None:
            self.x0 = np.array(x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        return 0.5 * float(np.dot(x, x))

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        return ensure_gradient(as_float_vector(x, 2), 2)


class IllConditionedQuadratic(MultivariateFunction):
    """``0.5*x^2 + 50*y^2`` with Hessian condition number 100."""

    name = "IllConditionedQuadratic"
    x0 = np.array([-2.0, -2.0], dtype=np.float64)
    global_minimizers = (np.array([0.0, 0.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        if x0 is not None:
            self.x0 = np.array(x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        return float(0.5 * x[0] ** 2 + 50.0 * x[1] ** 2)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        v = as_float_vector(x, 2)
        return np.array([v[0], 100.0 * v[1]], dtype=np.float64)


class Rosenbrock(MultivariateFunction):
    """Rosenbrock function; minimizer ``(1, 1)``, condition number about 2508."""

    name = "Rosenbrock"
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    global_minimizers = (np.array([1.0, 1.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        if x0 is not None:
            self.x0 = np.array(x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            value = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
        return float(value)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        v = as_float_vector(x, 2)
        return np.array(
            [-2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0] ** 2), 200.0 * (v[1] - v[0] ** 2)],
            dtype=np.float64,
        )


class Ackley(MultivariateFunction):
    """Two-dimensional Ackley objective with minimizer at the origin."""

    name = "Ackley"
    x0 = np.array([1.0, 1.0], dtype=np.float64)
    global_minimizers = (np.array([0.0, 0.0], dtype=np.float64),)
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        if x0 is not None:
            self.x0 = np.array(x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        r = math.sqrt(0.5 * float(np.dot(x, x)))
        c = 0.5 * float(np.cos(2.0 * math.pi * x[0]) + np.cos(2.0 * math.pi * x[1]))
        return float(-20.0 * math.exp(-0.2 * r) - math.exp(c) + math.e + 20.0)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        v = as_float_vector(x, 2)
        r = math.sqrt(0.5 * float(np.dot(v, v)))
        first = np.zeros(2, dtype=np.float64) if r <= 1e-15 else 2.0 * math.exp(-0.2 * r) * v / r
        c = 0.5 * float(np.cos(2.0 * math.pi * v[0]) + np.cos(2.0 * math.pi * v[1]))
        return first + math.pi * math.exp(c) * np.sin(2.0 * math.pi * v)


class Himmelblau(MultivariateFunction):
    """Himmelblau function with four global minimizers."""

    name = "Himmelblau"
    x0 = np.array([2.0, 2.0], dtype=np.float64)
    global_minimizers = (
        np.array([3.0, 2.0], dtype=np.float64),
        np.array([-2.805118, 3.131312], dtype=np.float64),
        np.array([-3.779310, -3.283186], dtype=np.float64),
        np.array([3.584428, -1.848126], dtype=np.float64),
    )
    f_min = 0.0

    def __init__(self, x0: list[float] | None = None) -> None:
        super().__init__(2)
        if x0 is not None:
            self.x0 = np.array(x0, dtype=np.float64)

    def _evaluate(self, x: FloatArray) -> float:
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        return float(a * a + b * b)

    def gradient(self, x: object) -> FloatArray:
        self._increment_grad_count()
        v = as_float_vector(x, 2)
        a = v[0] ** 2 + v[1] - 11.0
        b = v[0] + v[1] ** 2 - 7.0
        return np.array([4.0 * v[0] * a + 2.0 * b, 2.0 * a + 4.0 * v[1] * b], dtype=np.float64)


for _cls in (WellConditionedQuadratic, IllConditionedQuadratic, Rosenbrock, Ackley, Himmelblau):
    register_function(_cls.__name__, _cls)

register_function("well_quad", WellConditionedQuadratic)
register_function("ill_quad", IllConditionedQuadratic)
register_function("rosenbrock", Rosenbrock)
register_function("ackley", Ackley)
register_function("himmelblau", Himmelblau)
