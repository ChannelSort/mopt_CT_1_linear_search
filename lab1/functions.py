"""Lab 1 scalar objective functions."""

from __future__ import annotations

import math

import numpy as np

from optimlib.functions.base import UnivariateFunction
from optimlib.utils.registry import register_function


class GoodUnimodalFunction(UnivariateFunction):
    """A smooth quadratic function with a clear minimum."""

    name = "GoodUnimodalFunction"
    interval = (-6.0, 8.0)
    x_min = 2.0
    f_min = 1.0

    def _evaluate(self, x: float) -> float:
        return (x - 2.0) ** 2 + 1.0


class PlateauAsymmetricUnimodalFunction(UnivariateFunction):
    """Extremely asymmetric unimodal function from the original Lab 1."""

    name = "PlateauAsymmetricUnimodalFunction"
    interval = (-2.0, 6.0)
    x_min = 1.0
    f_min = 0.0

    def _evaluate(self, x: float) -> float:
        dx = x - 1.0
        left = math.exp(5.0 * (-dx))
        right = 0.01 * dx**2
        return left if dx < 0.0 else right


class SecondSpecialUnimodalFunction(UnivariateFunction):
    """Original special function ``exp(-x) + |x - 0.5|^1.5``."""

    name = "SecondSpecialUnimodalFunction"
    interval = (-1.0, 2.0)
    x_min = None
    f_min = None

    def _evaluate(self, x: float) -> float:
        return float(math.exp(-x) + abs(x - 0.5) ** 1.5)


class MultimodalRastrigin(UnivariateFunction):
    """One-dimensional Rastrigin objective for blind vs reconnaissance search."""

    name = "MultimodalRastrigin"
    interval = (-5.12, 5.12)
    x_min = 0.0
    f_min = 0.0

    def _evaluate(self, x: float) -> float:
        return 10.0 + x**2 - 10.0 * float(np.cos(2.0 * np.pi * x))


for _cls in (
    GoodUnimodalFunction,
    PlateauAsymmetricUnimodalFunction,
    SecondSpecialUnimodalFunction,
    MultimodalRastrigin,
):
    register_function(_cls.__name__, _cls)

register_function("good_unimodal", GoodUnimodalFunction)
register_function("plateau_asymmetric", PlateauAsymmetricUnimodalFunction)
register_function("second_special", SecondSpecialUnimodalFunction)
register_function("multimodal_rastrigin", MultimodalRastrigin)
