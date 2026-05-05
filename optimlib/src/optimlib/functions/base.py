"""Base classes for objective functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from optimlib.core.base import FloatArray, ObjectiveFunction
from optimlib.utils.numerics import approximate_derivative, approximate_gradient
from optimlib.utils.validation import as_float_scalar, as_float_vector, ensure_finite, ensure_gradient


class CallableCounterMixin:
    """Track objective and gradient calls."""

    def __init__(self) -> None:
        """Initialize counters."""
        self._call_count = 0
        self._grad_count = 0

    def reset_count(self) -> None:
        """Reset objective and gradient call counters."""
        self._call_count = 0
        self._grad_count = 0

    @property
    def call_count(self) -> int:
        """Return objective call count."""
        return self._call_count

    @property
    def grad_count(self) -> int:
        """Return gradient call count."""
        return self._grad_count

    def _increment_call_count(self, amount: int = 1) -> None:
        self._call_count += amount

    def _increment_grad_count(self) -> None:
        self._grad_count += 1


class UnivariateFunction(CallableCounterMixin, ObjectiveFunction, ABC):
    """Base class for scalar objective functions."""

    name: str
    interval: tuple[float, float]
    x_min: float | None = None
    f_min: float | None = None

    def __init__(self) -> None:
        """Initialize counters."""
        super().__init__()

    def __call__(self, x: Any) -> float:
        """Evaluate objective at scalar ``x``."""
        scalar = as_float_scalar(x)
        self._increment_call_count()
        return ensure_finite(self._evaluate(scalar), name=f"{self.name}(x)")

    @abstractmethod
    def _evaluate(self, x: float) -> float:
        """Evaluate function without changing counters."""

    def evaluate_many(self, x_grid: FloatArray) -> FloatArray:
        """Vectorized objective evaluation for a grid of scalar points."""
        grid = np.asarray(x_grid, dtype=np.float64)
        if not np.all(np.isfinite(grid)):
            raise ValueError("Grid contains NaN or Inf.")
        values = np.vectorize(self._evaluate, otypes=[np.float64])(grid)
        self._increment_call_count(int(grid.size))
        if not np.all(np.isfinite(values)):
            raise ValueError("Objective returned NaN or Inf on grid.")
        values_array: FloatArray = np.asarray(values, dtype=np.float64)
        return values_array

    def gradient(self, x: Any) -> FloatArray:
        """Return scalar derivative as shape ``(1,)``."""
        scalar = as_float_scalar(x)
        self._increment_grad_count()
        return np.array([approximate_derivative(self, scalar)], dtype=np.float64)


class MultivariateFunction(CallableCounterMixin, ObjectiveFunction, ABC):
    """Base class for differentiable multivariate objectives."""

    name: str
    x0: FloatArray
    global_minimizers: tuple[FloatArray, ...] = ()
    f_min: float | None = None

    def __init__(self, dim: int) -> None:
        """Initialize dimension and counters."""
        if dim <= 0:
            raise ValueError("dim must be positive.")
        super().__init__()
        self.dim = dim

    def __call__(self, x: Any) -> float:
        """Evaluate objective at a vector point."""
        x_vec = as_float_vector(x, dim=self.dim)
        self._increment_call_count()
        return ensure_finite(self._evaluate(x_vec), name=f"{self.name}(x)")

    @abstractmethod
    def _evaluate(self, x: FloatArray) -> float:
        """Evaluate function without changing counters."""

    def gradient(self, x: Any) -> FloatArray:
        """Return analytic gradient or central finite-difference fallback."""
        x_vec = as_float_vector(x, dim=self.dim)
        self._increment_grad_count()
        return ensure_gradient(approximate_gradient(self, x_vec), dim=self.dim)

    def initial_point(self) -> FloatArray:
        """Return a copy of the configured initial point."""
        return np.array(self.x0, dtype=np.float64, copy=True).reshape(self.dim)
