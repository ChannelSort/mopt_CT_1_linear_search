"""Numerical routines for gradients, convergence, and conditioning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from optimlib.core.base import FloatArray, StepState
from optimlib.core.config import OptimizerConfig
from optimlib.utils.validation import as_float_vector


def approximate_gradient(
    func: Callable[[FloatArray], float],
    x: FloatArray,
    h: float = 1e-5,
    method: Literal["central", "forward"] = "central",
) -> FloatArray:
    """Approximate gradient using finite differences.

    Central difference uses ``(f(x+h e_i)-f(x-h e_i))/(2h)`` and is the
    default. Forward difference is available only when explicitly requested.
    """
    if h <= 0.0:
        raise ValueError("h must be positive.")
    x_vec = as_float_vector(x)
    grad = np.empty_like(x_vec)
    for idx in range(x_vec.size):
        step = np.zeros_like(x_vec)
        step[idx] = h
        if method == "central":
            grad[idx] = (float(func(x_vec + step)) - float(func(x_vec - step))) / (2.0 * h)
        elif method == "forward":
            grad[idx] = (float(func(x_vec + step)) - float(func(x_vec))) / h
        else:
            raise ValueError(f"Unsupported finite-difference method: {method}")
    return grad


def approximate_derivative(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Approximate derivative of a scalar function by central difference."""
    if h <= 0.0:
        raise ValueError("h must be positive.")
    return (float(func(x + h)) - float(func(x - h))) / (2.0 * h)


def check_condition_number(hessian_approx: FloatArray) -> float:
    """Return matrix condition number in the 2-norm."""
    matrix = np.asarray(hessian_approx, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    return float(np.linalg.cond(matrix))


def is_converged(state: StepState, config: OptimizerConfig) -> bool:
    """Check gradient and step convergence criteria."""
    grad_ok = state.grad is not None and float(np.linalg.norm(state.grad)) <= config.tol_grad
    step_ok = state.step_size is not None and abs(state.step_size) <= config.tol_step
    return grad_ok or step_ok
