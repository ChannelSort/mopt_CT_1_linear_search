"""Validation helpers for numerical inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimlib.core.base import FloatArray
from optimlib.exceptions import FunctionEvaluationError


def as_float_vector(x: Any, dim: int | None = None) -> FloatArray:
    """Convert ``x`` to a flat finite float64 vector."""
    array = np.asarray(x, dtype=np.float64).reshape(-1)
    if dim is not None and array.shape != (dim,):
        raise ValueError(f"Expected shape ({dim},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise FunctionEvaluationError("Point contains NaN or Inf.")
    return array


def as_float_scalar(x: Any) -> float:
    """Convert scalar-like input to a finite float."""
    array = np.asarray(x, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"Expected scalar input, got shape {array.shape}.")
    value = float(array.reshape(-1)[0])
    if not np.isfinite(value):
        raise FunctionEvaluationError(f"Scalar is not finite: {value}.")
    return value


def ensure_finite(value: float, name: str = "value") -> float:
    """Validate finite scalar value."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise FunctionEvaluationError(f"{name} is not finite: {scalar}.")
    return scalar


def ensure_gradient(gradient: Any, dim: int) -> FloatArray:
    """Validate and return a flat finite gradient vector."""
    return as_float_vector(gradient, dim=dim)
