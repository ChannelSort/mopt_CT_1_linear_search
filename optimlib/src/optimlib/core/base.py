"""Core abstractions and result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ScalarOrArray = float | FloatArray


@dataclass(frozen=True, slots=True)
class StepState:
    """Immutable optimization step snapshot passed to callbacks.

    Attributes:
        iteration: Zero-based iteration index.
        x: Current point or scalar encoded as a flat vector.
        f: Objective value at ``x``.
        grad: Gradient at ``x`` when available.
        step_size: Accepted step size, interval length, or grid spacing.
        extra_metrics: Method-specific immutable diagnostics.
    """

    iteration: int
    x: FloatArray
    f: float
    grad: FloatArray | None = None
    step_size: float | None = None
    extra_metrics: Mapping[str, float | int | str | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy arrays so callbacks cannot mutate optimizer internals."""
        object.__setattr__(self, "x", np.array(self.x, dtype=np.float64, copy=True).reshape(-1))
        if self.grad is not None:
            object.__setattr__(self, "grad", np.array(self.grad, dtype=np.float64, copy=True).reshape(-1))
        object.__setattr__(self, "extra_metrics", dict(self.extra_metrics))


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Optimization result object.

    Attributes:
        x: Final point as a scalar or flat vector.
        f: Final objective value.
        n_iter: Number of completed iterations.
        n_calls: Number of objective evaluations.
        n_grad_calls: Number of gradient evaluations.
        converged: Whether a stopping criterion was met.
        message: Termination reason.
        history: Step snapshots, usually collected by ``HistoryCallback``.
        metadata: Method-specific diagnostics and experiment parameters.
    """

    x: ScalarOrArray
    f: float
    n_iter: int
    n_calls: int
    n_grad_calls: int
    converged: bool
    message: str
    history: Sequence[StepState] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy result arrays and history."""
        if not np.isscalar(self.x):
            object.__setattr__(self, "x", np.array(self.x, dtype=np.float64, copy=True).reshape(-1))
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "metadata", dict(self.metadata))


class ObjectiveFunction(ABC):
    """Abstract objective with objective and gradient counters."""

    name: str

    @abstractmethod
    def __call__(self, x: Any) -> float:
        """Evaluate objective at ``x``."""

    @abstractmethod
    def gradient(self, x: Any) -> FloatArray:
        """Evaluate or approximate gradient at ``x``."""

    @abstractmethod
    def reset_count(self) -> None:
        """Reset objective and gradient counters."""

    @property
    @abstractmethod
    def call_count(self) -> int:
        """Return objective evaluation count."""

    @property
    @abstractmethod
    def grad_count(self) -> int:
        """Return gradient evaluation count."""


class Optimizer(ABC):
    """Abstract optimizer interface."""

    name: str

    @abstractmethod
    def minimize(self, func: ObjectiveFunction, config: Any) -> OptimizationResult:
        """Minimize ``func`` with ``config``."""
