"""Custom exceptions for optimlib."""

from __future__ import annotations


class OptimLibError(Exception):
    """Base class for optimlib exceptions."""


class ConfigurationError(OptimLibError):
    """Raised when configuration loading or validation fails."""


class FunctionEvaluationError(OptimLibError):
    """Raised when a function, gradient, or point is invalid."""


class LineSearchError(OptimLibError):
    """Raised when a line search cannot produce a valid step."""


class StopOptimization(OptimLibError):
    """Signal used by callbacks for controlled early stopping."""

    def __init__(self, message: str = "Optimization stopped by callback.") -> None:
        """Initialize the stop signal.

        Args:
            message: Human-readable stop reason.
        """
        super().__init__(message)
        self.message = message
