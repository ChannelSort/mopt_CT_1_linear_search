"""Classical gradient methods for Lab 2."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from optimlib.core.base import FloatArray, ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import Callback, HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import FunctionEvaluationError, LineSearchError, StopOptimization
from optimlib.functions.base import MultivariateFunction, UnivariateFunction
from optimlib.optimizers.univariate import GoldenSection
from optimlib.utils.registry import register_optimizer
from optimlib.utils.validation import as_float_vector, ensure_finite, ensure_gradient


LOGGER = logging.getLogger(__name__)


class LineSearchAdapter(UnivariateFunction):
    """Convert ``f(x + alpha p)`` into a univariate objective."""

    def __init__(self, func: MultivariateFunction, x: FloatArray, direction: FloatArray) -> None:
        """Initialize adapter."""
        super().__init__()
        self.func = func
        self.x = np.array(x, dtype=np.float64, copy=True).reshape(func.dim)
        self.direction = np.array(direction, dtype=np.float64, copy=True).reshape(func.dim)
        self.name = "LineSearchAdapter"
        self.interval = (0.0, 1.0)

    @property
    def call_count(self) -> int:
        """Proxy objective calls to the underlying function."""
        return self.func.call_count

    @property
    def grad_count(self) -> int:
        """Proxy gradient calls to the underlying function."""
        return self.func.grad_count

    def reset_count(self) -> None:
        """Do not reset underlying counters during line search."""

    def __call__(self, alpha: object) -> float:
        """Evaluate ``phi(alpha)``."""
        a = float(np.asarray(alpha, dtype=np.float64).reshape(-1)[0])
        return self.func(self.x + a * self.direction)

    def gradient(self, alpha: object) -> FloatArray:
        """Return ``phi'(alpha)``."""
        a = float(np.asarray(alpha, dtype=np.float64).reshape(-1)[0])
        grad = self.func.gradient(self.x + a * self.direction)
        return np.array([float(np.dot(grad, self.direction))], dtype=np.float64)

    def _evaluate(self, x: float) -> float:
        return self.__call__(x)


class GradientOptimizer(ABC):
    """Base class for gradient optimizers with callback support."""

    name = "GradientOptimizer"

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Initialize optimizer."""
        self.callbacks = list(callbacks or [])

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        """Minimize a differentiable multivariate function."""
        if not isinstance(func, MultivariateFunction):
            raise TypeError("GradientOptimizer requires a MultivariateFunction.")

        history = HistoryCallback()
        callbacks: list[Callback] = [history, *self.callbacks]
        for callback in callbacks:
            callback.on_start()

        x = func.initial_point()
        f_value = ensure_finite(func(x), "initial objective")
        grad = ensure_gradient(func.gradient(x), dim=func.dim)
        converged = float(np.linalg.norm(grad)) <= config.tol_grad
        message = "Initial point satisfies gradient tolerance." if converged else "Maximum iterations reached."
        n_iter = 0

        try:
            for iteration in range(config.max_iter):
                if converged:
                    break
                direction = self._search_direction(grad)
                alpha = self._compute_step(func, x, f_value, grad, direction, config)
                if not np.isfinite(alpha) or alpha <= 0.0:
                    raise LineSearchError(f"Invalid step size: {alpha}")
                x_next = as_float_vector(x + alpha * direction, dim=func.dim)
                f_next = ensure_finite(func(x_next), "objective")
                grad_next = ensure_gradient(func.gradient(x_next), dim=func.dim)
                state = StepState(
                    iteration=iteration,
                    x=x_next,
                    f=f_next,
                    grad=grad_next,
                    step_size=alpha,
                    extra_metrics={"grad_norm": float(np.linalg.norm(grad_next))},
                )
                n_iter = iteration + 1
                step_norm = float(np.linalg.norm(x_next - x))
                x, f_value, grad = x_next, f_next, grad_next
                grad_norm = float(np.linalg.norm(grad))
                for callback in callbacks:
                    callback.on_step(state)
                if grad_norm <= config.tol_grad:
                    converged = True
                    message = f"Gradient norm tolerance reached: {grad_norm:.3e}."
                    break
                if step_norm <= config.tol_step:
                    converged = True
                    message = f"Step tolerance reached: {step_norm:.3e}."
                    break
        except StopOptimization as exc:
            message = exc.message
            converged = True

        result = OptimizationResult(
            x=x,
            f=f_value,
            n_iter=n_iter,
            n_calls=func.call_count,
            n_grad_calls=func.grad_count,
            converged=converged,
            message=message,
            history=history.history,
            metadata={"optimizer": self.name},
        )
        for callback in callbacks:
            callback.on_end(result)
        return result

    def _search_direction(self, grad: FloatArray) -> FloatArray:
        """Return default steepest descent direction ``p=-grad``."""
        return -grad

    @abstractmethod
    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        """Compute accepted step size."""


class ConstantStepGD(GradientOptimizer):
    """Gradient descent with fixed step size."""

    name = "ConstantStepGD"

    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        return config.alpha


class ArmijoBacktracking(GradientOptimizer):
    """Backtracking satisfying Armijo sufficient decrease."""

    name = "ArmijoBacktracking"

    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        derphi0 = float(np.dot(grad, direction))
        if derphi0 >= 0.0:
            raise LineSearchError("Direction is not a descent direction.")
        alpha = config.alpha_init
        for _ in range(config.max_backtrack):
            value = func(x + alpha * direction)
            if value <= f_value + config.c1 * alpha * derphi0:
                return alpha
            alpha *= config.rho
            if alpha <= config.tol_step:
                break
        LOGGER.debug("Armijo reached max_backtrack.")
        return max(alpha, config.tol_step)


class StrongWolfe(GradientOptimizer):
    """Strong Wolfe line search using Nocedal-Wright Algorithm 3.5."""

    name = "StrongWolfe"

    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        phi0 = f_value
        derphi0 = float(np.dot(grad, direction))
        if derphi0 >= 0.0:
            raise LineSearchError("Direction is not a descent direction.")

        alpha_prev = 0.0
        phi_prev = phi0
        alpha = config.alpha_init
        for iteration in range(config.max_backtrack):
            phi_alpha = self._phi(func, x, direction, alpha)
            if phi_alpha > phi0 + config.c1 * alpha * derphi0 or (iteration > 0 and phi_alpha >= phi_prev):
                return self._zoom(func, x, direction, alpha_prev, alpha, phi0, derphi0, config)
            derphi_alpha = self._derphi(func, x, direction, alpha)
            if abs(derphi_alpha) <= config.c2 * abs(derphi0):
                return alpha
            if derphi_alpha >= 0.0:
                return self._zoom(func, x, direction, alpha, alpha_prev, phi0, derphi0, config)
            alpha_prev = alpha
            phi_prev = phi_alpha
            alpha *= 2.0
        LOGGER.debug("Strong Wolfe bracketing reached max_backtrack.")
        return max(alpha_prev, config.tol_step)

    def _zoom(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        direction: FloatArray,
        alpha_lo: float,
        alpha_hi: float,
        phi0: float,
        derphi0: float,
        config: OptimizerConfig,
    ) -> float:
        """Zoom phase of the strong Wolfe algorithm."""
        lo = float(alpha_lo)
        hi = float(alpha_hi)
        phi_lo = self._phi(func, x, direction, lo)
        for _ in range(config.max_backtrack):
            alpha = 0.5 * (lo + hi)
            if abs(hi - lo) <= config.tol_step:
                return max(alpha, config.tol_step)
            phi_alpha = self._phi(func, x, direction, alpha)
            if phi_alpha > phi0 + config.c1 * alpha * derphi0 or phi_alpha >= phi_lo:
                hi = alpha
            else:
                derphi_alpha = self._derphi(func, x, direction, alpha)
                if abs(derphi_alpha) <= config.c2 * abs(derphi0):
                    return alpha
                if derphi_alpha * (hi - lo) >= 0.0:
                    hi = lo
                lo = alpha
                phi_lo = phi_alpha
        LOGGER.debug("Strong Wolfe zoom reached max_backtrack.")
        return max(0.5 * (lo + hi), config.tol_step)

    @staticmethod
    def _phi(func: MultivariateFunction, x: FloatArray, direction: FloatArray, alpha: float) -> float:
        return ensure_finite(func(x + alpha * direction), "phi")

    @staticmethod
    def _derphi(func: MultivariateFunction, x: FloatArray, direction: FloatArray, alpha: float) -> float:
        grad = ensure_gradient(func.gradient(x + alpha * direction), dim=func.dim)
        derivative = float(np.dot(grad, direction))
        if not np.isfinite(derivative):
            raise FunctionEvaluationError("Non-finite line-search derivative.")
        return derivative


class SteepestDescent(GradientOptimizer):
    """Steepest descent with exact line search via ``GoldenSection``."""

    name = "SteepestDescent"

    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        adapter = LineSearchAdapter(func, x, direction)
        upper = self._bracket_upper(adapter, f_value, config)
        line_config = config.model_copy(update={"interval": (0.0, upper), "tol": config.line_search_tol, "max_iter": config.line_search_max_iter})
        result = GoldenSection().minimize(adapter, line_config)
        return max(float(result.x), config.tol_step)

    @staticmethod
    def _bracket_upper(adapter: LineSearchAdapter, f0: float, config: OptimizerConfig) -> float:
        upper = config.alpha_init
        current = adapter(upper)
        if current >= f0:
            return upper
        for _ in range(config.max_backtrack):
            next_upper = 2.0 * upper
            candidate = adapter(next_upper)
            if candidate >= current:
                return next_upper
            upper, current = next_upper, candidate
        return upper


for _cls in (ConstantStepGD, ArmijoBacktracking, StrongWolfe, SteepestDescent):
    register_optimizer(_cls.__name__, _cls)

register_optimizer("constant_step_gd", ConstantStepGD)
register_optimizer("armijo_backtracking", ArmijoBacktracking)
register_optimizer("strong_wolfe", StrongWolfe)
register_optimizer("steepest_descent", SteepestDescent)
