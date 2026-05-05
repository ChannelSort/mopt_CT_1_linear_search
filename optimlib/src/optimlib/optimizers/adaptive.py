"""Adaptive and momentum gradient methods for Lab 3."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from optimlib.core.base import FloatArray, ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import StopOptimization
from optimlib.functions.base import MultivariateFunction
from optimlib.optimizers.gradient import GradientOptimizer
from optimlib.utils.registry import register_optimizer
from optimlib.utils.validation import as_float_vector, ensure_finite, ensure_gradient


class AdaptiveOptimizer(GradientOptimizer):
    """Base class for stateful gradient optimizers."""

    name = "AdaptiveOptimizer"

    @abstractmethod
    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        """Return additive update direction and diagnostics."""

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        """Minimize with stateful adaptive updates initialized per run."""
        if not isinstance(func, MultivariateFunction):
            raise TypeError("AdaptiveOptimizer requires a MultivariateFunction.")
        self._reset_state()
        history = HistoryCallback()
        callbacks = [history, *self.callbacks]
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
                update, metrics = self._step_from_state(func, x, grad, iteration, config)
                step_norm = float(np.linalg.norm(update))
                x_next = as_float_vector(x + update, dim=func.dim)
                f_next = ensure_finite(func(x_next), "objective")
                grad_next = ensure_gradient(func.gradient(x_next), dim=func.dim)
                state_metrics = dict(metrics)
                grad_norm = float(np.linalg.norm(grad_next))
                state_metrics["grad_norm"] = grad_norm
                state = StepState(iteration, x_next, f_next, grad_next, step_norm, state_metrics)
                n_iter = iteration + 1
                x, f_value, grad = x_next, f_next, grad_next
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

    def _reset_state(self) -> None:
        """Reset method-specific buffers."""

    def _compute_step(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        f_value: float,
        grad: FloatArray,
        direction: FloatArray,
        config: OptimizerConfig,
    ) -> float:
        return 1.0

    def _search_direction(self, grad: FloatArray) -> FloatArray:
        """Unused by adaptive methods; direction is built in ``_compute_step`` path."""
        return -grad

    def _step_from_state(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        """Return update vector for subclasses."""
        return self._adaptive_direction(func, x, grad, iteration, config)


class Momentum(AdaptiveOptimizer):
    """Polyak momentum: ``v_t = beta v_{t-1} + alpha grad_t``."""

    name = "Momentum"

    def _reset_state(self) -> None:
        self._v: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._v is None:
            self._v = np.zeros_like(grad)
        self._v = config.beta * self._v + config.alpha * grad
        return -self._v, {"velocity_norm": float(np.linalg.norm(self._v))}


class Nesterov(AdaptiveOptimizer):
    """Nesterov accelerated gradient with lookahead gradient."""

    name = "Nesterov"

    def _reset_state(self) -> None:
        self._v: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._v is None:
            self._v = np.zeros_like(grad)
        lookahead = x - config.beta * self._v
        lookahead_grad = ensure_gradient(func.gradient(lookahead), dim=func.dim)
        self._v = config.beta * self._v + config.alpha * lookahead_grad
        return -self._v, {"velocity_norm": float(np.linalg.norm(self._v))}


class AdaGrad(AdaptiveOptimizer):
    """AdaGrad accumulator from Duchi et al. 2011."""

    name = "AdaGrad"

    def _reset_state(self) -> None:
        self._s: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._s is None:
            self._s = np.zeros_like(grad)
        self._s = self._s + grad * grad
        update = -config.alpha * grad / (np.sqrt(self._s) + config.epsilon)
        return update, {"accumulator_norm": float(np.linalg.norm(self._s))}


class RMSProp(AdaptiveOptimizer):
    """RMSProp exponential squared-gradient average."""

    name = "RMSProp"

    def _reset_state(self) -> None:
        self._s: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._s is None:
            self._s = np.zeros_like(grad)
        self._s = config.rho * self._s + (1.0 - config.rho) * grad * grad
        update = -config.alpha * grad / (np.sqrt(self._s) + config.epsilon)
        return update, {"rms_norm": float(np.linalg.norm(self._s))}


class AdaDelta(AdaptiveOptimizer):
    """AdaDelta update from Zeiler 2012 without a global learning rate."""

    name = "AdaDelta"

    def _reset_state(self) -> None:
        self._eg2: FloatArray | None = None
        self._edx2: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._eg2 is None or self._edx2 is None:
            self._eg2 = np.zeros_like(grad)
            self._edx2 = np.zeros_like(grad)
        self._eg2 = config.rho * self._eg2 + (1.0 - config.rho) * grad * grad
        update = -np.sqrt(self._edx2 + config.epsilon) / np.sqrt(self._eg2 + config.epsilon) * grad
        self._edx2 = config.rho * self._edx2 + (1.0 - config.rho) * update * update
        return update, {"eg2_norm": float(np.linalg.norm(self._eg2))}


class Adam(AdaptiveOptimizer):
    """Adam optimizer with Kingma-Ba 2014 bias correction."""

    name = "Adam"

    def _reset_state(self) -> None:
        self._m: FloatArray | None = None
        self._v: FloatArray | None = None

    def _adaptive_direction(
        self,
        func: MultivariateFunction,
        x: FloatArray,
        grad: FloatArray,
        iteration: int,
        config: OptimizerConfig,
    ) -> tuple[FloatArray, dict[str, float]]:
        if self._m is None or self._v is None:
            self._m = np.zeros_like(grad)
            self._v = np.zeros_like(grad)
        t = iteration + 1
        self._m = config.beta1 * self._m + (1.0 - config.beta1) * grad
        self._v = config.beta2 * self._v + (1.0 - config.beta2) * grad * grad
        m_hat = self._m / (1.0 - config.beta1**t)
        v_hat = self._v / (1.0 - config.beta2**t)
        update = -config.alpha * m_hat / (np.sqrt(v_hat) + config.epsilon)
        return update, {"m_norm": float(np.linalg.norm(self._m)), "v_norm": float(np.linalg.norm(self._v))}


for _cls in (Momentum, Nesterov, AdaGrad, RMSProp, AdaDelta, Adam):
    register_optimizer(_cls.__name__, _cls)

register_optimizer("momentum", Momentum)
register_optimizer("nesterov", Nesterov)
register_optimizer("adagrad", AdaGrad)
register_optimizer("rmsprop", RMSProp)
register_optimizer("adadelta", AdaDelta)
register_optimizer("adam", Adam)
