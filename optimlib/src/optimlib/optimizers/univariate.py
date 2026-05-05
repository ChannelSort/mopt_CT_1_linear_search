"""Univariate interval optimization methods for Lab 1."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from optimlib.core.base import ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import Callback, HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import StopOptimization
from optimlib.functions.base import UnivariateFunction
from optimlib.utils.registry import register_optimizer


class IntervalOptimizer(ABC):
    """Base class for interval-based scalar optimizers."""

    name = "IntervalOptimizer"

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Initialize callback list."""
        self.callbacks = list(callbacks or [])

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        """Minimize a univariate function on ``config.interval``."""
        if not isinstance(func, UnivariateFunction):
            raise TypeError("IntervalOptimizer requires a UnivariateFunction.")
        interval = config.interval or func.interval
        if interval[0] >= interval[1]:
            raise ValueError("Invalid interval: left endpoint must be smaller than right.")

        history = HistoryCallback()
        callbacks: list[Callback] = [history, *self.callbacks]
        for callback in callbacks:
            callback.on_start()

        try:
            result = self._minimize_interval(func, config, interval, callbacks, history)
        except StopOptimization as exc:
            best = self._best_from_history(history.history)
            result = OptimizationResult(
                x=best[0],
                f=best[1],
                n_iter=len(history.history),
                n_calls=func.call_count,
                n_grad_calls=func.grad_count,
                converged=True,
                message=exc.message,
                history=history.history,
                metadata={"optimizer": self.name},
            )

        for callback in callbacks:
            callback.on_end(result)
        return result

    @abstractmethod
    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        """Run method-specific interval minimization."""

    def _emit(self, callbacks: Sequence[Callback], state: StepState) -> None:
        for callback in callbacks:
            callback.on_step(state)

    def _result(
        self,
        func: UnivariateFunction,
        x: float,
        f: float,
        converged: bool,
        message: str,
        history: HistoryCallback,
    ) -> OptimizationResult:
        return OptimizationResult(
            x=x,
            f=f,
            n_iter=len(history.history),
            n_calls=func.call_count,
            n_grad_calls=func.grad_count,
            converged=converged,
            message=message,
            history=history.history,
            metadata={"optimizer": self.name},
        )

    def _emit_interval(
        self,
        callbacks: Sequence[Callback],
        iteration: int,
        x: float,
        f: float,
        a: float,
        b: float,
    ) -> None:
        self._emit(
            callbacks,
            StepState(
                iteration=iteration,
                x=np.array([x], dtype=np.float64),
                f=f,
                step_size=b - a,
                extra_metrics={"interval_a": a, "interval_b": b},
            ),
        )

    @staticmethod
    def _best_from_history(history: Sequence[StepState]) -> tuple[float, float]:
        if not history:
            return math.nan, math.inf
        best = min(history, key=lambda state: state.f)
        return float(best.x[0]), best.f


class PassiveSearch(IntervalOptimizer):
    """Passive grid search with vectorized batch evaluation."""

    name = "PassiveSearch"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        a, b = interval
        n_segments = max(2, int(math.ceil(2.0 * (b - a) / config.tol)))
        max_evaluations = int(getattr(config, "max_evaluations", 1_000_000))
        n_points = n_segments + 1
        capped = n_points > max_evaluations
        if capped:
            n_segments = max_evaluations - 1
            n_points = max_evaluations
        grid = np.linspace(a, b, n_points, dtype=np.float64)
        values = func.evaluate_many(grid)
        best_idx = int(np.argmin(values))
        step = (b - a) / max(n_segments, 1)
        stride = max(1, n_points // 500)
        sampled = np.arange(0, n_points, stride, dtype=int)
        if int(sampled[-1]) != n_points - 1:
            sampled = np.append(sampled, n_points - 1)
        for iteration, idx in enumerate(sampled):
            running_best_idx = int(np.argmin(values[: int(idx) + 1]))
            x_best = float(grid[running_best_idx])
            f_best = float(values[running_best_idx])
            lower = float(np.clip(x_best - step, a, b))
            upper = float(np.clip(x_best + step, a, b))
            self._emit_interval(callbacks, iteration, x_best, f_best, lower, upper)
        message = f"Grid scan finished. h = {step:.3e}, localization width <= {2.0 * step:.3e}."
        if capped:
            message = (
                f"Limited to {max_evaluations} evaluations; actual localization width = {2.0 * step:.3e} "
                f"> requested {config.tol:.3e}. "
                + message
            )
        result = self._result(func, float(grid[best_idx]), float(values[best_idx]), not capped, message, history)
        return OptimizationResult(
            x=result.x,
            f=result.f,
            n_iter=n_segments,
            n_calls=result.n_calls,
            n_grad_calls=result.n_grad_calls,
            converged=result.converged,
            message=result.message,
            history=result.history,
            metadata=result.metadata,
        )


class Dichotomy(IntervalOptimizer):
    """Dichotomy search for unimodal scalar objectives."""

    name = "Dichotomy"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        a, b = interval
        delta = config.tol / 10.0
        iteration = 0
        x_best = 0.5 * (a + b)
        f_best = math.inf
        while (b - a) > config.tol and iteration < config.max_iter:
            mid = 0.5 * (a + b)
            x1 = mid - delta
            x2 = mid + delta
            f1 = func(x1)
            f2 = func(x2)
            if f1 < f2:
                b = x2
                x_best, f_best = x1, f1
            else:
                a = x1
                x_best, f_best = x2, f2
            self._emit_interval(callbacks, iteration, x_best, f_best, a, b)
            iteration += 1
        return self._result(func, x_best, f_best, (b - a) <= config.tol, "Interval tolerance reached.", history)


class GoldenSection(IntervalOptimizer):
    """Golden-section search for unimodal scalar objectives."""

    name = "GoldenSection"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        a, b = interval
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc = func(c)
        fd = func(d)
        iteration = 0
        x_best = c if fc < fd else d
        f_best = min(fc, fd)
        while (b - a) > config.tol and iteration < config.max_iter:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = func(c)
                x_best, f_best = c, fc
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = func(d)
                x_best, f_best = d, fd
            if fc < fd:
                x_best, f_best = c, fc
            else:
                x_best, f_best = d, fd
            self._emit_interval(callbacks, iteration, x_best, f_best, a, b)
            iteration += 1
        return self._result(func, x_best, f_best, (b - a) <= config.tol, "Interval tolerance reached.", history)


class Fibonacci(IntervalOptimizer):
    """Fibonacci interval search for unimodal scalar objectives."""

    name = "Fibonacci"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        a, b = interval
        fib = [1, 1]
        while fib[-1] < (b - a) / config.tol:
            fib.append(fib[-1] + fib[-2])
        n = len(fib) - 1
        c = a + fib[n - 2] / fib[n] * (b - a)
        d = a + fib[n - 1] / fib[n] * (b - a)
        fc = func(c)
        fd = func(d)
        x_best = c if fc < fd else d
        f_best = min(fc, fd)
        for iteration in range(1, min(n - 1, config.max_iter) + 1):
            last_step = iteration == n - 1
            if fc < fd:
                b, d, fd = d, c, fc
                if last_step:
                    c = 0.5 * (a + b) - config.tol / 10.0
                else:
                    c = a + fib[n - iteration - 2] / fib[n - iteration] * (b - a)
                fc = func(c)
            else:
                a, c, fc = c, d, fd
                if last_step:
                    d = 0.5 * (a + b) + config.tol / 10.0
                else:
                    d = a + fib[n - iteration - 1] / fib[n - iteration] * (b - a)
                fd = func(d)
            if fc < fd:
                x_best, f_best = c, fc
            else:
                x_best, f_best = d, fd
            self._emit_interval(callbacks, iteration - 1, x_best, f_best, a, b)
            if (b - a) <= config.tol:
                break
        return self._result(func, x_best, f_best, (b - a) <= config.tol, "Interval tolerance reached.", history)


class Parabola(IntervalOptimizer):
    """Parabolic interpolation matching the original Lab 1 stopping behavior."""

    name = "Parabola"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        a, c = interval
        b = 0.5 * (a + c)
        fa, fb, fc = func(a), func(b), func(c)
        for iteration in range(config.max_iter):
            numerator = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
            denominator = 2.0 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))
            if abs(denominator) < 1e-12:
                self._emit_interval(callbacks, iteration, b, fb, b - 1e-15, b + 1e-15)
                return self._result(func, b, fb, True, "Parabolic denominator degenerated.", history)
            u = b - numerator / denominator
            if not np.isfinite(u) or not a < u < c:
                self._emit_interval(callbacks, iteration, b, fb, a, c)
                return self._result(func, b, fb, False, "Parabolic interpolation left the interval.", history)
            fu = func(u)
            if u < b:
                if fu < fb:
                    c, fc = b, fb
                    b, fb = u, fu
                else:
                    a, fa = u, fu
            else:
                if fu < fb:
                    a, fa = b, fb
                    b, fb = u, fu
                else:
                    c, fc = u, fu
            self._emit_interval(callbacks, iteration, b, fb, a, c)
            if (c - a) <= config.tol:
                break
        return self._result(func, b, fb, (c - a) <= config.tol, "Parabolic interpolation finished.", history)


class BrentWrapper(IntervalOptimizer):
    """SciPy Brent minimizer using the interval as an initial bracket."""

    name = "BrentWrapper"

    def _minimize_interval(
        self,
        func: UnivariateFunction,
        config: OptimizerConfig,
        interval: tuple[float, float],
        callbacks: Sequence[Callback],
        history: HistoryCallback,
    ) -> OptimizationResult:
        result = minimize_scalar(func, bracket=interval, method="brent", options={"xtol": config.tol, "maxiter": config.max_iter})
        state = StepState(
            iteration=int(result.nit),
            x=np.array([float(result.x)], dtype=np.float64),
            f=float(result.fun),
            step_size=config.tol,
            extra_metrics={"interval_a": interval[0], "interval_b": interval[1]},
        )
        self._emit(callbacks, state)
        base = self._result(func, float(result.x), float(result.fun), bool(result.success), str(result.message), history)
        return OptimizationResult(
            x=base.x,
            f=base.f,
            n_iter=int(result.nit),
            n_calls=base.n_calls,
            n_grad_calls=base.n_grad_calls,
            converged=base.converged,
            message=base.message,
            history=base.history,
            metadata=base.metadata,
        )


for _cls in (PassiveSearch, Dichotomy, GoldenSection, Fibonacci, Parabola, BrentWrapper):
    register_optimizer(_cls.__name__, _cls)

register_optimizer("passive_search", PassiveSearch)
register_optimizer("dichotomy", Dichotomy)
register_optimizer("golden_section", GoldenSection)
register_optimizer("fibonacci", Fibonacci)
register_optimizer("parabola", Parabola)
register_optimizer("brent", BrentWrapper)
register_optimizer("brent_wrapper", BrentWrapper)
