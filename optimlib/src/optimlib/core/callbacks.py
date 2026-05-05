"""Callback implementations for optimization lifecycles."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from optimlib.core.base import OptimizationResult, StepState
from optimlib.exceptions import StopOptimization


class Callback(Protocol):
    """Optimization callback protocol."""

    def on_start(self) -> None:
        """Run before the first step."""

    def on_step(self, state: StepState) -> None:
        """Run after each accepted step."""

    def on_end(self, result: OptimizationResult) -> None:
        """Run after termination."""


class BaseCallback:
    """No-op callback base class."""

    def on_start(self) -> None:
        """Run before the first step."""

    def on_step(self, state: StepState) -> None:
        """Run after each accepted step."""

    def on_end(self, result: OptimizationResult) -> None:
        """Run after termination."""


class HistoryCallback(BaseCallback):
    """Collect immutable step states in memory."""

    def __init__(self) -> None:
        """Initialize empty history."""
        self.history: list[StepState] = []

    def on_start(self) -> None:
        """Clear history before a run."""
        self.history.clear()

    def on_step(self, state: StepState) -> None:
        """Append a state snapshot."""
        self.history.append(state)


class GradientNormCallback(BaseCallback):
    """Stop when ``||grad||`` is below a threshold."""

    def __init__(self, tolerance: float) -> None:
        """Initialize the callback."""
        self.tolerance = float(tolerance)

    def on_step(self, state: StepState) -> None:
        """Raise ``StopOptimization`` if the gradient norm is small."""
        if state.grad is None:
            return
        norm = float(np.linalg.norm(state.grad))
        if norm <= self.tolerance:
            raise StopOptimization(f"Gradient norm tolerance reached: {norm:.3e}.")


class TimerCallback(BaseCallback):
    """Measure elapsed wall-clock time and optionally stop by timeout."""

    def __init__(self, max_seconds: float | None = None) -> None:
        """Initialize timer callback."""
        self.max_seconds = max_seconds
        self.started_at: float | None = None
        self.elapsed_seconds = 0.0

    def on_start(self) -> None:
        """Start timing."""
        self.started_at = time.perf_counter()
        self.elapsed_seconds = 0.0

    def on_step(self, state: StepState) -> None:
        """Update elapsed time and stop on timeout."""
        if self.started_at is None:
            return
        self.elapsed_seconds = time.perf_counter() - self.started_at
        if self.max_seconds is not None and self.elapsed_seconds >= self.max_seconds:
            raise StopOptimization(f"Time limit reached: {self.elapsed_seconds:.3f}s.")

    def on_end(self, result: OptimizationResult) -> None:
        """Store total elapsed time."""
        if self.started_at is not None:
            self.elapsed_seconds = time.perf_counter() - self.started_at


class CheckpointCallback(BaseCallback):
    """Write step states to a JSONL checkpoint file."""

    def __init__(self, path: Path, every: int = 1) -> None:
        """Initialize checkpoint writer."""
        if every <= 0:
            raise ValueError("every must be positive.")
        self.path = path
        self.every = every

    def on_start(self) -> None:
        """Create parent directory and truncate checkpoint."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_step(self, state: StepState) -> None:
        """Append selected states as JSON lines."""
        if state.iteration % self.every != 0:
            return
        payload = asdict(state)
        payload["x"] = state.x.tolist()
        payload["grad"] = None if state.grad is None else state.grad.tolist()
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class PlottingCallback(BaseCallback):
    """Minimal real-time plotting callback for interactive runs."""

    def __init__(self, every: int = 1) -> None:
        """Initialize plotting cadence."""
        if every <= 0:
            raise ValueError("every must be positive.")
        self.every = every
        self._xs: list[int] = []
        self._fs: list[float] = []

    def on_start(self) -> None:
        """Reset data series."""
        self._xs.clear()
        self._fs.clear()

    def on_step(self, state: StepState) -> None:
        """Update a simple objective-value plot."""
        if state.iteration % self.every != 0:
            return
        self._xs.append(state.iteration)
        self._fs.append(state.f)
        plt.clf()
        plt.plot(self._xs, self._fs)
        plt.xlabel("iteration")
        plt.ylabel("f")
        plt.pause(0.001)
