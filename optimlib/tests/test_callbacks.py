from __future__ import annotations

import numpy as np

import lab2.functions  # noqa: F401
import optimlib  # noqa: F401
from lab2.functions import WellConditionedQuadratic
from optimlib.core.base import OptimizationResult, StepState
from optimlib.core.config import OptimizerConfig
from optimlib.exceptions import StopOptimization
from optimlib.optimizers.adaptive import Adam
from optimlib.optimizers.gradient import ConstantStepGD


class StopAfterFirstStep:
    def on_start(self) -> None:
        pass

    def on_step(self, state: StepState) -> None:
        raise StopOptimization("test stop")

    def on_end(self, result: OptimizationResult) -> None:
        pass


def test_gradient_stop_callback_returns_last_accepted_step() -> None:
    func = WellConditionedQuadratic()
    result = ConstantStepGD(callbacks=[StopAfterFirstStep()]).minimize(
        func,
        OptimizerConfig(max_iter=10, alpha=0.1),
    )

    assert result.n_iter == 1
    assert result.history
    np.testing.assert_allclose(result.x, result.history[-1].x)
    assert result.message == "test stop"


def test_adaptive_stop_callback_returns_last_accepted_step() -> None:
    func = WellConditionedQuadratic()
    result = Adam(callbacks=[StopAfterFirstStep()]).minimize(
        func,
        OptimizerConfig(max_iter=10, alpha=0.05),
    )

    assert result.n_iter == 1
    assert result.history
    np.testing.assert_allclose(result.x, result.history[-1].x)
    assert result.message == "test stop"
