from __future__ import annotations

import numpy as np

import optimlib  # noqa: F401
import lab2.functions  # noqa: F401
from lab2.functions import WellConditionedQuadratic
from optimlib.core.config import OptimizerConfig
from optimlib.optimizers.gradient import ArmijoBacktracking, SteepestDescent, StrongWolfe


def test_armijo_and_wolfe_converge_quickly_on_quadratic() -> None:
    for optimizer in (ArmijoBacktracking(), StrongWolfe(), SteepestDescent()):
        func = WellConditionedQuadratic()
        result = optimizer.minimize(func, OptimizerConfig(max_iter=20, tol_grad=1e-10, alpha_init=1.0))

        assert result.converged
        assert result.n_iter < 5
        np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-8)
        assert result.n_calls > 0
        assert result.n_grad_calls > 0


def test_strong_wolfe_has_zoom_phase() -> None:
    assert hasattr(StrongWolfe(), "_zoom")
