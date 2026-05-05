from __future__ import annotations

import numpy as np

import optimlib  # noqa: F401
import lab2.functions  # noqa: F401
from lab2.functions import WellConditionedQuadratic
from optimlib.core.config import OptimizerConfig
from optimlib.optimizers.adaptive import Adam


def test_adam_converges_and_counts_gradients() -> None:
    func = WellConditionedQuadratic()
    result = Adam().minimize(func, OptimizerConfig(max_iter=500, tol_grad=1e-5, alpha=0.05))

    assert result.converged
    assert result.n_grad_calls >= result.n_iter
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-3)
