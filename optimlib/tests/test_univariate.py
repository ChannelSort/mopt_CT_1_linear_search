from __future__ import annotations

import numpy as np

import optimlib  # noqa: F401
import lab1.functions  # noqa: F401
from lab1.functions import (
    GoodUnimodalFunction,
    MultimodalRastrigin,
    PlateauAsymmetricUnimodalFunction,
    SecondSpecialUnimodalFunction,
)
from optimlib.core.config import OptimizerConfig
from optimlib.optimizers.univariate import GoldenSection, PassiveSearch


def test_lab1_functions_match_original_formulas() -> None:
    good = GoodUnimodalFunction()
    plateau = PlateauAsymmetricUnimodalFunction()
    second = SecondSpecialUnimodalFunction()
    rastrigin = MultimodalRastrigin()

    assert good.interval == (-6.0, 8.0)
    assert good(2.0) == 1.0
    assert plateau.interval == (-2.0, 6.0)
    assert plateau(1.0) == 0.0
    assert second.interval == (-1.0, 2.0)
    assert np.isclose(second(0.5), np.exp(-0.5))
    assert rastrigin.interval == (-5.12, 5.12)
    assert np.isclose(rastrigin(0.0), 0.0)


def test_golden_section_converges_on_good_unimodal() -> None:
    func = GoodUnimodalFunction()
    result = GoldenSection().minimize(func, OptimizerConfig(interval=func.interval, tol=1e-6, max_iter=200))

    assert result.converged
    assert abs(float(result.x) - 2.0) < 1e-4
    assert abs(result.f - 1.0) < 1e-8
    assert result.n_calls > 0


def test_passive_search_uses_evaluate_many_counter() -> None:
    func = GoodUnimodalFunction()
    result = PassiveSearch().minimize(
        func,
        OptimizerConfig(interval=func.interval, tol=1e-3, max_iter=101, max_evaluations=101),
    )

    assert result.n_calls == 101
    assert np.isfinite(result.f)
