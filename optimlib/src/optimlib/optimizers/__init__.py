"""Optimization algorithms."""

from optimlib.optimizers.univariate import BrentWrapper, Dichotomy, Fibonacci, GoldenSection, Parabola, PassiveSearch
from optimlib.optimizers.gradient import ArmijoBacktracking, ConstantStepGD, SteepestDescent, StrongWolfe
from optimlib.optimizers.adaptive import AdaDelta, AdaGrad, Adam, Momentum, Nesterov, RMSProp

__all__ = [
    "AdaDelta",
    "AdaGrad",
    "Adam",
    "ArmijoBacktracking",
    "BrentWrapper",
    "ConstantStepGD",
    "Dichotomy",
    "Fibonacci",
    "GoldenSection",
    "Momentum",
    "Nesterov",
    "Parabola",
    "PassiveSearch",
    "RMSProp",
    "SteepestDescent",
    "StrongWolfe",
]
