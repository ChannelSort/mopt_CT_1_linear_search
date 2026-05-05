"""Core contracts and configuration."""

from optimlib.core.base import ObjectiveFunction, OptimizationResult, Optimizer, StepState
from optimlib.core.config import ExperimentConfig, FunctionConfig, OptimizerConfig, OptimizerSpec, ParamGridConfig

__all__ = [
    "ExperimentConfig",
    "FunctionConfig",
    "ObjectiveFunction",
    "OptimizationResult",
    "Optimizer",
    "OptimizerConfig",
    "OptimizerSpec",
    "ParamGridConfig",
    "StepState",
]
