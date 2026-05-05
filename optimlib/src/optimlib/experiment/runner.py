"""Experiment grid runner."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from optimlib.core.base import OptimizationResult
from optimlib.core.config import ExperimentConfig, FunctionConfig, OptimizerConfig, OptimizerSpec
from optimlib.experiment.serialization import save_dataframe
from optimlib.utils.registry import GLOBAL_REGISTRY, Registry


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One experiment-combination result."""

    function_name: str
    optimizer_name: str
    tolerance: float
    params: dict[str, Any]
    result: OptimizationResult | None
    error: str | None = None


def _run_single(
    function_config: FunctionConfig,
    optimizer_spec: OptimizerSpec,
    optimizer_config: OptimizerConfig,
    params: dict[str, Any],
) -> ExperimentRun:
    """Run one isolated experiment combination."""
    try:
        func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
        func.reset_count()
        optimizer = GLOBAL_REGISTRY.get_optimizer(optimizer_spec.name, **optimizer_spec.params)
        result = optimizer.minimize(func, optimizer_config)
        return ExperimentRun(function_config.name, optimizer_spec.name, optimizer_config.tol, params, result)
    except Exception as exc:
        LOGGER.debug("Experiment combination failed.", exc_info=True)
        return ExperimentRun(function_config.name, optimizer_spec.name, optimizer_config.tol, params, None, str(exc))


class OptimizationExperiment:
    """Load config, execute a grid, and save aggregate artifacts."""

    def __init__(self, config: ExperimentConfig, registry: Registry | None = None) -> None:
        """Initialize experiment runner."""
        self.config = config
        self.registry = registry or GLOBAL_REGISTRY
        self.runs: list[ExperimentRun] = []

    @classmethod
    def from_yaml(cls, path: Path) -> "OptimizationExperiment":
        """Create experiment from YAML config."""
        return cls(ExperimentConfig.from_yaml(path))

    def execute(self) -> list[ExperimentRun]:
        """Execute function x optimizer x param_grid x tolerance."""
        tasks: list[tuple[FunctionConfig, OptimizerSpec, OptimizerConfig, dict[str, Any]]] = []
        for function_config in self.config.normalized_functions():
            for optimizer_spec in self.config.normalized_optimizers():
                for params in optimizer_spec.param_grid.combinations():
                    for tol in self.config.stopping_tolerances:
                        optimizer_config = self.config.config_for(tol, params)
                        tasks.append((function_config, optimizer_spec, optimizer_config, params))

        if self.config.parallel:
            runs: list[ExperimentRun] = []
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(_run_single, *task) for task in tasks]
                for future in as_completed(futures):
                    runs.append(future.result())
            self.runs = sorted(runs, key=lambda item: (item.function_name, item.optimizer_name, item.tolerance, str(item.params)))
        else:
            self.runs = [_run_single(*task) for task in tasks]
        self.save_tables()
        return self.runs

    def to_dataframe(self) -> pd.DataFrame:
        """Convert runs to a flat pandas DataFrame."""
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            result = run.result
            row = {
                "function": run.function_name,
                "optimizer": run.optimizer_name,
                "tolerance": run.tolerance,
                "params": run.params,
                "converged": None if result is None else result.converged,
                "x": None if result is None else (float(result.x) if isinstance(result.x, float) else result.x.tolist()),
                "f": None if result is None else result.f,
                "n_iter": None if result is None else result.n_iter,
                "n_calls": None if result is None else result.n_calls,
                "n_grad_calls": None if result is None else result.n_grad_calls,
                "message": run.error if result is None else result.message,
                "error": run.error,
            }
            row.update({f"param_{key}": value for key, value in run.params.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def save_tables(self, stem: str = "summary") -> dict[str, Path]:
        """Save aggregate CSV, JSON, and LaTeX tables."""
        return save_dataframe(self.to_dataframe(), self.config.output_dir, stem=stem)
