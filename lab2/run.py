"""Run Lab 2 experiments."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = ROOT / "optimlib" / "src"
for path in (LIB_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import optimlib  # noqa: F401,E402
import lab2.functions  # noqa: F401,E402
from optimlib.experiment.runner import OptimizationExperiment
from optimlib.utils.registry import GLOBAL_REGISTRY
from optimlib.visualization.plotting import plot_contours_and_trajectories, plot_convergence


def main() -> None:
    """Execute Lab 2 and save tables and plots."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    for function_config in experiment.config.normalized_functions():
        func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
        selected = [run for run in runs if run.function_name == function_config.name and run.result is not None]
        plot_contours_and_trajectories(func, selected[:8], plot_dir, f"{function_config.name}_trajectories")
        plot_convergence(selected[:8], plot_dir, f"{function_config.name}_convergence", f_min=float(func.f_min or 0.0))
    print(f"Lab 2 completed: {len(runs)} runs")
    for name, path in experiment.save_tables().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
