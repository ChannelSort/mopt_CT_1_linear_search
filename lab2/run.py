"""Run Lab 2 experiments."""

from __future__ import annotations

import json
import re
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


def _variant_basename(function_name: str, function_params: dict) -> str:
    """Filesystem-safe basename for a function instance (e.g. different x0)."""
    if not function_params:
        return function_name
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", compact)
    return f"{function_name}_{safe}"


def main() -> None:
    """Execute Lab 2 and save tables and plots."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    traj_tol = float(experiment.config.plots.get("trajectory_tolerance", 1e-8))
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    for function_config in experiment.config.normalized_functions():
        func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
        fparams = dict(function_config.params)
        selected = [
            run
            for run in runs
            if run.function_name == function_config.name
            and run.function_params == fparams
            and run.result is not None
            and abs(run.tolerance - traj_tol) <= 1e-12 * max(1.0, abs(traj_tol))
        ]
        stem = _variant_basename(function_config.name, fparams)
        plot_contours_and_trajectories(func, selected, plot_dir, f"{stem}_trajectories")
        plot_convergence(selected, plot_dir, f"{stem}_convergence", f_min=float(func.f_min or 0.0))
    print(f"Lab 2 completed: {len(runs)} runs")
    for name, path in experiment.save_tables().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
