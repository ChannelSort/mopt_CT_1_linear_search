"""Run Lab 3 adaptive-method experiments."""

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
import lab3.functions  # noqa: F401,E402
from optimlib.experiment.runner import ExperimentRun, OptimizationExperiment
from optimlib.utils.registry import GLOBAL_REGISTRY
from optimlib.visualization.plotting import plot_contours_and_trajectories, plot_param_sensitivity


def _variant_basename(function_name: str, function_params: dict) -> str:
    if not function_params:
        return function_name
    compact = json.dumps(function_params, sort_keys=True, separators=(",", ":"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", compact)
    return f"{function_name}_{safe}"


def _best_runs_per_optimizer(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """Pick one run per optimizer (fewest iterations among converged, else best available)."""
    converged = [r for r in runs if r.result is not None and r.result.converged]
    pool = converged if converged else [r for r in runs if r.result is not None]
    by_opt: dict[str, ExperimentRun] = {}
    for r in pool:
        assert r.result is not None
        prev = by_opt.get(r.optimizer_name)
        if prev is None or r.result.n_iter < prev.result.n_iter:
            by_opt[r.optimizer_name] = r
    return list(by_opt.values())


def main() -> None:
    """Execute Lab 3, save sensitivity plots and best-run trajectories per objective."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    traj_tol = float(experiment.config.plots.get("trajectory_tolerance", 1e-8))
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    tol_scale = 1e-12 * max(1.0, abs(traj_tol))

    for function_config in experiment.config.normalized_functions():
        func = GLOBAL_REGISTRY.get_function(function_config.name, **function_config.params)
        fparams = dict(function_config.params)
        stem = _variant_basename(function_config.name, fparams)
        family = [run for run in runs if run.function_name == function_config.name and run.function_params == fparams]

        for optimizer in ("Momentum", "Nesterov", "RMSProp"):
            subset = [run for run in family if run.optimizer_name == optimizer]
            if subset:
                y_param = "rho" if optimizer == "RMSProp" else "beta"
                plot_param_sensitivity(subset, plot_dir, "alpha", y_param, basename=f"{stem}_{optimizer}_heatmap")

        ada_grad = [run for run in family if run.optimizer_name == "AdaGrad"]
        if ada_grad:
            plot_param_sensitivity(ada_grad, plot_dir, "alpha", None, basename=f"{stem}_AdaGrad_sensitivity")

        ada_delta = [run for run in family if run.optimizer_name == "AdaDelta"]
        if ada_delta:
            plot_param_sensitivity(ada_delta, plot_dir, "rho", None, basename=f"{stem}_AdaDelta_sensitivity")

        adam = [run for run in family if run.optimizer_name == "Adam"]
        if adam:
            plot_param_sensitivity(adam, plot_dir, "beta1", "beta2", basename=f"{stem}_Adam_heatmap")

        traj_runs = [
            run
            for run in family
            if run.result is not None
            and abs(run.tolerance - traj_tol) <= tol_scale
        ]
        best = _best_runs_per_optimizer(traj_runs)
        plot_contours_and_trajectories(func, best, plot_dir, f"{stem}_best_trajectories")

    print(f"Lab 3 completed: {len(runs)} runs")
    for name, path in experiment.save_tables().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
