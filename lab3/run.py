"""Run Lab 3 adaptive-method experiments."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = ROOT / "optimlib" / "src"
for path in (LIB_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import optimlib  # noqa: F401,E402
import lab3.functions  # noqa: F401,E402
from optimlib.experiment.runner import OptimizationExperiment
from optimlib.visualization.plotting import plot_param_sensitivity


def main() -> None:
    """Execute Lab 3 and save sensitivity heatmaps."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    runs = experiment.execute()
    plot_dir = experiment.config.output_dir / "plots"
    for optimizer in ("Momentum", "Nesterov", "RMSProp"):
        subset = [run for run in runs if run.optimizer_name == optimizer]
        plot_param_sensitivity(subset, plot_dir, "alpha", "beta" if optimizer != "RMSProp" else "rho", basename=f"{optimizer}_heatmap")
    adam = [run for run in runs if run.optimizer_name == "Adam"]
    plot_param_sensitivity(adam, plot_dir, "beta1", "beta2", basename="Adam_heatmap")
    print(f"Lab 3 completed: {len(runs)} runs")
    for name, path in experiment.save_tables().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
