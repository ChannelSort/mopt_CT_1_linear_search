"""Run Lab 1 experiments."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = ROOT / "optimlib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimlib  # noqa: F401,E402
import lab1.functions  # noqa: F401,E402
from optimlib.experiment.runner import OptimizationExperiment


def main() -> None:
    """Execute Lab 1 and save aggregate tables."""
    experiment = OptimizationExperiment.from_yaml(Path(__file__).with_name("config.yaml"))
    runs = experiment.execute()
    paths = experiment.save_tables()
    print(f"Lab 1 completed: {len(runs)} runs")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
