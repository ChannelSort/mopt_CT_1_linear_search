from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import optimlib  # noqa: F401,E402
import lab2.functions  # noqa: F401,E402
from lab2.functions import WellConditionedQuadratic
from optimlib.core.config import ExperimentConfig, OptimizerConfig
from optimlib.experiment.runner import OptimizationExperiment
from optimlib.optimizers.gradient import ArmijoBacktracking
from optimlib.visualization.plotting import plot_contours_and_trajectories


def test_lab2_yaml_validates() -> None:
    config = ExperimentConfig.from_yaml(Path("../lab2/config.yaml"))

    assert config.normalized_functions()[0].name == "WellConditionedQuadratic"
    assert any(spec.name == "StrongWolfe" for spec in config.normalized_optimizers())


def test_runner_isolates_bad_optimizer(tmp_path: Path) -> None:
    config = ExperimentConfig(
        functions=["WellConditionedQuadratic"],
        optimizers=["MissingOptimizer"],
        output_dir=tmp_path,
        stopping_tolerances=[1e-4],
    )
    experiment = OptimizationExperiment(config)
    runs = experiment.execute()

    assert len(runs) == 1
    assert runs[0].result is None
    assert runs[0].error is not None
    assert (tmp_path / "summary.csv").exists()


def test_registry_accepts_snake_case_aliases(tmp_path: Path) -> None:
    config = ExperimentConfig(
        functions=["well_quad"],
        optimizers=["armijo_backtracking"],
        output_dir=tmp_path,
        stopping_tolerances=[1e-4],
    )
    runs = OptimizationExperiment(config).execute()

    assert len(runs) == 1
    assert runs[0].error is None
    assert runs[0].result is not None
    assert runs[0].result.converged


def test_plotting_smoke(tmp_path: Path) -> None:
    func = WellConditionedQuadratic()
    result = ArmijoBacktracking().minimize(func, OptimizerConfig(max_iter=10, tol_grad=1e-8, alpha_init=1.0))
    from optimlib.experiment.runner import ExperimentRun

    paths = plot_contours_and_trajectories(
        WellConditionedQuadratic(),
        [ExperimentRun("WellConditionedQuadratic", "ArmijoBacktracking", 1e-8, {}, result)],
        tmp_path,
    )

    assert paths["png"].exists()
