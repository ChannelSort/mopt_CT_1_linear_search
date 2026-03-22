from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


# ==========================================
# 1. Objective models
# ==========================================


class ObjectiveFunction(ABC):
    """
    Base class for all objective functions.

    The object wraps the mathematical model and counts function evaluations
    performed by custom optimizers and SciPy optimizers alike.
    """

    def __init__(
        self,
        name: str,
        bounds: Tuple[float, float],
        is_unimodal: bool = True,
        true_min: float | None = None,
    ) -> None:
        self.name = name
        self.bounds = bounds
        self.is_unimodal = is_unimodal
        self.call_count = 0
        self._true_min = true_min

    def reset_count(self) -> None:
        """Reset the function evaluation counter before a new run."""
        self.call_count = 0

    def __call__(self, x: float) -> float:
        """Evaluate the function at one point and count the call."""
        self.call_count += 1
        return float(np.asarray(self._evaluate(np.asarray(x, dtype=float))).item())

    def evaluate_many(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation for methods such as passive search.

        TODO: keep using vectorized calls in future optimizers whenever possible.
        """
        x_array = np.asarray(x, dtype=float)
        self.call_count += int(x_array.size)
        return np.asarray(self._evaluate(x_array), dtype=float)

    def get_bounds(self) -> Tuple[float, float]:
        return self.bounds

    def get_true_min(self) -> float | None:
        return self._true_min

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """Return the mathematical value of the function."""


class GoodUnimodalFunction(ObjectiveFunction):
    """A smooth quadratic function with a clear minimum."""

    def __init__(self) -> None:
        super().__init__(
            name="Good unimodal quadratic",
            bounds=(-6.0, 8.0),
            is_unimodal=True,
            true_min=2.0,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return (x - 2.0) ** 2 + 1.0


class PlateauAsymmetricUnimodalFunction(ObjectiveFunction):
    """
    A unimodal function with a flat minimum area and asymmetric growth.

    The minimum is reached on a short plateau around x = 1.5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Plateau asymmetric unimodal",
            bounds=(-2.0, 6.0),
            is_unimodal=True,
            true_min=1.5,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        center = 1.5
        half_plateau = 0.2
        distance = np.abs(x - center)

        left_branch = 2.0 * np.maximum(0.0, center - x - half_plateau) ** 2
        right_branch = 0.2 * np.maximum(0.0, x - center - half_plateau) ** 4
        return np.where(distance <= half_plateau, 0.0, left_branch + right_branch + 0.05)


class SecondSpecialUnimodalFunction(ObjectiveFunction):
    """
    TODO: implement the second custom unimodal function for the final report.

    Suggested directions:
    - strong asymmetry without a plateau
    - sharp minimum next to a nearly flat region
    """

    def __init__(self) -> None:
        super().__init__(
            name="TODO second special unimodal",
            bounds=(-1.0, 1.0),
            is_unimodal=True,
            true_min=None,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: implement the second special unimodal function.")


class MultimodalFunction(ObjectiveFunction):
    """
    TODO: implement a multimodal test function for the exploration experiment.
    """

    def __init__(self) -> None:
        super().__init__(
            name="TODO multimodal function",
            bounds=(-5.0, 5.0),
            is_unimodal=False,
            true_min=None,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: implement the multimodal function.")


# ==========================================
# 2. Optimization result model
# ==========================================


@dataclass
class OptimizationResult:
    x_min: float
    f_min: float
    n_iterations: int
    n_calls: int
    history: List[Tuple[float, float]] = field(default_factory=list)
    converged: bool = True
    message: str = ""
    history_available: bool = True


# ==========================================
# 3. Optimizers
# ==========================================


class Optimizer(ABC):
    """Strategy interface for one-dimensional optimization methods."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        """Find a function minimum for the requested precision."""


class PassiveSearchOptimizer(Optimizer):
    """
    Passive search over a uniform grid.

    According to the lecture notes, we split [a, b] into n parts with step
    h = (b - a) / n and use the estimate x* in [x_k - h, x_k + h]. Therefore,
    the localization error is controlled by eps ~= 2h, so we choose n from
    2h <= eps.
    """

    def __init__(self, max_evaluations: int = 200_000) -> None:
        super().__init__("Passive search")
        self.max_evaluations = max_evaluations

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        a, b = func.get_bounds()
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")

        interval_length = b - a
        if interval_length <= 0:
            raise ValueError("Function bounds must satisfy a < b.")

        # From the notes: eps = 2h, h = (b - a) / n.
        # Hence n >= 2 * (b - a) / eps.
        n_segments = max(2, int(np.ceil(2.0 * interval_length / eps)))
        step = interval_length / n_segments
        sample_count = n_segments + 1

        if sample_count > self.max_evaluations:
            raise ValueError(
                "Passive search requires too many evaluations for this epsilon. "
                "TODO: narrow the interval or configure a higher evaluation limit."
            )

        x_grid = np.linspace(a, b, sample_count)
        y_grid = func.evaluate_many(x_grid)

        best_index = 0
        best_x = float(x_grid[best_index])
        history: List[Tuple[float, float]] = [
            (max(a, best_x - step), min(b, best_x + step))
        ]

        for current_index in range(1, sample_count):
            if y_grid[current_index] < y_grid[best_index]:
                best_index = current_index

            best_x = float(x_grid[best_index])
            history.append((max(a, best_x - step), min(b, best_x + step)))

        x_min = float(x_grid[best_index])
        f_min = float(y_grid[best_index])

        return OptimizationResult(
            x_min=x_min,
            f_min=f_min,
            n_iterations=n_segments,
            n_calls=func.call_count,
            history=history,
            converged=True,
            message=(
                "Grid scan finished successfully. "
                f"Step h = {step:.3e}, estimated localization width <= {2.0 * step:.3e}."
            ),
            history_available=True,
        )


class BrentOptimizer(Optimizer):
    """
    SciPy Brent optimizer wrapper.

    NOTE: SciPy does not expose the interval history, so the interval-dynamics
    plot for this method remains a placeholder for now. The `brent` method uses
    the interval only as an initial bracket, not as strict bounds.
    """

    def __init__(self, max_iterations: int = 500) -> None:
        super().__init__("Brent (SciPy)")
        self.max_iterations = max_iterations

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        a, b = func.get_bounds()
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")
        if a >= b:
            raise ValueError("Function bounds must satisfy a < b.")

        initial_bracket = (a, b)
        result = minimize_scalar(
            func,
            bracket=initial_bracket,
            method="brent",
            options={"xtol": eps, "maxiter": self.max_iterations},
        )

        message = str(result.message)
        if not (a <= float(result.x) <= b):
            message += (
                " Result lies outside the original [a, b] interval because "
                "SciPy Brent treats the provided segment only as an initial bracket."
            )

        return OptimizationResult(
            x_min=float(result.x),
            f_min=float(result.fun),
            n_iterations=int(result.nit),
            n_calls=func.call_count,
            history=[],
            converged=bool(result.success),
            message=message,
            history_available=False,
        )


class DichotomyOptimizer(Optimizer):
    """TODO: implement dichotomy search."""

    def __init__(self) -> None:
        super().__init__("Dichotomy")

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        raise NotImplementedError("TODO: implement dichotomy search.")


class GoldenSectionOptimizer(Optimizer):
    """TODO: implement golden-section search."""

    def __init__(self) -> None:
        super().__init__("Golden section")

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        raise NotImplementedError("TODO: implement golden-section search.")


class FibonacciOptimizer(Optimizer):
    """TODO: implement Fibonacci search."""

    def __init__(self) -> None:
        super().__init__("Fibonacci")

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        raise NotImplementedError("TODO: implement Fibonacci search.")


class ParabolaOptimizer(Optimizer):
    """TODO: implement parabola-based search."""

    def __init__(self) -> None:
        super().__init__("Parabola")

    def minimize(self, func: ObjectiveFunction, eps: float) -> OptimizationResult:
        raise NotImplementedError("TODO: implement parabola search.")


# ==========================================
# 4. Experiment manager
# ==========================================


class OptimizationExperiment:
    """
    Runs all function/method/epsilon combinations and stores structured results.
    """

    def __init__(
        self,
        optimizers: List[Optimizer],
        functions: List[ObjectiveFunction],
        output_dir: str | Path = "outputs",
    ) -> None:
        self.optimizers = optimizers
        self.functions = functions
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.results_data: List[dict] = []

    def run(self, epsilons: List[float]) -> None:
        """Run all experiment combinations and store both successes and TODO rows."""
        self.results_data = []

        for func in self.functions:
            for opt in self.optimizers:
                for eps in epsilons:
                    func.reset_count()

                    try:
                        result = opt.minimize(func, eps)
                        self.results_data.append(
                            {
                                "function": func.name,
                                "is_unimodal": func.is_unimodal,
                                "optimizer": opt.name,
                                "epsilon": float(eps),
                                "iterations": result.n_iterations,
                                "calls": result.n_calls,
                                "x_min": result.x_min,
                                "f_min": result.f_min,
                                "history": result.history,
                                "history_available": result.history_available,
                                "converged": result.converged,
                                "status": "success" if result.converged else "warning",
                                "message": result.message,
                            }
                        )
                    except NotImplementedError as error:
                        self.results_data.append(
                            {
                                "function": func.name,
                                "is_unimodal": func.is_unimodal,
                                "optimizer": opt.name,
                                "epsilon": float(eps),
                                "iterations": np.nan,
                                "calls": np.nan,
                                "x_min": np.nan,
                                "f_min": np.nan,
                                "history": [],
                                "history_available": False,
                                "converged": False,
                                "status": "todo",
                                "message": str(error),
                            }
                        )
                    except Exception as error:
                        self.results_data.append(
                            {
                                "function": func.name,
                                "is_unimodal": func.is_unimodal,
                                "optimizer": opt.name,
                                "epsilon": float(eps),
                                "iterations": np.nan,
                                "calls": np.nan,
                                "x_min": np.nan,
                                "f_min": np.nan,
                                "history": [],
                                "history_available": False,
                                "converged": False,
                                "status": "error",
                                "message": str(error),
                            }
                        )

    def results_dataframe(self) -> pd.DataFrame:
        """Return all recorded runs as a DataFrame."""
        if not self.results_data:
            return pd.DataFrame()
        return pd.DataFrame(self.results_data)

    def build_summary_table(self, include_failed: bool = True) -> pd.DataFrame:
        """
        Build a clean, numbered table for console output and export.

        TODO: adapt the final table layout once the team agrees on the report style.
        """
        df = self.results_dataframe()
        if df.empty:
            return pd.DataFrame()

        if not include_failed:
            df = df[df["status"] == "success"].copy()

        df = df.sort_values(by=["function", "optimizer", "epsilon"], ascending=[True, True, False]).reset_index(drop=True)
        table = pd.DataFrame(
            {
                "No.": np.arange(1, len(df) + 1),
                "Function": df["function"],
                "Optimizer": df["optimizer"],
                "Epsilon": df["epsilon"],
                "Iterations": df["iterations"],
                "Function calls": df["calls"],
                "x_min": df["x_min"],
                "f(x_min)": df["f_min"],
                "Status": df["status"],
                "Comment": df["message"],
            }
        )
        return table

    def print_summary_table(self, include_failed: bool = True) -> None:
        """Print the summary table with numbered rows and clear column names."""
        table = self.build_summary_table(include_failed=include_failed)
        if table.empty:
            print("No experiment data available.")
            return
        print(table.to_string(index=False))

    def save_summary_table(self, filename: str = "experiment_summary.csv", include_failed: bool = True) -> Path:
        """Save the summary table to CSV for future report integration."""
        table = self.build_summary_table(include_failed=include_failed)
        output_path = self.tables_dir / filename
        table.to_csv(output_path, index=False)
        return output_path

    def plot_metrics(self, only_unimodal: bool = True, show: bool = False) -> List[Path]:
        """
        Plot epsilon vs iterations and epsilon vs function calls for each function.
        """
        df = self.results_dataframe()
        if df.empty:
            return []

        df = df[df["status"] == "success"].copy()
        if only_unimodal:
            df = df[df["is_unimodal"]].copy()

        saved_paths: List[Path] = []
        for function_name in df["function"].unique():
            df_func = df[df["function"] == function_name].sort_values("epsilon")
            if df_func.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for optimizer_name in df_func["optimizer"].unique():
                df_opt = df_func[df_func["optimizer"] == optimizer_name].sort_values("epsilon")
                axes[0].plot(
                    df_opt["epsilon"],
                    df_opt["iterations"],
                    marker="o",
                    linewidth=2,
                    label=optimizer_name,
                )
                axes[1].plot(
                    df_opt["epsilon"],
                    df_opt["calls"],
                    marker="o",
                    linewidth=2,
                    label=optimizer_name,
                )

            axes[0].set_xscale("log")
            axes[1].set_xscale("log")
            axes[0].set_xlabel("Epsilon")
            axes[1].set_xlabel("Epsilon")
            axes[0].set_ylabel("Iterations")
            axes[1].set_ylabel("Function evaluations")
            axes[0].set_title(f"Iterations vs epsilon: {function_name}")
            axes[1].set_title(f"Function evaluations vs epsilon: {function_name}")

            for ax in axes:
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.legend()
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.axvline(df_func["epsilon"].min(), color="black", linewidth=0.0)

            fig.suptitle(f"Efficiency comparison for {function_name}", fontsize=13)
            fig.tight_layout()

            output_path = self.plots_dir / f"{self._slugify(function_name)}_metrics.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            saved_paths.append(output_path)

            if show:
                plt.show()
            plt.close(fig)

        return saved_paths

    def plot_interval_dynamics(
        self,
        function_name: str,
        optimizer_name: str,
        eps: float,
        show: bool = False,
    ) -> Path:
        """
        Plot the lower and upper bounds of the uncertainty interval by iteration.

        If the selected method does not expose interval history yet, the function
        still generates a placeholder plot with a TODO note.
        """
        df = self.results_dataframe()
        if df.empty:
            raise ValueError("No experiment data available.")

        mask = (
            (df["function"] == function_name)
            & (df["optimizer"] == optimizer_name)
            & np.isclose(df["epsilon"], eps)
        )
        row = df[mask]
        if row.empty:
            raise ValueError("Requested experiment combination was not found.")

        record = row.iloc[0]
        fig, ax = plt.subplots(figsize=(10, 6))

        if bool(record["history_available"]) and len(record["history"]) >= 2:
            history = record["history"]
            iterations = np.arange(len(history))
            lower_bounds = [interval[0] for interval in history]
            upper_bounds = [interval[1] for interval in history]

            ax.plot(iterations, lower_bounds, marker="o", label="Lower bound a")
            ax.plot(iterations, upper_bounds, marker="o", label="Upper bound b")
            ax.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.25, label="Uncertainty interval")
        else:
            ax.text(
                0.5,
                0.5,
                "TODO: interval history is not available for this method yet.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Interval boundary value")
        ax.set_title(f"Interval dynamics: {optimizer_name}, {function_name}, eps={eps:g}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(loc="best") if ax.lines else None

        output_path = self.plots_dir / (
            f"{self._slugify(function_name)}_{self._slugify(optimizer_name)}_eps_{self._slugify(str(eps))}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)
        return output_path

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_")


# ==========================================
# 5. Demo entry point
# ==========================================


def build_demo_experiment() -> OptimizationExperiment:
    """
    Assemble the current working skeleton used by the team right now.

    TODO: expand this list when other optimizers and functions are implemented.
    """
    functions: List[ObjectiveFunction] = [
        GoodUnimodalFunction(),
        PlateauAsymmetricUnimodalFunction(),
        # TODO: add SecondSpecialUnimodalFunction() when implemented.
        # TODO: add MultimodalFunction() for the exploration experiment.
    ]

    optimizers: List[Optimizer] = [
        PassiveSearchOptimizer(),
        BrentOptimizer(),
        # TODO: add DichotomyOptimizer().
        # TODO: add GoldenSectionOptimizer().
        # TODO: add FibonacciOptimizer().
        # TODO: add ParabolaOptimizer().
    ]

    return OptimizationExperiment(optimizers=optimizers, functions=functions)


if __name__ == "__main__":
    experiment = build_demo_experiment()

    # NOTE: passive search is expensive for very small eps on wide intervals.
    # TODO: revisit the final epsilon grid when the full project configuration is fixed.
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

    experiment.run(epsilons)
    experiment.print_summary_table(include_failed=True)

    summary_path = experiment.save_summary_table()
    plot_paths = experiment.plot_metrics(only_unimodal=True, show=False)

    # Build one interval plot per successful run as a ready-to-use example.
    df = experiment.results_dataframe()
    for _, record in df[df["status"] == "success"].iterrows():
        experiment.plot_interval_dynamics(
            function_name=record["function"],
            optimizer_name=record["optimizer"],
            eps=float(record["epsilon"]),
            show=False,
        )

    print(f"\nSummary table saved to: {summary_path}")
    print("Metric plots:")
    for path in plot_paths:
        print(f" - {path}")
