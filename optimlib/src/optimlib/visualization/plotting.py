"""Plotting functions for trajectories, convergence, and sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from optimlib.core.base import FloatArray
from optimlib.experiment.runner import ExperimentRun
from optimlib.functions.base import MultivariateFunction


def _history_points(run: ExperimentRun) -> FloatArray:
    if run.result is None or not run.result.history:
        return np.empty((0, 2), dtype=np.float64)
    points = [state.x for state in run.result.history if state.x.size >= 2 and np.all(np.isfinite(state.x[:2]))]
    return np.vstack(points)[:, :2] if points else np.empty((0, 2), dtype=np.float64)


def _limits(func: MultivariateFunction, results: Iterable[ExperimentRun]) -> tuple[float, float, float, float]:
    points = [func.initial_point()]
    points.extend(func.global_minimizers)
    for run in results:
        history = _history_points(run)
        if history.size:
            points.extend(history)
        if run.result is not None and not isinstance(run.result.x, float):
            points.append(run.result.x[:2])
    stacked = np.vstack(points)
    lower = np.nanpercentile(stacked, 2.0, axis=0)
    upper = np.nanpercentile(stacked, 98.0, axis=0)
    span = np.maximum(upper - lower, 1.0)
    pad = 0.3 * span
    return float(lower[0] - pad[0]), float(upper[0] + pad[0]), float(lower[1] - pad[1]), float(upper[1] + pad[1])


def plot_contours_and_trajectories(
    func: MultivariateFunction,
    results: list[ExperimentRun],
    output_dir: Path,
    basename: str | None = None,
) -> dict[str, Path]:
    """Plot contours with clipped optimization trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = _limits(func, results)
    xs = np.linspace(x_min, x_max, 220)
    ys = np.linspace(y_min, y_max, 220)
    xx, yy = np.meshgrid(xs, ys)
    values = np.array([func(np.array([x, y], dtype=np.float64)) for x, y in zip(xx.ravel(), yy.ravel(), strict=True)]).reshape(xx.shape)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    use_log = positive.size > 0 and float(np.max(positive) / np.min(positive)) > 1e3

    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log:
        contour = ax.contourf(xx, yy, np.maximum(values, np.min(positive)), levels=40, cmap="viridis", norm=LogNorm())
    else:
        contour = ax.contourf(xx, yy, values, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax)

    for run in results:
        points = _history_points(run)
        if points.size == 0:
            continue
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        points = points[mask]
        if points.size == 0:
            continue
        ax.plot(points[:, 0], points[:, 1], marker="o", markersize=2.5, linewidth=1.0, label=f"{run.optimizer_name} {run.params}")
        if points.shape[0] > 1:
            delta = np.diff(points, axis=0)
            ax.quiver(points[:-1, 0], points[:-1, 1], delta[:, 0], delta[:, 1], angles="xy", scale_units="xy", scale=1.0, width=0.002)
    for minimizer in func.global_minimizers:
        ax.scatter(minimizer[0], minimizer[1], marker="*", s=120, c="white", edgecolors="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{func.name}: trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if results:
        ax.legend(fontsize=6)
    plt.tight_layout()
    stem = basename or f"{func.name}_trajectories"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def plot_convergence(results: list[ExperimentRun], output_dir: Path, basename: str = "convergence", f_min: float = 0.0) -> dict[str, Path]:
    """Plot ``||grad||``, ``|f-f*|``, and step sizes on log axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for run in results:
        if run.result is None or not run.result.history:
            continue
        iterations = np.array([state.iteration for state in run.result.history])
        grad_norm = np.array([np.linalg.norm(state.grad) if state.grad is not None else np.nan for state in run.result.history])
        f_error = np.array([abs(state.f - f_min) for state in run.result.history])
        steps = np.array([state.step_size if state.step_size is not None else np.nan for state in run.result.history])
        label = f"{run.optimizer_name} {run.params}"
        axes[0].semilogy(iterations, np.maximum(grad_norm, 1e-300), label=label)
        axes[1].semilogy(iterations, np.maximum(f_error, 1e-300), label=label)
        axes[2].semilogy(iterations, np.maximum(steps, 1e-300), label=label)
    axes[0].set_ylabel("||grad||")
    axes[1].set_ylabel("|f-f*|")
    axes[2].set_ylabel("alpha / step")
    axes[2].set_xlabel("iteration")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    if results:
        axes[0].legend(fontsize=6)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    pdf = output_dir / f"{basename}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def plot_param_sensitivity(
    results: list[ExperimentRun],
    output_dir: Path,
    x_param: str,
    y_param: str | None = None,
    metric: str = "n_iter",
    basename: str = "param_sensitivity",
) -> dict[str, Path]:
    """Plot line sensitivity for one parameter or heatmap for two parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in results:
        if run.result is None:
            continue
        rows.append({**run.params, metric: getattr(run.result, metric)})
    if not rows:
        return {}
    import pandas as pd

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    if y_param is None:
        sns.lineplot(data=data, x=x_param, y=metric, marker="o", ax=ax)
    else:
        pivot = data.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="magma", ax=ax)
    plt.tight_layout()
    png = output_dir / f"{basename}.png"
    pdf = output_dir / f"{basename}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return {"png": png, "pdf": pdf}
