from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.core.feasibility import is_primal_feasible
from lpas.solver.scipy_handoff import solve_with_scipy


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _feasible_grid(case, resolution: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_max, y_max = case.axis_limits[:2]
    xs = np.linspace(0.0, x_max, resolution)
    ys = np.linspace(0.0, y_max, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    feasible = np.asarray([is_primal_feasible(case.problem, point) for point in points], dtype=bool).reshape(grid_x.shape)
    return grid_x, grid_y, feasible


def _plot_constraint_lines(ax, case, active_mask: np.ndarray | None = None) -> None:
    x_max, y_max = case.axis_limits[:2]
    xs = np.linspace(0.0, x_max, 400)
    active_mask = np.zeros(case.problem.m, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    for index, (row, rhs) in enumerate(zip(case.problem.A, case.problem.b, strict=True)):
        color = "tab:red" if active_mask[index] else "0.55"
        linewidth = 2.2 if active_mask[index] else 1.0
        if abs(row[1]) < 1e-12:
            x_value = rhs / max(row[0], 1e-12)
            ax.axvline(x_value, color=color, linewidth=linewidth, alpha=0.9)
        else:
            ys = (rhs - row[0] * xs) / row[1]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def _plot_terminal_points(ax, trace, reference) -> None:
    if trace.raw_best_x is not None:
        ax.scatter(
            trace.raw_best_x[0],
            trace.raw_best_x[1],
            s=70,
            marker="o",
            facecolors="none",
            edgecolors="#0f766e",
            linewidths=1.8,
            label="raw best sample",
        )
    if trace.polished_x is not None:
        ax.scatter(
            trace.polished_x[0],
            trace.polished_x[1],
            s=78,
            marker="D",
            color="#2563eb",
            label="polished vertex",
        )
    if reference.x is not None:
        ax.scatter(reference.x[0], reference.x[1], s=90, marker="*", color="tab:red", label="HiGHS optimum")


def plot_feasible_region(case, trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    grid_x, grid_y, feasible = _feasible_grid(case)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(grid_x, grid_y, feasible.astype(float), levels=[-0.1, 0.5, 1.1], colors=["#ffffff", "#dff3e3"], alpha=0.9)
    _plot_constraint_lines(ax, case)
    if trace.captured_x is not None:
        ax.scatter(trace.captured_x[:, 0], trace.captured_x[:, 1], s=12, color="#1f78b4", alpha=0.25, label="sampled candidates")
    reference = solve_with_scipy(case.problem)
    _plot_terminal_points(ax, trace, reference)
    ax.set_title(f"{case.name}: samples, raw best, polished vertex, and HiGHS")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sampling_scores(case, trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_constraint_lines(ax, case)
    if trace.captured_x is not None and trace.captured_scores is not None:
        scatter = ax.scatter(
            trace.captured_x[:, 0],
            trace.captured_x[:, 1],
            c=trace.captured_scores,
            cmap="viridis",
            s=18,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax, label="candidate score")
    ax.set_title(f"{case.name}: sampled candidates colored by score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_elite_samples(case, trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_constraint_lines(ax, case)
    if trace.captured_x is not None and trace.captured_is_elite is not None:
        elite_points = trace.captured_x[trace.captured_is_elite]
        ax.scatter(elite_points[:, 0], elite_points[:, 1], s=26, color="#0f766e", alpha=0.75)
    ax.set_title(f"{case.name}: elite samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_active_constraints(case, trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    reference = solve_with_scipy(case.problem)
    active_mask = trace.final_active_mask if trace.final_active_mask is not None else reference.primal_active_mask
    if active_mask is None:
        active_mask = np.zeros(case.problem.m, dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_constraint_lines(ax, case, active_mask=active_mask)
    if trace.captured_x is not None and trace.captured_is_elite is not None:
        elite_points = trace.captured_x[trace.captured_is_elite]
        ax.scatter(elite_points[:, 0], elite_points[:, 1], s=22, color="#1d4ed8", alpha=0.65)
    _plot_terminal_points(ax, trace, reference)
    ax.set_title(f"{case.name}: active constraints with raw and polished solutions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_naive_vs_geometry(case, naive_trace, geometry_trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    reference = solve_with_scipy(case.problem)
    for ax, trace, title, color in [
        (axes[0], naive_trace, "Naive Monte Carlo", "#9a3412"),
        (axes[1], geometry_trace, "Geometry-aware + polishing", "#0f766e"),
    ]:
        _plot_constraint_lines(ax, case)
        if trace.captured_x is not None:
            ax.scatter(trace.captured_x[:, 0], trace.captured_x[:, 1], s=14, color=color, alpha=0.28)
        if trace.captured_x is not None and trace.captured_is_elite is not None:
            elite = trace.captured_x[trace.captured_is_elite]
            ax.scatter(elite[:, 0], elite[:, 1], s=24, color=color, alpha=0.85)
        _plot_terminal_points(ax, trace, reference)
        ax.set_title(title)
    fig.suptitle(f"{case.name}: naive vs geometry-aware sampling")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_convergence(naive_trace, geometry_trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    geometry_iterations = [entry.iteration for entry in geometry_trace.history]
    naive_iterations = [entry.iteration for entry in naive_trace.history]

    axes[0].plot(naive_iterations, [entry.best_feasible_objective for entry in naive_trace.history], label="naive", color="#9a3412")
    axes[0].plot(
        geometry_iterations,
        [entry.best_feasible_objective for entry in geometry_trace.history],
        label="geometry-aware",
        color="#0f766e",
    )
    axes[0].set_title("Best feasible objective")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("objective")
    if geometry_trace.polished_objective is not None:
        axes[0].axhline(geometry_trace.polished_objective, color="#2563eb", linestyle="--", label="geometry + polishing")
    if geometry_trace.reference_result.objective is not None:
        axes[0].axhline(geometry_trace.reference_result.objective, color="tab:red", linestyle=":", label="HiGHS")
    axes[0].legend(loc="best")

    axes[1].plot(
        naive_iterations,
        [entry.active_set_recovery_accuracy for entry in naive_trace.history],
        label="naive",
        color="#9a3412",
    )
    axes[1].plot(
        geometry_iterations,
        [entry.active_set_recovery_accuracy for entry in geometry_trace.history],
        label="geometry-aware",
        color="#0f766e",
    )
    axes[1].set_title("Active-set recovery")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("Jaccard")
    if geometry_trace.polished_active_set_similarity is not None:
        axes[1].axhline(geometry_trace.polished_active_set_similarity, color="#2563eb", linestyle="--", label="geometry + polishing")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_3d_projections(case, naive_trace, geometry_trace, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    axis_pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=False)
    for row, (trace, label, color) in enumerate(
        [
            (naive_trace, "Naive Monte Carlo", "#9a3412"),
            (geometry_trace, "Geometry-aware", "#0f766e"),
        ]
    ):
        if trace.captured_x is None:
            continue
        for col, (x_axis, y_axis) in enumerate(axis_pairs):
            ax = axes[row, col]
            ax.scatter(trace.captured_x[:, x_axis], trace.captured_x[:, y_axis], s=12, color=color, alpha=0.35)
            if trace.captured_is_elite is not None:
                elite = trace.captured_x[trace.captured_is_elite]
                ax.scatter(elite[:, x_axis], elite[:, y_axis], s=20, color=color, alpha=0.85)
            ax.set_xlabel(f"x{x_axis + 1}")
            ax.set_ylabel(f"x{y_axis + 1}")
            ax.set_xlim(0.0, case.axis_limits[x_axis])
            ax.set_ylim(0.0, case.axis_limits[y_axis])
            ax.set_title(f"{label}: (x{x_axis + 1}, x{y_axis + 1})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
