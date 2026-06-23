"""Render ARC grids to images for the paper's figures and media gallery.

Uses the canonical ARC 10-colour palette. matplotlib is an optional dependency
(extras ``viz``); importing this module without it raises a clear error only
when a render function is actually called.
"""

from __future__ import annotations

from typing import Sequence

from arc_solver.io.task import Grid, Task

# Canonical ARC colour palette, indexed 0-9.
ARC_PALETTE = [
    "#101010",  # 0 black
    "#1E93FF",  # 1 blue
    "#F93C31",  # 2 red
    "#4FCC30",  # 3 green
    "#FFDC00",  # 4 yellow
    "#999999",  # 5 grey
    "#E53AA3",  # 6 magenta
    "#FF851B",  # 7 orange
    "#87D8F1",  # 8 cyan
    "#921231",  # 9 maroon
]


def _require_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        return plt, ListedColormap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for rendering; install extras: pip install '.[viz]'"
        ) from exc


def _draw(ax, grid: Grid, cmap, title: str | None = None) -> None:
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = len(grid), len(grid[0])
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#303030", linewidth=0.5)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#303030", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=9)


def grid_to_png(grid: Grid, path: str, *, cell: float = 0.3) -> None:
    """Render a single grid to a PNG file."""
    plt, ListedColormap = _require_mpl()
    cmap = ListedColormap(ARC_PALETTE)
    h, w = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(max(w * cell, 1), max(h * cell, 1)))
    _draw(ax, grid, cmap)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_task_figure(
    task: Task, path: str, *, predictions: Sequence[Grid] | None = None
) -> None:
    """Render a task's train pairs and test input(s), optionally with predictions.

    Layout: one column per train pair (input over output), then the test
    input(s) and any model predictions.
    """
    plt, ListedColormap = _require_mpl()
    cmap = ListedColormap(ARC_PALETTE)

    n_train = len(task.train)
    n_test = len(task.test)
    n_cols = n_train + n_test
    n_rows = 3 if predictions is not None else 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows), squeeze=False
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for j, pair in enumerate(task.train):
        _draw(axes[0][j], pair.input, cmap, f"train {j + 1} in")
        if pair.output is not None:
            _draw(axes[1][j], pair.output, cmap, f"train {j + 1} out")
    for k, pair in enumerate(task.test):
        col = n_train + k
        _draw(axes[0][col], pair.input, cmap, f"test {k + 1} in")
        if pair.output is not None:
            _draw(axes[1][col], pair.output, cmap, "truth")
        if predictions is not None and k < len(predictions):
            _draw(axes[2][col], predictions[k], cmap, "prediction")

    fig.suptitle(task.task_id, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
