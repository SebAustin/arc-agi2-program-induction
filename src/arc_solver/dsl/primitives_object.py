"""Object-centric primitives built on connected components.

These encapsulate 'objectness' inside Grid->Grid functions so the linear
pipeline never needs a separate object value type. Gravity, selection of the
largest/smallest object, and colour-based filtering live here.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import TaskContext, register
from arc_solver.grid.components import connected_components
from arc_solver.grid.grid_ops import background_color

_OBJ_COST = 5.0
_DIRS = ("up", "down", "left", "right")


def _dir_space(ctx: TaskContext) -> list[tuple]:
    return [(d,) for d in _DIRS]


def _color_space(ctx: TaskContext) -> list[tuple]:
    return [(c,) for c in ctx.input_colors]


@register("keep_largest", cost_bits=_OBJ_COST)
def keep_largest(arr: np.ndarray) -> np.ndarray | None:
    """Blank every cell except the single largest object (by cell count)."""
    bg = background_color(arr)
    comps = connected_components(arr, background=bg)
    if not comps:
        return None
    biggest = max(comps, key=lambda c: c.size)
    out = np.full_like(arr, bg)
    for r, c in biggest.cells:
        out[r, c] = arr[r, c]
    return out


@register("keep_smallest", cost_bits=_OBJ_COST)
def keep_smallest(arr: np.ndarray) -> np.ndarray | None:
    bg = background_color(arr)
    comps = connected_components(arr, background=bg)
    if not comps:
        return None
    smallest = min(comps, key=lambda c: c.size)
    out = np.full_like(arr, bg)
    for r, c in smallest.cells:
        out[r, c] = arr[r, c]
    return out


@register("remove_color", cost_bits=_OBJ_COST, param_space=_color_space)
def remove_color(arr: np.ndarray, color: int) -> np.ndarray | None:
    """Replace a colour with the background (delete those objects)."""
    bg = background_color(arr)
    if color == bg or not (arr == color).any():
        return None
    out = arr.copy()
    out[arr == color] = bg
    return out


@register("gravity", cost_bits=_OBJ_COST, param_space=_dir_space)
def gravity(arr: np.ndarray, direction: str) -> np.ndarray | None:
    """Slide all non-background cells in ``direction`` until blocked.

    Columns/rows are compacted independently (sand-falling semantics).
    """
    bg = background_color(arr)
    out = np.full_like(arr, bg)
    if direction in ("down", "up"):
        for j in range(arr.shape[1]):
            col = [arr[i, j] for i in range(arr.shape[0]) if arr[i, j] != bg]
            if direction == "down":
                for k, v in enumerate(reversed(col)):
                    out[arr.shape[0] - 1 - k, j] = v
            else:
                for k, v in enumerate(col):
                    out[k, j] = v
    elif direction in ("left", "right"):
        for i in range(arr.shape[0]):
            row = [arr[i, j] for j in range(arr.shape[1]) if arr[i, j] != bg]
            if direction == "right":
                for k, v in enumerate(reversed(row)):
                    out[i, arr.shape[1] - 1 - k] = v
            else:
                for k, v in enumerate(row):
                    out[i, k] = v
    else:
        return None
    if np.array_equal(out, arr):
        return None  # no movement -> inapplicable, prune it
    return out
