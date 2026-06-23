"""Arithmetic / split-combine primitives.

A recurring ARC pattern presents two halves of a grid separated by a divider (or
stacked) and asks for a cell-wise combination. These primitives split the grid
in half and overlay the halves with boolean logic, painting hits in a chosen
colour.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import TaskContext, register
from arc_solver.grid.grid_ops import background_color

_ARITH_COST = 5.5


def _color_space(ctx: TaskContext) -> list[tuple]:
    return [(c,) for c in ctx.colors]


def _halves(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split into two equal halves along the longer even axis."""
    h, w = arr.shape
    if w % 2 == 0 and w >= 2:
        return arr[:, : w // 2], arr[:, w // 2 :]
    if h % 2 == 0 and h >= 2:
        return arr[: h // 2, :], arr[h // 2 :, :]
    return None


def _combine(arr: np.ndarray, color: int, mode: str) -> np.ndarray | None:
    parts = _halves(arr)
    if parts is None:
        return None
    a, b = parts
    if a.shape != b.shape:
        return None
    bg = background_color(arr)
    fa, fb = a != bg, b != bg
    if mode == "and":
        hit = fa & fb
    elif mode == "or":
        hit = fa | fb
    elif mode == "xor":
        hit = fa ^ fb
    else:
        return None
    out = np.full(a.shape, bg, dtype=arr.dtype)
    out[hit] = color
    return out


@register("overlay_and", cost_bits=_ARITH_COST, param_space=_color_space)
def overlay_and(arr: np.ndarray, color: int) -> np.ndarray | None:
    return _combine(arr, color, "and")


@register("overlay_or", cost_bits=_ARITH_COST, param_space=_color_space)
def overlay_or(arr: np.ndarray, color: int) -> np.ndarray | None:
    return _combine(arr, color, "or")


@register("overlay_xor", cost_bits=_ARITH_COST, param_space=_color_space)
def overlay_xor(arr: np.ndarray, color: int) -> np.ndarray | None:
    return _combine(arr, color, "xor")
