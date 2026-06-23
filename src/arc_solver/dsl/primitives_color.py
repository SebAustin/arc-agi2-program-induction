"""Colour primitives: recolouring, swapping, and palette mapping.

Parameter spaces are conditioned on the colours actually present in the task, so
the enumerator never tries colours that cannot appear.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import TaskContext, register
from arc_solver.grid.grid_ops import background_color

_COLOR_COST = 4.0


def _color_pairs(ctx: TaskContext) -> list[tuple]:
    """All ordered (src, dst) colour pairs over colours seen in the task."""
    cols = ctx.colors
    return [(s, d) for s in cols for d in cols if s != d]


def _single_colors(ctx: TaskContext) -> list[tuple]:
    return [(c,) for c in ctx.colors]


@register("recolor", cost_bits=_COLOR_COST, param_space=_color_pairs)
def recolor(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Replace every ``src`` cell with ``dst``."""
    out = arr.copy()
    out[arr == src] = dst
    return out


@register("color_map", cost_bits=_COLOR_COST, param_space=lambda ctx: [])
def color_map(arr: np.ndarray, mapping: tuple) -> np.ndarray:
    """Apply a full colour remapping given as a tuple of ``(src, dst)`` pairs.

    Colours absent from the mapping are kept. The parameter space is empty for
    the enumerator (a full bijection is too large to enumerate); the
    :class:`FittedProposer` constructs this op directly from the train pairs.
    """
    out = arr.copy()
    for src, dst in mapping:
        out[arr == src] = dst
    return out


@register("swap_colors", cost_bits=_COLOR_COST, param_space=_color_pairs)
def swap_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray | None:
    """Exchange colours ``a`` and ``b``. Inapplicable if neither is present."""
    if not (arr == a).any() and not (arr == b).any():
        return None
    out = arr.copy()
    out[arr == a] = b
    out[arr == b] = a
    return out


@register("set_foreground", cost_bits=_COLOR_COST, param_space=_single_colors)
def set_foreground(arr: np.ndarray, color: int) -> np.ndarray | None:
    """Recolour every non-background cell to ``color``."""
    bg = background_color(arr)
    if color == bg:
        return None
    out = arr.copy()
    out[arr != bg] = color
    return out


@register("replace_bg", cost_bits=_COLOR_COST, param_space=_single_colors)
def replace_bg(arr: np.ndarray, color: int) -> np.ndarray | None:
    """Recolour the background to ``color``."""
    bg = background_color(arr)
    if color == bg:
        return None
    out = arr.copy()
    out[arr == bg] = color
    return out
