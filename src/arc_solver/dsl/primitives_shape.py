"""Shape primitives: cropping, tiling, scaling, mirroring, and symmetry repair.

These change grid dimensions, so their parameter spaces lean on the observed
scale factors in the task context to stay tractable.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import TaskContext, register
from arc_solver.grid.grid_ops import background_color
from arc_solver.grid.symmetry import repair_symmetry

_SHAPE_COST = 4.0


def _scale_space(ctx: TaskContext) -> list[tuple]:
    """Integer scale factors observed between train inputs and outputs (plus 2,3)."""
    seen = {f for f in ctx.scale_factors if f[0] >= 1 and f[1] >= 1}
    seen.update({(2, 2), (3, 3), (1, 2), (2, 1), (1, 3), (3, 1)})
    return [f for f in sorted(seen) if f != (1, 1)]


def _color_space(ctx: TaskContext) -> list[tuple]:
    return [(c,) for c in ctx.colors]


@register("crop_to_content", cost_bits=_SHAPE_COST)
def crop_to_content(arr: np.ndarray) -> np.ndarray | None:
    """Crop to the bounding box of all non-background cells."""
    bg = background_color(arr)
    mask = arr != bg
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    if (r1 - r0 + 1, c1 - c0 + 1) == arr.shape:
        return None  # nothing to crop
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()  # own the buffer (no aliasing)


@register("tile", cost_bits=_SHAPE_COST + 1.0, param_space=_scale_space)
def tile(arr: np.ndarray, ny: int, nx: int) -> np.ndarray | None:
    """Repeat the grid ``ny`` times vertically and ``nx`` times horizontally."""
    if ny < 1 or nx < 1 or (ny == 1 and nx == 1):
        return None
    return np.tile(arr, (ny, nx))


@register("scale", cost_bits=_SHAPE_COST + 1.0, param_space=_scale_space)
def scale(arr: np.ndarray, ky: int, kx: int) -> np.ndarray | None:
    """Upscale each cell into a ``ky`` x ``kx`` block (nearest-neighbour)."""
    if ky < 1 or kx < 1 or (ky == 1 and kx == 1):
        return None
    return np.kron(arr, np.ones((ky, kx), dtype=arr.dtype))


@register("mirror_concat_h", cost_bits=_SHAPE_COST)
def mirror_concat_h(arr: np.ndarray) -> np.ndarray:
    """Append the horizontal mirror to the right: [A | flip_h(A)]."""
    return np.concatenate([arr, np.fliplr(arr)], axis=1)


@register("mirror_concat_v", cost_bits=_SHAPE_COST)
def mirror_concat_v(arr: np.ndarray) -> np.ndarray:
    """Append the vertical mirror below: [A ; flip_v(A)]."""
    return np.concatenate([arr, np.flipud(arr)], axis=0)


@register("trim_border", cost_bits=_SHAPE_COST)
def trim_border(arr: np.ndarray) -> np.ndarray | None:
    """Remove the outermost one-cell ring."""
    if arr.shape[0] <= 2 or arr.shape[1] <= 2:
        return None
    return arr[1:-1, 1:-1].copy()  # own the buffer (no aliasing)


@register("repair_symmetry", cost_bits=_SHAPE_COST, param_space=_color_space)
def repair_symmetry_op(arr: np.ndarray, noise_color: int) -> np.ndarray | None:
    """Fill cells of ``noise_color`` using the grid's own symmetric images."""
    return repair_symmetry(arr, noise_color)


@register("const_grid", cost_bits=8.0, param_space=lambda ctx: [])
def const_grid(arr: np.ndarray, grid: tuple) -> np.ndarray:
    """Return a fixed grid regardless of the input.

    Used for tasks whose output is constant across all demonstrations. The
    parameter space is empty for the enumerator; the :class:`FittedProposer`
    supplies the concrete grid. Its high MDL cost means it only wins when no
    simpler input-dependent rule fits.
    """
    return np.array(grid, dtype=np.int16)
