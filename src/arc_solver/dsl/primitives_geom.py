"""Geometric primitives: rotations, reflections, transposes.

These are the cheapest, highest-yield ARC operations, so they carry low MDL
cost. All are parameter-free (a single empty param tuple).
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import register
from arc_solver.grid import grid_ops as G

_GEOM_COST = 2.0


@register("identity", cost_bits=0.5)
def identity(arr: np.ndarray) -> np.ndarray:
    return arr


@register("rot90", cost_bits=_GEOM_COST)
def rot90(arr: np.ndarray) -> np.ndarray:
    return G.rot90(arr)


@register("rot180", cost_bits=_GEOM_COST)
def rot180(arr: np.ndarray) -> np.ndarray:
    return G.rot180(arr)


@register("rot270", cost_bits=_GEOM_COST)
def rot270(arr: np.ndarray) -> np.ndarray:
    return G.rot270(arr)


@register("flip_h", cost_bits=_GEOM_COST)
def flip_h(arr: np.ndarray) -> np.ndarray:
    return G.flip_h(arr)


@register("flip_v", cost_bits=_GEOM_COST)
def flip_v(arr: np.ndarray) -> np.ndarray:
    return G.flip_v(arr)


@register("transpose", cost_bits=_GEOM_COST)
def transpose(arr: np.ndarray) -> np.ndarray:
    return G.transpose(arr)


@register("anti_transpose", cost_bits=_GEOM_COST + 0.5)
def anti_transpose(arr: np.ndarray) -> np.ndarray:
    return G.anti_transpose(arr)
