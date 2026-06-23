"""Low-level grid operations on numpy arrays.

The canonical ``Grid`` type (``tuple[tuple[int, ...], ...]``) is immutable and
hashable; numpy is used only *inside* operations for speed. ``to_np`` / ``to_grid``
are the boundary converters. All transforms here are pure: they return new arrays.
"""

from __future__ import annotations

import numpy as np

from arc_solver.config import MAX_GRID_CELLS, MAX_GRID_DIM
from arc_solver.io.task import Grid


def to_np(grid: Grid) -> np.ndarray:
    """Convert an immutable Grid into a 2-D int numpy array."""
    return np.array(grid, dtype=np.int16)


def to_grid(arr: np.ndarray) -> Grid:
    """Convert a 2-D int array back into an immutable Grid."""
    return tuple(tuple(int(v) for v in row) for row in arr)


def within_size_guard(arr: np.ndarray) -> bool:
    """True iff an array is within the hard size limits (anti-blowup guard)."""
    h, w = arr.shape
    return 1 <= h <= MAX_GRID_DIM and 1 <= w <= MAX_GRID_DIM and h * w <= MAX_GRID_CELLS


# --- Geometric transforms (numpy in / numpy out) --------------------------- #


def rot90(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=1)


def rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=2)


def rot270(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=3)


def flip_h(arr: np.ndarray) -> np.ndarray:
    """Mirror left-right."""
    return np.fliplr(arr)


def flip_v(arr: np.ndarray) -> np.ndarray:
    """Mirror top-bottom."""
    return np.flipud(arr)


def transpose(arr: np.ndarray) -> np.ndarray:
    return arr.T


def anti_transpose(arr: np.ndarray) -> np.ndarray:
    """Transpose across the anti-diagonal."""
    return np.rot90(arr.T, k=2)


# --- Colour helpers -------------------------------------------------------- #


def palette(arr: np.ndarray) -> list[int]:
    """Sorted list of distinct colours present."""
    return sorted(int(v) for v in np.unique(arr))


def color_counts(arr: np.ndarray) -> dict[int, int]:
    """Map of colour -> number of cells."""
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def background_color(arr: np.ndarray) -> int:
    """Heuristic background = most frequent colour (ties broken toward 0)."""
    counts = color_counts(arr)
    if not counts:
        return 0
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == 0))
    return best[0]
