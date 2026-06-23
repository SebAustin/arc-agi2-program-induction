"""Symmetry detection and occlusion repair.

Many ARC tasks hide part of a symmetric pattern behind a 'noise' colour and ask
for the completed grid. ``repair_symmetry`` reconstructs occluded cells by
overlaying the grid's own symmetric images (horizontal, vertical, 180-rotation,
and both diagonals) wherever they agree.
"""

from __future__ import annotations

import numpy as np


def _candidate_views(arr: np.ndarray) -> list[np.ndarray]:
    """Symmetric images of ``arr`` that share its shape."""
    views = [np.fliplr(arr), np.flipud(arr), np.rot90(arr, 2)]
    if arr.shape[0] == arr.shape[1]:  # square-only diagonal symmetries
        views.append(arr.T)
        views.append(np.rot90(arr.T, 2))
    return views


def repair_symmetry(arr: np.ndarray, noise_color: int) -> np.ndarray | None:
    """Fill cells equal to ``noise_color`` using symmetric counterparts.

    Returns a repaired copy, or ``None`` if nothing could be filled (so the
    enumerator can discard the program as inapplicable).
    """
    out = arr.copy()
    mask = out == noise_color
    if not mask.any():
        return None
    filled_any = False
    for view in _candidate_views(arr):
        if view.shape != arr.shape:
            continue
        fillable = mask & (view != noise_color)
        if fillable.any():
            out[fillable] = view[fillable]
            mask = out == noise_color
            filled_any = True
        if not mask.any():
            break
    return out if filled_any else None


def is_symmetric(arr: np.ndarray, kind: str) -> bool:
    """Test a named symmetry: 'h', 'v', 'rot180', 'transpose'."""
    if kind == "h":
        return bool(np.array_equal(arr, np.fliplr(arr)))
    if kind == "v":
        return bool(np.array_equal(arr, np.flipud(arr)))
    if kind == "rot180":
        return bool(np.array_equal(arr, np.rot90(arr, 2)))
    if kind == "transpose":
        return arr.shape[0] == arr.shape[1] and bool(np.array_equal(arr, arr.T))
    raise ValueError(f"unknown symmetry kind: {kind}")
