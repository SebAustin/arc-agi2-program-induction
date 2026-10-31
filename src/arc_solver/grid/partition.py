"""Grid partitioning: detect separator lines and split a grid into panels.

A *separator* is a full row or column of a single colour; the *panels* are the
rectangular blocks it leaves behind (GLOSSARY: 'Panel', 'Separator'). Used by the
F2 fitter; pure numpy in / numpy out, no DSL dependency.
"""

from __future__ import annotations

import numpy as np

from arc_solver.grid.grid_ops import palette

_MIN_PANELS = 2


def _bands(n: int, seps: set[int]) -> list[tuple[int, int]]:
    """Contiguous (start, end) index bands between separator positions."""
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        if i in seps:
            if start is not None:
                bands.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        bands.append((start, n - 1))
    return bands


def split_panels(
    arr: np.ndarray, sep_color: int
) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    """Split ``arr`` by full lines of ``sep_color`` into panels (row-major).

    Returns ``(panels, (n_row_bands, n_col_bands))`` or ``None`` when ``sep_color``
    forms no full line, or leaves fewer than two panels.
    """
    h, w = arr.shape
    sep_rows = {r for r in range(h) if bool(np.all(arr[r, :] == sep_color))}
    sep_cols = {c for c in range(w) if bool(np.all(arr[:, c] == sep_color))}
    if not sep_rows and not sep_cols:
        return None
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    if not row_bands or not col_bands:
        return None
    panels = [
        arr[r0 : r1 + 1, c0 : c1 + 1].copy()
        for (r0, r1) in row_bands
        for (c0, c1) in col_bands
    ]
    if len(panels) < _MIN_PANELS:
        return None
    return panels, (len(row_bands), len(col_bands))


def candidate_separator_colors(arr: np.ndarray) -> list[int]:
    """Colours that form at least one full line and split ``arr`` into panels."""
    return [c for c in palette(arr) if split_panels(arr, c) is not None]
