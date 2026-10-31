"""F2 panel primitives: select a panel, or combine panels cellwise.

Both are **fitter-only** (``param_space=lambda ctx: []``): the F2 fitter infers
the separator colour and the rule from the demonstrations and emits these with
concrete parameters, so the enumerative search is untouched (ADR-009 rationale).
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import register
from arc_solver.grid.partition import split_panels

_SELECT_COST = 5.0
_COMBINE_COST = 8.0


def _unique_index(panels: list[np.ndarray]) -> int | None:
    """Index of the one panel that differs while all the others are equal."""
    if len(panels) < 3:
        return None
    keys = [(p.shape, p.tobytes()) for p in panels]
    singles = [i for i, k in enumerate(keys) if keys.count(k) == 1]
    if len(singles) != 1:
        return None
    rest = [keys[i] for i in range(len(keys)) if i != singles[0]]
    return singles[0] if len(set(rest)) == 1 else None


@register("panel_select", cost_bits=_SELECT_COST, param_space=lambda ctx: [])
def panel_select(
    arr: np.ndarray, sep_color: int, selector: tuple
) -> np.ndarray | None:
    """Return one panel of ``arr``, chosen by ``selector``.

    ``selector`` is ``("index", k)`` for the k-th panel (row-major), or
    ``("unique",)`` for the odd-one-out panel.
    """
    split = split_panels(arr, int(sep_color))
    if split is None:
        return None
    panels, _ = split
    kind = selector[0]
    if kind == "index":
        k = int(selector[1])
        return panels[k].copy() if 0 <= k < len(panels) else None
    if kind == "unique":
        idx = _unique_index(panels)
        return panels[idx].copy() if idx is not None else None
    return None


@register("panel_combine", cost_bits=_COMBINE_COST, param_space=lambda ctx: [])
def panel_combine(
    arr: np.ndarray, sep_color: int, mapping_spec: tuple, bg: int
) -> np.ndarray | None:
    """Combine same-shape panels cellwise via a learned truth table.

    For each cell, the key is the tuple of ``1``/``0`` (panel non-background here?)
    across panels; ``mapping_spec`` maps each key to an output colour. Returns
    ``None`` if the panels are not all the same shape or a key is unmapped — so a
    test input whose panel structure differs is cleanly rejected.
    """
    split = split_panels(arr, int(sep_color))
    if split is None:
        return None
    panels, _ = split
    shape = panels[0].shape
    if any(p.shape != shape for p in panels):
        return None
    mapping = {tuple(int(x) for x in key): int(val) for key, val in mapping_spec}
    out = np.empty(shape, dtype=arr.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            key = tuple(1 if int(p[i, j]) != int(bg) else 0 for p in panels)
            if key not in mapping:
                return None
            out[i, j] = mapping[key]
    return out
