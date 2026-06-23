"""Grid comparison metrics used by the verifier and selector."""

from __future__ import annotations

import numpy as np

from arc_solver.io.task import Grid


def grid_equal(a: Grid | None, b: Grid | None) -> bool:
    """Exact equality, tolerant of ``None`` (a failed program output)."""
    if a is None or b is None:
        return False
    return a == b


def same_shape(a: Grid, b: Grid) -> bool:
    return len(a) == len(b) and len(a[0]) == len(b[0])


def cell_agreement(a: Grid | None, b: Grid | None) -> float:
    """Fraction of matching cells when shapes agree, else 0.0.

    Used only as a *soft* signal to steer refinement — never to admit a program
    into the posterior (admission requires exact equality on all train pairs).
    """
    if a is None or b is None or not same_shape(a, b):
        return 0.0
    arr_a = np.array(a)
    arr_b = np.array(b)
    return float((arr_a == arr_b).mean())
