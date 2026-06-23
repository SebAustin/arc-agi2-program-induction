"""Program interpreter.

Threads a numpy grid through a linear pipeline of primitives. The grid is
converted to numpy once at the start and back to an immutable ``Grid`` at the
end. Any failure — unknown op, exception, type error, or a result violating the
size guard — collapses the whole program to ``None`` ('does not apply'), which
the search treats as an inadmissible hypothesis.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from arc_solver.dsl.program import Program
from arc_solver.dsl.registry import PRIMITIVES
from arc_solver.grid.grid_ops import to_grid, to_np, within_size_guard
from arc_solver.io.task import Grid


def run(program: Program, grid: Grid) -> Grid | None:
    """Execute ``program`` on a single grid, returning ``None`` on any failure."""
    arr = to_np(grid)
    for op in program:
        prim = PRIMITIVES.get(op.name)
        if prim is None:
            return None
        try:
            result = prim.fn(arr, *op.params)
        except Exception:
            return None
        if result is None:
            return None
        if not isinstance(result, np.ndarray) or result.ndim != 2 or result.size == 0:
            return None
        if not within_size_guard(result):
            return None
        arr = result
    return to_grid(arr)


def run_batch(program: Program, grids: Sequence[Grid]) -> list[Grid | None]:
    """Execute a program on several grids (e.g. all train inputs)."""
    return [run(program, g) for g in grids]
