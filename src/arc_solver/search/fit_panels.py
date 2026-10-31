"""F2 fitter — infer a grid-partition rule from the demonstrations.

For each separator colour common to every demonstration, propose:
  * ``panel_select`` for each panel index and the odd-one-out (consistency
    verification downstream filters the wrong ones);
  * ``panel_combine`` with a truth table fitted from the demonstrations, for each
    candidate background.

Fitter-emitted programs are cheap and verified by the engine, so over-proposing
is harmless. See GLOSSARY ('Panel', 'Separator') and SPEC.md (F2).
"""

from __future__ import annotations

from typing import Iterator

from arc_solver.dsl.program import Op, Program
from arc_solver.grid.grid_ops import background_color, to_np
from arc_solver.grid.partition import candidate_separator_colors, split_panels
from arc_solver.io.task import Task


def _bg_candidates(arr) -> list[int]:
    cands: list[int] = []
    for bg in (0, background_color(arr)):
        if bg not in cands:
            cands.append(int(bg))
    return cands


def _fit_combine(demos, sep: int, bg: int) -> tuple | None:
    """Fit a cellwise ``mask-tuple -> colour`` truth table, or ``None``."""
    mapping: dict[tuple, int] = {}
    n_panels: int | None = None
    for ai, ao in demos:
        split = split_panels(ai, sep)
        if split is None:
            return None
        panels, _ = split
        if n_panels is None:
            n_panels = len(panels)
        elif len(panels) != n_panels:
            return None
        shape = panels[0].shape
        if any(p.shape != shape for p in panels) or ao.shape != shape:
            return None
        for i in range(shape[0]):
            for j in range(shape[1]):
                key = tuple(1 if int(p[i, j]) != bg else 0 for p in panels)
                val = int(ao[i, j])
                if mapping.get(key, val) != val:
                    return None  # conflicting rule -> not a clean combine
                mapping[key] = val
    if not mapping:
        return None
    return tuple(sorted(mapping.items()))


def fit_panels(task: Task) -> Iterator[Program]:
    demos = [
        (to_np(p.input), to_np(p.output)) for p in task.train if p.output is not None
    ]
    if not demos:
        return
    common = set.intersection(
        *(set(candidate_separator_colors(ai)) for ai, _ in demos)
    )
    for sep in sorted(common):
        split0 = split_panels(demos[0][0], sep)
        if split0 is None:
            continue
        n_panels = len(split0[0])
        for k in range(n_panels):
            yield (Op("panel_select", (int(sep), ("index", k))),)
        yield (Op("panel_select", (int(sep), ("unique",))),)
        for bg in _bg_candidates(demos[0][0]):
            mapping = _fit_combine(demos, int(sep), bg)
            if mapping is not None:
                yield (Op("panel_combine", (int(sep), mapping, int(bg))),)
