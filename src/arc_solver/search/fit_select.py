"""F4 fitter — infer an object-selection or counting rule from the demos.

Searches the small object-model set and a fixed list of selection criteria
(largest / smallest / unique-shape / unique-colour, cropped or not), and a simple
object-count-to-bar rule. Only programs that reproduce *every* demonstration are
yielded; the engine verifies again downstream. See GLOSSARY ('Object') and
SPEC.md (F4).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from arc_solver.dsl.primitives_object import count_grid, select_object
from arc_solver.dsl.program import Op, Program
from arc_solver.grid.grid_ops import to_np
from arc_solver.grid.objects import ObjectModel, candidate_models, extract_objects
from arc_solver.io.task import Task

_CRITERIA = (("largest",), ("smallest",), ("unique_shape",), ("unique_color",))


def _eq(result, target) -> bool:
    return (
        result is not None
        and result.shape == target.shape
        and bool(np.array_equal(result, target))
    )


def _fit_count(demos, spec) -> tuple[int, str] | None:
    """Infer a consistent (colour, axis) for an object-count bar, or ``None``."""
    model = ObjectModel.from_spec(spec)
    seen: set[tuple[int, str]] = set()
    for ai, ao in demos:
        n = len(extract_objects(ai, model))
        colors = {int(v) for v in ao.flatten()}
        if len(colors) != 1:
            return None
        color = colors.pop()
        if ao.shape == (n, 1):
            seen.add((color, "v"))
        elif ao.shape == (1, n):
            seen.add((color, "h"))
        else:
            return None
    return next(iter(seen)) if len(seen) == 1 else None


def fit_select(task: Task) -> Iterator[Program]:
    demos = [
        (to_np(p.input), to_np(p.output)) for p in task.train if p.output is not None
    ]
    if not demos:
        return
    for model in candidate_models(demos[0][0]):
        spec = model.to_spec()
        for criterion in _CRITERIA:
            for crop in (True, False):
                if all(
                    _eq(select_object(ai, spec, criterion, crop), ao)
                    for ai, ao in demos
                ):
                    yield (Op("select_object", (spec, criterion, crop)),)
        count = _fit_count(demos, spec)
        if count is not None:
            color, axis = count
            yield (Op("count_grid", (spec, int(color), axis)),)
