"""F1 fitter — infer a per-object recolour or uniform per-object transform.

Two rule shapes, each searched over the small object-model set:
  * ``map_object_color`` — a learned ``attribute -> colour`` table (key by size,
    colour, size-rank, or shape);
  * ``per_object`` — a single shape-preserving transform applied to every object
    in place.

Only programs that reproduce every demonstration (verified by re-applying the
primitive) are yielded. See GLOSSARY ('Per-object mapping') and SPEC.md (F1).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from arc_solver.dsl.primitives_object import (
    _object_keys,
    map_object_color,
    per_object,
)
from arc_solver.dsl.program import Op, Program
from arc_solver.grid.grid_ops import to_np
from arc_solver.grid.objects import ObjectModel, candidate_models, extract_objects
from arc_solver.io.task import Task

# Keys are low-capacity object attributes. "shape" is deliberately excluded: a
# shape->colour table tends to memorise a per-object bijection (one entry per
# object) that is train-consistent but does not generalise, and — because the
# fitter fires and the engine early-stops — it can shadow a correct enumerative
# solve. See SPEC.md (F1) and the train regression on 9565186b.
_INNER_TRANSFORMS = ("flip_h", "flip_v", "rot180", "rot90", "rot270")
_KEY_KINDS = ("size", "color", "size_rank")


def _eq(result, target) -> bool:
    return (
        result is not None
        and result.shape == target.shape
        and bool(np.array_equal(result, target))
    )


def _fit_recolor(demos, spec, key_kind) -> tuple | None:
    """Learn an ``attribute -> colour`` table, then verify it reproduces demos."""
    model = ObjectModel.from_spec(spec)
    mapping: dict = {}
    for ai, ao in demos:
        if ai.shape != ao.shape:
            return None
        objs = extract_objects(ai, model)
        if not objs:
            return None
        for obj, key in zip(objs, _object_keys(objs, key_kind)):
            if key is None:
                return None
            out_colors = {int(ao[r, c]) for r, c in obj.cells}
            if len(out_colors) != 1:
                return None  # object not recoloured uniformly -> not this rule
            col = out_colors.pop()
            if mapping.get(key, col) != col:
                return None
            mapping[key] = col
    if not mapping:
        return None
    spec_mapping = tuple(sorted(mapping.items()))
    if all(_eq(map_object_color(ai, spec, key_kind, spec_mapping), ao) for ai, ao in demos):
        return spec_mapping
    return None


def fit_object_recolor(task: Task) -> Iterator[Program]:
    demos = [
        (to_np(p.input), to_np(p.output)) for p in task.train if p.output is not None
    ]
    if not demos:
        return
    for model in candidate_models(demos[0][0]):
        spec = model.to_spec()
        for inner in _INNER_TRANSFORMS:
            if all(_eq(per_object(ai, inner, (), spec), ao) for ai, ao in demos):
                yield (Op("per_object", (inner, (), spec)),)
        for key_kind in _KEY_KINDS:
            mapping = _fit_recolor(demos, spec, key_kind)
            if mapping is not None:
                yield (Op("map_object_color", (spec, key_kind, mapping)),)
