"""Object-centric primitives built on connected components.

These encapsulate 'objectness' inside Grid->Grid functions so the linear
pipeline never needs a separate object value type. Gravity, selection of the
largest/smallest object, and colour-based filtering live here.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import PRIMITIVES, TaskContext, register
from arc_solver.grid.components import connected_components
from arc_solver.grid.grid_ops import background_color
from arc_solver.grid.objects import (
    ObjectModel,
    extract_objects,
    normalized_shape,
)

_OBJ_COST = 5.0
_DIRS = ("up", "down", "left", "right")


def _dir_space(ctx: TaskContext) -> list[tuple]:
    return [(d,) for d in _DIRS]


def _color_space(ctx: TaskContext) -> list[tuple]:
    return [(c,) for c in ctx.input_colors]


@register("keep_largest", cost_bits=_OBJ_COST)
def keep_largest(arr: np.ndarray) -> np.ndarray | None:
    """Blank every cell except the single largest object (by cell count)."""
    bg = background_color(arr)
    comps = connected_components(arr, background=bg)
    if not comps:
        return None
    biggest = max(comps, key=lambda c: c.size)
    out = np.full_like(arr, bg)
    for r, c in biggest.cells:
        out[r, c] = arr[r, c]
    return out


@register("keep_smallest", cost_bits=_OBJ_COST)
def keep_smallest(arr: np.ndarray) -> np.ndarray | None:
    bg = background_color(arr)
    comps = connected_components(arr, background=bg)
    if not comps:
        return None
    smallest = min(comps, key=lambda c: c.size)
    out = np.full_like(arr, bg)
    for r, c in smallest.cells:
        out[r, c] = arr[r, c]
    return out


@register("remove_color", cost_bits=_OBJ_COST, param_space=_color_space)
def remove_color(arr: np.ndarray, color: int) -> np.ndarray | None:
    """Replace a colour with the background (delete those objects)."""
    bg = background_color(arr)
    if color == bg or not (arr == color).any():
        return None
    out = arr.copy()
    out[arr == color] = bg
    return out


@register("gravity", cost_bits=_OBJ_COST, param_space=_dir_space)
def gravity(arr: np.ndarray, direction: str) -> np.ndarray | None:
    """Slide all non-background cells in ``direction`` until blocked.

    Columns/rows are compacted independently (sand-falling semantics).
    """
    bg = background_color(arr)
    out = np.full_like(arr, bg)
    if direction in ("down", "up"):
        for j in range(arr.shape[1]):
            col = [arr[i, j] for i in range(arr.shape[0]) if arr[i, j] != bg]
            if direction == "down":
                for k, v in enumerate(reversed(col)):
                    out[arr.shape[0] - 1 - k, j] = v
            else:
                for k, v in enumerate(col):
                    out[k, j] = v
    elif direction in ("left", "right"):
        for i in range(arr.shape[0]):
            row = [arr[i, j] for j in range(arr.shape[1]) if arr[i, j] != bg]
            if direction == "right":
                for k, v in enumerate(reversed(row)):
                    out[i, arr.shape[1] - 1 - k] = v
            else:
                for k, v in enumerate(row):
                    out[i, k] = v
    else:
        return None
    if np.array_equal(out, arr):
        return None  # no movement -> inapplicable, prune it
    return out


@register("per_object", cost_bits=6.0, param_space=lambda ctx: [])
def per_object(
    arr: np.ndarray,
    inner_name: str,
    inner_params: tuple,
    model_spec: tuple,
) -> np.ndarray | None:
    """Apply an inner primitive to each object's bounding-box block, in place.

    The ``per_object`` combinator (ADR-009) extracts objects under ``model_spec``
    and rewrites each object's bounding-box block with
    ``inner_name(block, *inner_params)``. The inner op must preserve the block's
    shape, since object placement is otherwise ambiguous. Fitter-only: it
    advertises an empty parameter space (``param_space=lambda ctx: []``) so the
    enumerative search never grounds it and the search budget is unchanged.
    """
    if inner_name == "per_object":
        return None  # no self-nesting
    inner = PRIMITIVES.get(inner_name)
    if inner is None:
        return None
    objs = extract_objects(arr, ObjectModel.from_spec(tuple(model_spec)))
    if not objs:
        return None
    out = arr.copy()
    for obj in objs:
        r0, c0, r1, c1 = obj.bbox
        block = arr[r0 : r1 + 1, c0 : c1 + 1]
        res = inner.fn(block, *tuple(inner_params))
        if not isinstance(res, np.ndarray) or res.shape != block.shape:
            return None
        out[r0 : r1 + 1, c0 : c1 + 1] = res
    if np.array_equal(out, arr):
        return None
    return out


def _select_index(objs: list, criterion: tuple) -> int | None:
    """Index of the object chosen by ``criterion``, or ``None`` if ambiguous."""
    kind = criterion[0]
    if kind == "largest":
        return max(range(len(objs)), key=lambda i: objs[i].size)
    if kind == "smallest":
        return min(range(len(objs)), key=lambda i: objs[i].size)
    if kind == "unique_shape":
        shapes = [normalized_shape(o) for o in objs]
        uniq = [i for i, s in enumerate(shapes) if shapes.count(s) == 1]
        return uniq[0] if len(uniq) == 1 else None
    if kind == "unique_color":
        colors = [o.color for o in objs]
        uniq = [
            i for i, c in enumerate(colors) if c is not None and colors.count(c) == 1
        ]
        return uniq[0] if len(uniq) == 1 else None
    if kind == "largest_bbox":
        return max(range(len(objs)), key=lambda i: objs[i].height * objs[i].width)
    if kind == "topmost":
        return min(range(len(objs)), key=lambda i: objs[i].bbox[0])
    if kind == "bottommost":
        return max(range(len(objs)), key=lambda i: objs[i].bbox[2])
    if kind == "leftmost":
        return min(range(len(objs)), key=lambda i: objs[i].bbox[1])
    if kind == "rightmost":
        return max(range(len(objs)), key=lambda i: objs[i].bbox[3])
    if kind == "unique_size":
        sizes = [o.size for o in objs]
        uniq = [i for i, s in enumerate(sizes) if sizes.count(s) == 1]
        return uniq[0] if len(uniq) == 1 else None
    return None


@register("select_object", cost_bits=5.0, param_space=lambda ctx: [])
def select_object(
    arr: np.ndarray, model_spec: tuple, criterion: tuple, crop: bool
) -> np.ndarray | None:
    """Select one object under ``model_spec`` by ``criterion`` (F4).

    With ``crop`` the result is the selected object's bounding-box block; without
    it the object is kept on a blank canvas the size of the input. Fitter-only.
    """
    model = ObjectModel.from_spec(tuple(model_spec))
    objs = extract_objects(arr, model)
    if not objs:
        return None
    idx = _select_index(objs, criterion)
    if idx is None:
        return None
    obj = objs[idx]
    bg = model.background
    if crop:
        r0, c0, r1, c1 = obj.bbox
        out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=arr.dtype)
        for r, c in obj.cells:
            out[r - r0, c - c0] = arr[r, c]
        return out
    out = np.full_like(arr, bg)
    for r, c in obj.cells:
        out[r, c] = arr[r, c]
    return None if np.array_equal(out, arr) else out


@register("count_grid", cost_bits=6.0, param_space=lambda ctx: [])
def count_grid(
    arr: np.ndarray, model_spec: tuple, color: int, axis: str
) -> np.ndarray | None:
    """Emit a bar whose length is the object count under ``model_spec`` (F4).

    ``axis='v'`` returns an N×1 column, ``'h'`` a 1×N row, filled with ``color``.
    Fitter-only.
    """
    n = len(extract_objects(arr, ObjectModel.from_spec(tuple(model_spec))))
    if n < 1:
        return None
    if axis == "v":
        return np.full((n, 1), int(color), dtype=arr.dtype)
    if axis == "h":
        return np.full((1, n), int(color), dtype=arr.dtype)
    return None


def _object_keys(objs: list, key_kind: str) -> list:
    """Per-object key used to look up a colour in ``map_object_color`` (F1)."""
    if key_kind == "size":
        return [o.size for o in objs]
    if key_kind == "color":
        return [o.color for o in objs]  # may contain None for colour-spanning objs
    if key_kind == "size_rank":
        order = sorted({o.size for o in objs}, reverse=True)  # 0 = largest
        rank = {s: i for i, s in enumerate(order)}
        return [rank[o.size] for o in objs]
    if key_kind == "shape":
        return [tuple(sorted(normalized_shape(o))) for o in objs]
    return [None] * len(objs)


@register("map_object_color", cost_bits=6.0, param_space=lambda ctx: [])
def map_object_color(
    arr: np.ndarray, model_spec: tuple, key_kind: str, mapping_spec: tuple
) -> np.ndarray | None:
    """Recolour each object by a learned ``attribute -> colour`` table (F1).

    The key per object is ``key_kind`` (size / colour / size_rank / shape). Every
    object's key must be present in ``mapping_spec`` or the program does not apply
    (returns ``None``). Background and non-object cells are left untouched.
    Fitter-only.
    """
    model = ObjectModel.from_spec(tuple(model_spec))
    objs = extract_objects(arr, model)
    if not objs:
        return None
    mapping = {_as_key(k): int(v) for k, v in mapping_spec}
    out = arr.copy()
    for obj, key in zip(objs, _object_keys(objs, key_kind)):
        if key is None:
            return None
        hk = _as_key(key)
        if hk not in mapping:
            return None
        for r, c in obj.cells:
            out[r, c] = mapping[hk]
    return None if np.array_equal(out, arr) else out


def _as_key(key):
    """Normalise a mapping key to a hashable form (lists -> tuples recursively)."""
    if isinstance(key, list):
        return tuple(_as_key(k) for k in key)
    if isinstance(key, tuple):
        return tuple(_as_key(k) for k in key)
    return key
