"""Tests for F1: per-object recolouring / transform and the fitter.

See GLOSSARY ('Per-object mapping') and SPEC.md (F1). The ``map_object_color``
primitive is fitter-only; the fitter learns the attribute->colour table (or the
uniform per-object transform) from the demonstrations.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.interpreter import run
from arc_solver.dsl.registry import load_primitives
from arc_solver.io.task import Pair, Task


def _grid(rows):
    return tuple(tuple(r) for r in rows)


def test_map_object_color_by_size():
    load_primitives()
    from arc_solver.dsl.primitives_object import map_object_color

    arr = np.array([[1, 1, 0, 2], [1, 0, 0, 0]])  # obj size3 (colour1), size1 (colour2)
    out = map_object_color(arr, (4, False, 0), "size", ((1, 9), (3, 8)))
    assert out is not None
    assert out.tolist() == [[8, 8, 0, 9], [8, 0, 0, 0]]


def test_map_object_color_unmapped_key_none():
    load_primitives()
    from arc_solver.dsl.primitives_object import map_object_color

    arr = np.array([[1, 1, 0, 2], [1, 0, 0, 0]])  # sizes 3 and 1
    assert map_object_color(arr, (4, False, 0), "size", ((3, 8),)) is None  # size1 absent


def _fits(progs, train, test):
    return any(
        all(run(p, pr.input) == pr.output for pr in train)
        and run(p, test[0].input) == test[0].output
        for p in progs
    )


def test_fit_recolor_by_size_end_to_end():
    from arc_solver.search.fit_recolor import fit_object_recolor

    load_primitives()
    train = (
        Pair(_grid([[1, 1, 0, 2], [1, 0, 0, 0]]), _grid([[5, 5, 0, 6], [5, 0, 0, 0]])),
        Pair(_grid([[3, 0, 0, 4, 4], [0, 0, 0, 4, 0]]),
             _grid([[6, 0, 0, 5, 5], [0, 0, 0, 5, 0]])),
    )
    test = (Pair(_grid([[7, 7, 7, 0, 8], [0, 0, 0, 0, 0]]),
                 _grid([[5, 5, 5, 0, 6], [0, 0, 0, 0, 0]])),)
    assert _fits(list(fit_object_recolor(Task("rc", train, test))), train, test)


def test_fit_per_object_flip_end_to_end():
    from arc_solver.search.fit_recolor import fit_object_recolor

    load_primitives()
    train = (
        Pair(_grid([[1, 1, 0, 2, 0], [1, 0, 0, 2, 2]]),
             _grid([[1, 1, 0, 0, 2], [0, 1, 0, 2, 2]])),
        Pair(_grid([[3, 3, 3, 0, 0], [0, 0, 3, 0, 4]]),
             _grid([[3, 3, 3, 0, 0], [3, 0, 0, 0, 4]])),
    )
    test = (Pair(_grid([[5, 5, 0, 0, 6], [5, 0, 0, 0, 6]]),
                 _grid([[5, 5, 0, 0, 6], [0, 5, 0, 0, 6]])),)
    assert _fits(list(fit_object_recolor(Task("flip", train, test))), train, test)
