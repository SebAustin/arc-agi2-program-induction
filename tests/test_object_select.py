"""Tests for F4: object selection / counting primitives and the fitter.

See GLOSSARY ('Object', 'Per-object mapping') and SPEC.md (F4). The primitives are
fitter-only; the fitter infers the object model, the selection criterion, and
whether to crop, from the demonstrations.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.interpreter import run
from arc_solver.dsl.registry import load_primitives
from arc_solver.io.task import Pair, Task


def _grid(rows):
    return tuple(tuple(r) for r in rows)


def _select():
    load_primitives()
    from arc_solver.dsl.primitives_object import select_object

    return select_object


def test_select_largest_cropped():
    sel = _select()
    arr = np.array([[1, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]])
    out = sel(arr, (4, False, 0), ("largest",), True)
    assert out is not None and out.tolist() == [[1, 1], [1, 0]]


def test_select_smallest_cropped():
    sel = _select()
    arr = np.array([[1, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]])
    out = sel(arr, (4, False, 0), ("smallest",), True)
    assert out is not None and out.tolist() == [[2]]


def test_select_largest_not_cropped_blanks_others():
    sel = _select()
    arr = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
    out = sel(arr, (4, False, 0), ("largest",), False)
    assert out is not None
    assert out.tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 0]]  # the 2 removed


def test_select_unique_shape():
    sel = _select()
    arr = np.array([[1, 0, 2, 0, 3, 3], [0, 0, 0, 0, 0, 0]])  # two singles, one pair
    out = sel(arr, (4, False, 0), ("unique_shape",), True)
    assert out is not None and out.tolist() == [[3, 3]]


def test_select_unique_color():
    sel = _select()
    arr = np.array([[1, 0, 1, 0, 2], [0, 0, 0, 0, 0]])  # colours [1,1,2] -> 2 unique
    out = sel(arr, (4, False, 0), ("unique_color",), True)
    assert out is not None and out.tolist() == [[2]]


def test_count_grid_vertical():
    load_primitives()
    from arc_solver.dsl.primitives_object import count_grid

    arr = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 0]])  # three objects
    out = count_grid(arr, (4, False, 0), 5, "v")
    assert out is not None and out.tolist() == [[5], [5], [5]]


def _fits(progs, train, test):
    return any(
        all(run(p, pr.input) == pr.output for pr in train)
        and run(p, test[0].input) == test[0].output
        for p in progs
    )


def test_fit_select_largest_crop_end_to_end():
    from arc_solver.search.fit_select import fit_select

    load_primitives()
    train = (
        Pair(_grid([[2, 0, 0], [2, 2, 0], [0, 0, 4]]), _grid([[2, 0], [2, 2]])),
        Pair(_grid([[0, 7, 0], [0, 7, 0], [3, 0, 0]]), _grid([[7], [7]])),
    )
    test = (Pair(_grid([[5, 5, 0], [0, 5, 0], [0, 0, 8]]), _grid([[5, 5], [0, 5]])),)
    assert _fits(list(fit_select(Task("sel", train, test))), train, test)


def test_fit_select_count_end_to_end():
    from arc_solver.search.fit_select import fit_select

    load_primitives()
    train = (
        Pair(_grid([[1, 0, 0], [0, 0, 2]]), _grid([[5], [5]])),
        Pair(_grid([[1, 0, 2], [0, 0, 0], [3, 0, 0]]), _grid([[5], [5], [5]])),
    )
    test = (Pair(_grid([[4, 0, 0], [0, 0, 6]]), _grid([[5], [5]])),)
    assert _fits(list(fit_select(Task("count", train, test))), train, test)
