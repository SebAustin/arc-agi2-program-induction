"""Tests for F2: separator detection, panel primitives, and the panel fitter.

See GLOSSARY ('Panel', 'Separator') and SPEC.md. The panel primitives are
fitter-only; the fitter infers the separator colour and the panel rule from the
demonstrations.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.interpreter import run
from arc_solver.dsl.registry import load_primitives
from arc_solver.grid.partition import candidate_separator_colors, split_panels
from arc_solver.io.task import Pair, Task


def _grid(rows):
    return tuple(tuple(r) for r in rows)


def test_split_vertical_separator():
    arr = np.array([[1, 1, 5, 2, 0], [1, 0, 5, 0, 2]])
    res = split_panels(arr, 5)
    assert res is not None
    panels, layout = res
    assert layout == (1, 2)
    assert panels[0].tolist() == [[1, 1], [1, 0]]
    assert panels[1].tolist() == [[2, 0], [0, 2]]


def test_split_grid_four_panels():
    arr = np.array([[1, 5, 2], [5, 5, 5], [3, 5, 4]])  # a cross of 5s
    res = split_panels(arr, 5)
    assert res is not None
    panels, layout = res
    assert layout == (2, 2)
    assert [p.tolist() for p in panels] == [[[1]], [[2]], [[3]], [[4]]]


def test_split_none_when_no_separator():
    assert split_panels(np.array([[1, 2], [3, 4]]), 5) is None


def test_candidate_separator_colors_finds_line():
    arr = np.array([[1, 1, 5, 2, 0], [1, 0, 5, 0, 2]])
    assert 5 in candidate_separator_colors(arr)


def test_panel_select_index():
    load_primitives()
    from arc_solver.dsl.primitives_partition import panel_select

    arr = np.array([[1, 1, 5, 2, 0], [1, 0, 5, 0, 2]])
    out = panel_select(arr, 5, ("index", 1))
    assert out is not None and out.tolist() == [[2, 0], [0, 2]]


def test_panel_select_out_of_range_none():
    load_primitives()
    from arc_solver.dsl.primitives_partition import panel_select

    arr = np.array([[1, 1, 5, 2, 0], [1, 0, 5, 0, 2]])
    assert panel_select(arr, 5, ("index", 9)) is None


def test_panel_combine_truth_table():
    load_primitives()
    from arc_solver.dsl.primitives_partition import panel_combine

    arr = np.array([[1, 1, 9, 1, 1], [1, 0, 9, 0, 1]])  # L=[[1,1],[1,0]] R=[[1,1],[0,1]]
    mapping = (((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 2))  # AND -> colour 2
    out = panel_combine(arr, 9, mapping, 0)
    assert out is not None and out.tolist() == [[2, 2], [0, 0]]


def test_fit_panels_select_end_to_end():
    from arc_solver.search.fit_panels import fit_panels

    load_primitives()
    train = (
        Pair(_grid([[1, 1, 5, 2, 0], [1, 0, 5, 0, 2]]), _grid([[2, 0], [0, 2]])),
        Pair(_grid([[3, 0, 5, 0, 4], [0, 3, 5, 4, 4]]), _grid([[0, 4], [4, 4]])),
    )
    test = (Pair(_grid([[7, 7, 5, 8, 0], [0, 7, 5, 8, 8]]), _grid([[8, 0], [8, 8]])),)
    task = Task("sel", train, test)
    progs = list(fit_panels(task))
    assert any(
        all(run(p, pr.input) == pr.output for pr in train)
        and run(p, test[0].input) == test[0].output
        for p in progs
    )


def test_fit_panels_combine_end_to_end():
    from arc_solver.search.fit_panels import fit_panels

    load_primitives()
    train = (
        Pair(_grid([[1, 1, 9, 1, 0], [0, 1, 9, 1, 1]]), _grid([[2, 0], [0, 2]])),
        Pair(_grid([[0, 1, 9, 1, 0], [1, 0, 9, 0, 0]]), _grid([[0, 0], [0, 0]])),
    )
    test = (Pair(_grid([[1, 1, 9, 1, 1], [1, 0, 9, 0, 1]]), _grid([[2, 2], [0, 0]])),)
    task = Task("comb", train, test)
    progs = list(fit_panels(task))
    assert any(
        all(run(p, pr.input) == pr.output for pr in train)
        and run(p, test[0].input) == test[0].output
        for p in progs
    )
