"""Targeted tests for object-centric primitives and viz rendering."""

from __future__ import annotations

import os

import numpy as np
import pytest

from arc_solver.dsl.primitives_object import (
    gravity,
    keep_largest,
    keep_smallest,
    remove_color,
)


def test_keep_largest_blanks_others():
    arr = np.array([[1, 0, 2], [1, 0, 2], [1, 0, 0]])  # colour1 size3 > colour2 size2
    out = keep_largest(arr)
    assert out is not None
    assert set(np.unique(out)) == {0, 1}


def test_keep_smallest():
    arr = np.array([[1, 0, 2], [1, 0, 0], [1, 0, 0]])
    out = keep_smallest(arr)
    assert out is not None
    assert set(np.unique(out)) == {0, 2}


def test_keep_largest_none_when_empty():
    assert keep_largest(np.zeros((3, 3), dtype=int)) is None


def test_remove_color():
    arr = np.array([[0, 0, 1], [0, 2, 1]])  # background 0, remove colour 1
    out = remove_color(arr, 1)
    assert out is not None
    assert 1 not in set(np.unique(out))


def test_remove_color_absent_returns_none():
    arr = np.array([[1, 1], [1, 1]])
    assert remove_color(arr, 5) is None


@pytest.mark.parametrize(
    "direction,start,expected",
    [
        ("down", [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ("up", [[0, 0], [1, 0]], [[1, 0], [0, 0]]),
        ("right", [[1, 0], [0, 0]], [[0, 1], [0, 0]]),
        ("left", [[0, 1], [0, 0]], [[1, 0], [0, 0]]),
    ],
)
def test_gravity_directions(direction, start, expected):
    out = gravity(np.array(start), direction)
    assert out is not None
    assert out.tolist() == expected


def test_gravity_no_movement_returns_none():
    arr = np.array([[0, 0], [1, 0]])  # already at bottom
    assert gravity(arr, "down") is None


def test_gravity_unknown_direction():
    assert gravity(np.array([[1, 0]]), "sideways") is None


def test_viz_grid_to_png(tmp_path):
    pytest.importorskip("matplotlib")
    from arc_solver.viz.render import grid_to_png

    path = os.path.join(tmp_path, "g.png")
    grid_to_png(((1, 2, 3), (4, 5, 6)), path)
    assert os.path.exists(path) and os.path.getsize(path) > 0


def test_viz_task_figure(tmp_path, flip_task):
    pytest.importorskip("matplotlib")
    from arc_solver.viz.render import render_task_figure

    path = os.path.join(tmp_path, "task.png")
    render_task_figure(flip_task, path, predictions=[((0, 9), (2, 1))])
    assert os.path.exists(path) and os.path.getsize(path) > 0
