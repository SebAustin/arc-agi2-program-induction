"""Tests for grid ops, components, symmetry, and metrics."""

from __future__ import annotations

import numpy as np

from arc_solver.grid import grid_ops as G
from arc_solver.grid.components import connected_components
from arc_solver.grid.metrics import cell_agreement, grid_equal, same_shape
from arc_solver.grid.symmetry import is_symmetric, repair_symmetry


def test_to_np_to_grid_round_trip():
    g = ((1, 2, 3), (4, 5, 6))
    assert G.to_grid(G.to_np(g)) == g


def test_rot90_four_times_is_identity():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    out = arr
    for _ in range(4):
        out = G.rot90(out)
    assert np.array_equal(out, arr)


def test_flip_involution():
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(G.flip_h(G.flip_h(arr)), arr)
    assert np.array_equal(G.flip_v(G.flip_v(arr)), arr)


def test_background_is_most_common():
    arr = np.array([[0, 0, 0], [0, 1, 0]])
    assert G.background_color(arr) == 0


def test_palette_and_counts():
    arr = np.array([[0, 1], [1, 2]])
    assert G.palette(arr) == [0, 1, 2]
    assert G.color_counts(arr) == {0: 1, 1: 2, 2: 1}


def test_connected_components_separates_objects():
    arr = np.array([[1, 0, 2], [1, 0, 0]])
    comps = connected_components(arr, background=0)
    assert len(comps) == 2
    sizes = sorted(c.size for c in comps)
    assert sizes == [1, 2]


def test_connected_components_8_connectivity():
    arr = np.array([[1, 0], [0, 1]])
    assert len(connected_components(arr, connectivity=8, background=0)) == 1
    assert len(connected_components(arr, connectivity=4, background=0)) == 2


def test_repair_symmetry_fills_noise():
    # A vertically-symmetric grid with one occluded cell (colour 9).
    arr = np.array([[1, 2, 1], [3, 9, 3]])
    out = repair_symmetry(arr, noise_color=9)
    assert out is not None
    assert 9 not in set(out.flatten())


def test_repair_symmetry_returns_none_without_noise():
    arr = np.array([[1, 2], [2, 1]])
    assert repair_symmetry(arr, noise_color=9) is None


def test_is_symmetric():
    assert is_symmetric(np.array([[1, 1], [2, 2]]), "h")
    assert is_symmetric(np.array([[1, 2], [1, 2]]), "v")


def test_metrics():
    assert grid_equal(((1,),), ((1,),))
    assert not grid_equal(((1,),), None)
    assert same_shape(((1, 2),), ((3, 4),))
    assert cell_agreement(((1, 2),), ((1, 9),)) == 0.5
    assert cell_agreement(((1,),), None) == 0.0
