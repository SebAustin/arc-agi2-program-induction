"""Tests for the DSL: primitives, interpreter, and program serialisation."""

from __future__ import annotations

from arc_solver.dsl import run, serialize, deserialize, pretty
from arc_solver.dsl.program import Op
from arc_solver.dsl.registry import PRIMITIVES, load_primitives


def setup_module(_):
    load_primitives()


def test_registry_populated():
    assert len(PRIMITIVES) >= 20
    for name in ("identity", "rot90", "recolor", "gravity", "color_map"):
        assert name in PRIMITIVES


def test_identity_program_is_input():
    g = ((1, 2), (3, 4))
    assert run((), g) == g
    assert run((Op("identity"),), g) == g


def test_rot90_x4_identity_via_interpreter():
    g = ((1, 2, 3), (4, 5, 6))
    assert run((Op("rot90"),) * 4, g) == g


def test_recolor_and_color_map():
    g = ((1, 2), (2, 1))
    assert run((Op("recolor", (1, 9)),), g) == ((9, 2), (2, 9))
    assert run((Op("color_map", (((1, 7), (2, 8)),)),), g) == ((7, 8), (8, 7))


def test_unknown_op_returns_none():
    assert run((Op("does_not_exist"),), ((1,),)) is None


def test_pipeline_composition():
    g = ((1, 0), (0, 0))
    # flip_h then recolor
    out = run((Op("flip_h"), Op("recolor", (1, 5))), g)
    assert out == ((0, 5), (0, 0))


def test_serialize_round_trip_simple():
    prog = (Op("rot90"), Op("recolor", (1, 2)))
    assert deserialize(serialize(prog)) == prog


def test_serialize_round_trip_nested_params():
    prog = (Op("color_map", (((1, 2), (3, 4)),)),)
    restored = deserialize(serialize(prog))
    assert restored == prog
    # params must stay hashable after a round trip
    assert hash(restored[0])


def test_pretty():
    assert pretty(()) == "<identity>"
    assert pretty((Op("rot90"),)) == "rot90"
    assert "recolor(1,2)" in pretty((Op("recolor", (1, 2)),))


def test_tile_and_scale():
    assert run((Op("tile", (2, 2)),), ((1, 2),)) == ((1, 2, 1, 2), (1, 2, 1, 2))
    assert run((Op("scale", (2, 2)),), ((1,),)) == ((1, 1), (1, 1))


def test_gravity_down():
    g = ((1, 0), (0, 0))
    assert run((Op("gravity", ("down",)),), g) == ((0, 0), (1, 0))


def test_size_guard_blocks_blowup():
    # Tiling a 30x30 grid 3x3 -> 90x90 = 8100 cells > MAX_GRID_CELLS -> None.
    big = tuple(tuple(1 for _ in range(30)) for _ in range(30))
    assert run((Op("tile", (3, 3)),), big) is None
