"""Tests for the pass@2-aware selector."""

from __future__ import annotations

from arc_solver.dsl.program import Op
from arc_solver.select.posterior import ScoredProgram, posterior_weights
from arc_solver.select.selector import pick_for_input, pick_top2, _fallback_alt


def _scored(programs):
    return posterior_weights(list(programs))


def test_attempt1_is_posterior_mode():
    # Two programs map the input to grid A, one to grid B -> A is the mode.
    # rot180 of ((1,2),(3,4)) = ((4,3),(2,1)); flip_h+flip_v also = rot180.
    g = ((1, 2), (3, 4))
    scored = _scored([(Op("rot180"),), (Op("flip_h"), Op("flip_v")), (Op("rot90"),)])
    a1, a2 = pick_for_input(scored, g)
    assert a1 == ((4, 3), (2, 1))  # rot180 output (mass from two programs)
    assert a2 != a1  # second attempt disagrees


def test_attempt2_differs_when_single_output():
    g = ((1, 2), (3, 4))
    scored = _scored([(Op("rot90"),)])  # one program -> one output
    a1, a2 = pick_for_input(scored, g)
    assert a1 != a2  # fallback supplies a distinct second guess


def test_fallback_when_no_programs():
    g = ((1, 2), (3, 4))
    a1, a2 = pick_for_input([], g)
    assert a1 == g  # identity fallback
    assert a2 != a1


def test_pick_top2_one_per_test_input():
    g1, g2 = ((1, 2),), ((3, 4),)
    scored = _scored([(Op("flip_h"),)])
    out = pick_top2(scored, [g1, g2])
    assert len(out) == 2


def test_fallback_alt_distinct():
    g = ((1, 2), (3, 4))
    assert _fallback_alt(g) != g
    # symmetric grid -> mirror equals grid, falls back to rot180
    sym = ((1, 1), (2, 2))
    assert _fallback_alt(sym) != sym
