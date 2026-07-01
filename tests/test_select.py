"""Tests for the pass@2-aware selector."""

from __future__ import annotations

from arc_solver.dsl.program import Op
from arc_solver.select.posterior import posterior_weights
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


def test_fallback_alt_never_equals_fully_symmetric_grid():
    """Regression: a grid invariant under mirror AND rot180 must still yield a
    distinct second attempt (previously returned the grid unchanged, silently
    duplicating attempt_1 and wasting a pass@2 guess)."""
    for g in (
        ((3, 3, 3), (3, 3, 3)),  # solid colour
        ((1, 2, 1),),  # single-row palindrome
        ((5,),),  # 1x1
        ((7, 7), (7, 7)),  # uniform square
        ((1, 2, 1), (2, 3, 2), (1, 2, 1)),  # doubly-symmetric square
    ):
        assert _fallback_alt(g) != g


def test_fallback_alt_nonsquare():
    assert _fallback_alt(((1, 2, 3),)) != ((1, 2, 3),)


def test_attempt2_is_fallback_when_all_programs_agree():
    """Two differently-costed programs mapping the input to the SAME output must
    collapse to one hypothesis, so attempt_2 comes from the fallback (distinct)."""
    g = ((1, 2), (3, 4))
    # identity (1 op) and rot90x4 (4 ops) both reproduce the input.
    scored = _scored([(Op("identity"),), (Op("rot90"),) * 4])
    a1, a2 = pick_for_input(scored, g)
    assert a1 == g
    assert a2 == _fallback_alt(a1)
    assert a2 != a1


def test_selection_deterministic_under_reordering():
    """attempt_1/attempt_2 must be a function of the posterior mass alone, not of
    the order programs are supplied (canonical tie-break on exact weight ties)."""
    g = ((1, 2), (3, 4))
    p_rot90 = (Op("rot90"),)  # equal MDL cost to flip_h -> 50/50 weight tie
    p_fliph = (Op("flip_h"),)
    a = pick_for_input(_scored([p_rot90, p_fliph]), g)
    b = pick_for_input(_scored([p_fliph, p_rot90]), g)
    assert a == b
