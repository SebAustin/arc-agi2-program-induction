"""Tests for MDL, verifier, posterior, enumerative & fitted proposers."""

from __future__ import annotations

from arc_solver.config import Budget
from arc_solver.dsl.program import Op
from arc_solver.search.engine import SolveEngine
from arc_solver.search.enumerative import EnumerativeProposer
from arc_solver.search.fitted import FittedProposer, _fit_color_map, _fit_scale
from arc_solver.search.mdl import (
    _COLOR_BITS,
    _param_bits,
    description_length,
    prior_logp,
)
from arc_solver.search.verifier import consistent, partial_score
from arc_solver.select.posterior import posterior_weights
from conftest import make_task


# --- MDL -------------------------------------------------------------------- #


def test_mdl_longer_program_costs_more():
    short = (Op("rot90"),)
    long = (Op("rot90"), Op("flip_h"), Op("transpose"))
    assert description_length(long) > description_length(short)


def test_mdl_params_add_cost():
    no_param = (Op("identity"),)
    with_param = (Op("recolor", (1, 2)),)
    assert description_length(with_param) > description_length(no_param)


def test_prior_logp_monotonic_with_length():
    short = (Op("rot90"),)
    long = (Op("rot90"), Op("rot90"))
    assert prior_logp(short) > prior_logp(long)  # shorter = higher prior


# --- Verifier --------------------------------------------------------------- #


def test_consistent_true_for_correct_program(flip_task):
    assert consistent((Op("flip_h"),), flip_task)


def test_consistent_false_for_wrong_program(flip_task):
    assert not consistent((Op("rot90"),), flip_task)


def test_partial_score_between_zero_and_one(flip_task):
    s = partial_score((Op("identity"),), flip_task)
    assert 0.0 <= s < 1.0


# --- Posterior -------------------------------------------------------------- #


def test_posterior_weights_sum_to_one():
    progs = [(Op("rot90"),), (Op("rot90"), Op("flip_h"))]
    scored = posterior_weights(progs)
    assert abs(sum(s.weight for s in scored) - 1.0) < 1e-9


def test_posterior_prefers_shorter_program():
    progs = [(Op("rot90"),), (Op("rot90"), Op("flip_h"), Op("transpose"))]
    scored = {tuple(s.program): s.weight for s in posterior_weights(progs)}
    assert scored[(Op("rot90"),)] > scored[(Op("rot90"), Op("flip_h"), Op("transpose"))]


def test_posterior_empty():
    assert posterior_weights([]) == []


# --- Enumerative ------------------------------------------------------------ #


def _find_consistent(proposer, task, budget):
    for prog in proposer.propose(task, budget=budget):
        if consistent(prog, task):
            return prog
    return None


def test_enumerative_finds_flip(flip_task):
    prog = _find_consistent(EnumerativeProposer(), flip_task, Budget(max_seconds=5))
    assert prog is not None and consistent(prog, flip_task)


def test_enumerative_respects_program_budget(flip_task):
    budget = Budget(max_programs=5, max_seconds=5)
    proposed = list(EnumerativeProposer().propose(flip_task, budget=budget))
    assert len(proposed) <= 5


# --- Fitted ----------------------------------------------------------------- #


def test_fit_color_map(recolor_task):
    mapping = _fit_color_map(recolor_task)
    assert mapping == {3: 5}


def test_fit_scale(tile_task):
    assert _fit_scale(tile_task) == (2, 2)


def test_fitted_proposes_solution(recolor_task):
    prog = _find_consistent(FittedProposer(), recolor_task, Budget())
    assert prog is not None and consistent(prog, recolor_task)


# --- MDL param formula ------------------------------------------------------ #


def test_param_bits_formula():
    """Pin the exact per-type parameter cost (the formula IS the object here)."""
    assert _param_bits(True) == 1.0  # bool before int (bool is an int subclass)
    assert _param_bits(5) == _COLOR_BITS
    assert _param_bits("down") == 2.0  # small string enum
    assert _param_bits((1, 2)) == 2 * _COLOR_BITS  # sum over elements
    assert _param_bits({1: 2, 3: 4}) == 2 * (2 * _COLOR_BITS)  # len * 2*colour


# --- Verifier None-output path ---------------------------------------------- #


def test_consistent_skips_none_output_pair():
    """A train pair with a missing output is skipped, not treated as a failure."""
    g = ((1, 2), (3, 4))
    task = make_task([(g, g), (g, None)], [(g, None)])
    assert consistent((Op("identity"),), task)


# --- Engine: dedup, early-stop, determinism --------------------------------- #


class _ListProposer:
    """A fake proposer that yields a fixed program list and records invocation."""

    def __init__(self, programs):
        self._programs = programs
        self.called = False

    def propose(self, task, *, budget, feedback=None):
        self.called = True
        yield from self._programs


def test_engine_dedup_keeps_cheapest_program(identity_task):
    expensive = (Op("rot90"),) * 4  # identity behaviour, 4 ops
    cheap = (Op("identity"),)  # identity behaviour, 1 op
    engine = SolveEngine(
        [_ListProposer([expensive, cheap])],
        Budget(max_seconds=5),
        early_stop_admissible=999,
    )
    result = engine.solve(identity_task)
    assert result.meta["n_admissible"] == 1  # collapsed to one behaviour
    assert result.best_program == cheap  # cheaper representative survives


def test_engine_early_stop_skips_second_proposer(identity_task):
    first = _ListProposer([(Op("identity"),)])
    second = _ListProposer([(Op("rot90"),)])
    SolveEngine([first, second], Budget(max_seconds=5), early_stop_admissible=1).solve(
        identity_task
    )
    assert first.called
    assert not second.called  # skipped once the first proposer already fit


def test_engine_deterministic_across_runs(recolor_task):
    engine = SolveEngine(
        [FittedProposer(), EnumerativeProposer()], Budget(max_seconds=5)
    )
    assert engine.solve(recolor_task).attempts == engine.solve(recolor_task).attempts
