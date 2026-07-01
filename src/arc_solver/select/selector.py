"""Pass@2-aware candidate selection.

Given admissible programs (each consistent with all train pairs) and weighted by
the posterior, we choose two output grids per test input to maximise expected
pass@2:

* **attempt_1** = the posterior *mode* over outputs — aggregate the weight of all
  programs that map the test input to the same grid, and take the heaviest grid.
* **attempt_2** = the next-heaviest grid that *disagrees* with attempt_1. Because
  pass@2 rewards covering two distinct hypotheses, the second attempt hedges
  toward the most probable alternative rather than restating the first.

This is the metric-aware core of the method: posterior mode + maximum-marginal
disagreement.
"""

from __future__ import annotations

from collections import defaultdict

from arc_solver.dsl.interpreter import run
from arc_solver.io.task import Grid
from arc_solver.select.posterior import ScoredProgram

_NUM_COLORS = 10  # ARC palette is colours 0-9


def _output_mass(
    scored: list[ScoredProgram], test_input: Grid
) -> list[tuple[Grid, float]]:
    """Aggregate posterior weight by predicted output grid, heaviest first."""
    mass: dict[Grid, float] = defaultdict(float)
    for sp in scored:
        out = run(sp.program, test_input)
        if out is not None:
            mass[out] += sp.weight
    # Heaviest first; ties broken by the canonical (lexicographically smallest)
    # grid so the posterior *mode* is a well-defined function of the posterior
    # mass alone, not of proposer/enumeration order (reproducibility).
    return sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))


def pick_for_input(scored: list[ScoredProgram], test_input: Grid) -> tuple[Grid, Grid]:
    """Choose (attempt_1, attempt_2) for a single test input."""
    ranked = _output_mass(scored, test_input)
    if not ranked:
        # No admissible program produced an output: fall back to identity, and
        # a trivial perturbation so the two attempts differ.
        return test_input, _fallback_alt(test_input)
    attempt_1 = ranked[0][0]
    attempt_2 = next((g for g, _ in ranked[1:] if g != attempt_1), None)
    if attempt_2 is None:
        attempt_2 = _fallback_alt(attempt_1)
    return attempt_1, attempt_2


def pick_top2(
    scored: list[ScoredProgram], test_inputs: list[Grid]
) -> list[tuple[Grid, Grid]]:
    """Choose (attempt_1, attempt_2) for each test input of a task."""
    return [pick_for_input(scored, x) for x in test_inputs]


def _fallback_alt(grid: Grid) -> Grid:
    """A cheap second guess *guaranteed distinct* from ``grid``.

    Tries a ladder of structure-preserving transforms (horizontal mirror,
    vertical mirror, 180-degree rotation) and returns the first that differs, so
    the hedge stays meaningful for ordinary grids. A grid invariant under all of
    them is constant-under-symmetry (e.g. a solid colour or a palindrome that is
    also row-order-symmetric); for that degenerate case we deterministically
    perturb the top-left cell so ``attempt_2`` never collapses onto ``attempt_1``
    and silently waste a pass@2 guess.
    """
    flip_h = tuple(tuple(reversed(row)) for row in grid)
    flip_v = tuple(reversed(grid))
    rot180 = tuple(tuple(reversed(row)) for row in reversed(grid))
    for candidate in (flip_h, flip_v, rot180):
        if candidate != grid:
            return candidate
    # Fully symmetric grid: no transform escapes it. Perturb one cell.
    rows = [list(row) for row in grid]
    rows[0][0] = (rows[0][0] + 1) % _NUM_COLORS
    return tuple(tuple(row) for row in rows)
