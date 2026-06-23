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


def _output_mass(
    scored: list[ScoredProgram], test_input: Grid
) -> list[tuple[Grid, float]]:
    """Aggregate posterior weight by predicted output grid, heaviest first."""
    mass: dict[Grid, float] = defaultdict(float)
    for sp in scored:
        out = run(sp.program, test_input)
        if out is not None:
            mass[out] += sp.weight
    return sorted(mass.items(), key=lambda kv: kv[1], reverse=True)


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
    """A cheap second guess distinct from ``grid`` when no alternative exists.

    Uses the horizontal mirror; if the grid is mirror-symmetric (so the mirror
    equals the grid), falls back to a 180-degree rotation.
    """
    mirrored = tuple(tuple(reversed(row)) for row in grid)
    if mirrored != grid:
        return mirrored
    return tuple(tuple(reversed(row)) for row in reversed(grid))
