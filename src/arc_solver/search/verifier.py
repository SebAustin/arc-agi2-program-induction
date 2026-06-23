"""Program verification against the demonstrations.

A program is *admissible* only if it reproduces **every** train output exactly.
This is the likelihood term of the posterior: it is 1 for admissible programs
and 0 otherwise, so the MDL prior alone ranks the survivors. ``partial_score``
is a soft signal (fraction of pairs solved + cell agreement) used **only** to
steer refinement — never to admit an overfit program.
"""

from __future__ import annotations

from arc_solver.dsl.interpreter import run
from arc_solver.dsl.program import Program
from arc_solver.grid.metrics import cell_agreement, grid_equal
from arc_solver.io.task import Task


def consistent(program: Program, task: Task) -> bool:
    """True iff the program maps every train input to its exact output."""
    for pair in task.train:
        if pair.output is None:
            continue
        if not grid_equal(run(program, pair.input), pair.output):
            return False
    return True


def partial_score(program: Program, task: Task) -> float:
    """Soft fit in [0, 1]: mean over train pairs of exact-or-cell agreement.

    Exact matches score 1.0; otherwise the per-pair cell agreement (when shapes
    line up) contributes partially. Used to rank near-misses during refinement.
    """
    pairs = [p for p in task.train if p.output is not None]
    if not pairs:
        return 0.0
    total = 0.0
    for pair in pairs:
        pred = run(program, pair.input)
        if grid_equal(pred, pair.output):
            total += 1.0
        else:
            total += cell_agreement(pred, pair.output)
    return total / len(pairs)
