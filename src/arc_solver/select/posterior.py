"""Posterior over admissible programs.

Every program here has already passed verification (it reproduces all train
pairs), so the likelihood is 1 and the posterior is proportional to the MDL
prior. Weights are a softmax over ``prior_logp`` — shorter programs carry
exponentially more mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arc_solver.dsl.program import Program
from arc_solver.search.mdl import prior_logp


@dataclass(frozen=True)
class ScoredProgram:
    """An admissible program with its (unnormalised) log-posterior and weight."""

    program: Program
    logp: float
    weight: float


def posterior_weights(programs: list[Program]) -> list[ScoredProgram]:
    """Compute normalised posterior weights for admissible programs.

    Returns an empty list for empty input. Weights sum to 1.
    """
    if not programs:
        return []
    logps = [prior_logp(p) for p in programs]
    m = max(logps)
    exps = [math.exp(lp - m) for lp in logps]
    z = sum(exps)
    return [
        ScoredProgram(program=p, logp=lp, weight=e / z)
        for p, lp, e in zip(programs, logps, exps)
    ]
