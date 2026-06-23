"""The solve engine: proposers -> verify -> posterior -> select.

The engine runs each proposer in priority order (offline enumerative first — it
is always available and needs no network; the LLM proposer second, if
configured). It collects programs that are *admissible* (consistent with every
train pair), de-duplicates them by their behaviour on the train inputs (keeping
the cheapest representative), weights the survivors by the posterior, and asks
the selector for two candidates per test input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from arc_solver.config import Budget
from arc_solver.dsl.interpreter import run_batch
from arc_solver.dsl.program import Program
from arc_solver.search.mdl import description_length
from arc_solver.search.proposer import Proposer
from arc_solver.search.verifier import consistent
from arc_solver.select.posterior import ScoredProgram, posterior_weights
from arc_solver.select.selector import pick_top2
from arc_solver.io.task import Grid, Task

_MAX_ADMISSIBLE = 64  # enough alternatives for a diverse attempt_2


@dataclass(frozen=True)
class SolveResult:
    """Outcome of solving a single task."""

    task_id: str
    scored: tuple[ScoredProgram, ...]  # admissible programs, posterior-weighted
    attempts: tuple[tuple[Grid, Grid], ...]  # (attempt_1, attempt_2) per test input
    solved_train: bool  # at least one admissible program found
    meta: dict = field(default_factory=dict)

    @property
    def best_program(self) -> Program | None:
        return self.scored[0].program if self.scored else None


class SolveEngine:
    def __init__(
        self,
        proposers: Sequence[Proposer],
        budget: Budget | None = None,
        *,
        early_stop_admissible: int = 1,
    ) -> None:
        if not proposers:
            raise ValueError("SolveEngine needs at least one proposer")
        self.proposers = list(proposers)
        self.budget = budget or Budget()
        # Once a (cheap, earlier) proposer yields at least this many admissible
        # programs, later (expensive) proposers are skipped for this task.
        self.early_stop_admissible = early_stop_admissible

    def solve(self, task: Task) -> SolveResult:
        start = time.monotonic()
        train_inputs: list[Grid] = [p.input for p in task.train]
        # signature -> (cheapest_bits, program) keeps one program per behaviour.
        admissible: dict[tuple, tuple[float, Program]] = {}
        n_proposed = 0

        for proposer in self.proposers:
            for program in proposer.propose(task, budget=self.budget):
                n_proposed += 1
                if not consistent(program, task):
                    continue
                sig = tuple(run_batch(program, train_inputs))
                bits = description_length(program)
                prev = admissible.get(sig)
                if prev is None or bits < prev[0]:
                    admissible[sig] = (bits, program)
                if len(admissible) >= _MAX_ADMISSIBLE:
                    break
            if len(admissible) >= _MAX_ADMISSIBLE:
                break
            # Skip remaining (more expensive) proposers if we already have a fit.
            if len(admissible) >= self.early_stop_admissible:
                break

        programs = [prog for _, prog in admissible.values()]
        scored = posterior_weights(programs)
        scored.sort(key=lambda sp: sp.weight, reverse=True)
        test_inputs = [p.input for p in task.test]
        attempts = pick_top2(scored, test_inputs)

        return SolveResult(
            task_id=task.task_id,
            scored=tuple(scored),
            attempts=tuple(attempts),
            solved_train=bool(programs),
            meta={
                "n_proposed": n_proposed,
                "n_admissible": len(programs),
                "seconds": round(time.monotonic() - start, 3),
            },
        )
