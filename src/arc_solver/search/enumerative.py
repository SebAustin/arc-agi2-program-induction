"""Offline enumerative proposer — the reproducible CPU core.

Best-first search over DSL pipelines, expanded in increasing description-length
(MDL) order, so the shortest hypotheses are proposed first. Two ideas keep the
combinatorics in check:

1. **Task-conditioned grounding** — each primitive is grounded only with the
   parameter values its ``param_space`` deems relevant to *this* task.
2. **Behavioural de-duplication** — programs are keyed by the tuple of outputs
   they produce on the train inputs. Two programs with identical train behaviour
   are interchangeable, so only the first (cheapest) is kept and expanded. This
   collapses the search from 'all programs' to 'all distinct behaviours'.

No network, no randomness: identical inputs yield identical proposals.
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Iterator

from arc_solver.config import Budget
from arc_solver.dsl.interpreter import run_batch
from arc_solver.dsl.program import Op, Program
from arc_solver.dsl.registry import PRIMITIVES, TaskContext, load_primitives
from arc_solver.search.mdl import description_length
from arc_solver.search.proposer import Feedback
from arc_solver.io.task import Grid, Task


def _grounded_ops(ctx: TaskContext) -> list[Op]:
    """All primitive/parameter combinations relevant to the task."""
    ops: list[Op] = []
    for prim in PRIMITIVES.values():
        for params in prim.param_space(ctx):
            ops.append(Op(name=prim.name, params=tuple(params)))
    return ops


class EnumerativeProposer:
    """MDL-ordered best-first enumerator over the DSL."""

    def __init__(self) -> None:
        load_primitives()

    def propose(
        self, task: Task, *, budget: Budget, feedback: Feedback | None = None
    ) -> Iterator[Program]:
        ctx = TaskContext.from_task(task)
        grounded = _grounded_ops(ctx)
        train_inputs: list[Grid] = [p.input for p in task.train]

        start = time.monotonic()
        counter = itertools.count()  # tiebreaker to keep heap entries comparable
        # Frontier holds (bits, tiebreak, program). Seed with the identity program.
        frontier: list[tuple[float, int, Program]] = [(0.0, next(counter), tuple())]
        seen_signatures: set[tuple] = set()
        popped = 0

        while frontier and popped < budget.max_programs:
            if time.monotonic() - start > budget.max_seconds:
                return
            bits, _, program = heapq.heappop(frontier)
            popped += 1

            outputs = run_batch(program, train_inputs)
            signature = tuple(outputs)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            # Identity (empty program) has an all-inputs signature; still a valid
            # candidate worth yielding for tasks whose answer equals the input.
            yield program

            # Only extend programs that apply to *every* train input — a None
            # anywhere can never be repaired by appending more ops.
            if len(program) >= budget.max_depth or any(o is None for o in outputs):
                continue
            for op in grounded:
                child = program + (op,)
                heapq.heappush(
                    frontier, (description_length(child), next(counter), child)
                )
