"""The unifying proposal interface.

Every search backend — offline enumerative search, an LLM emitting programs, or a
fine-tuned model — implements :class:`Proposer`. A proposer yields candidate
``Program``s; the engine verifies and selects. The *proposal distribution* is the
prior: the enumerator follows the MDL prior directly, while an LLM is an
*amortised* approximation of the same posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

from arc_solver.config import Budget
from arc_solver.dsl.program import Program
from arc_solver.io.task import Grid, Task


@dataclass(frozen=True)
class Feedback:
    """Execution-grounded feedback for the refinement loop.

    Carries the best near-miss program seen so far together with its per-train
    predictions, so a proposer can re-propose conditioned on what went wrong.
    """

    program: Program
    train_predictions: tuple[Grid | None, ...]
    score: float


class Proposer(Protocol):
    """Yields candidate programs, best (cheapest/most-likely) first."""

    def propose(
        self, task: Task, *, budget: Budget, feedback: Feedback | None = None
    ) -> Iterator[Program]: ...
