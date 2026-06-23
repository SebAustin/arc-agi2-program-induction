"""LLM-amortised proposer with execution-grounded refinement.

The model is an *amortised approximation of the posterior*: it proposes programs
the search would otherwise have to enumerate. Proposals are sandboxed Python
``transform`` functions wrapped as ``py_transform`` ops, so they are verified and
weighted exactly like symbolic programs. When a proposal fails some train pairs,
the proposer feeds the failures back and asks for a correction (the Greenblatt /
Imbue refinement loop), for up to ``budget.max_refine_rounds`` rounds.

The proposer needs an :class:`LLMClient`; tests pass a ``MockClient`` so the
whole loop runs without any network access.
"""

from __future__ import annotations

from typing import Iterator

from arc_solver.config import Budget
from arc_solver.dsl.interpreter import run
from arc_solver.dsl.program import Op, Program
from arc_solver.search import prompt as P
from arc_solver.search.llm_client import LLMClient
from arc_solver.search.proposer import Feedback
from arc_solver.search.verifier import consistent
from arc_solver.io.task import Task

# Importing registers the py_transform primitive (kept out of the offline core).
import arc_solver.dsl.primitives_llm  # noqa: F401,E402


class LLMProposer:
    def __init__(
        self, client: LLMClient, *, samples_per_round: int = 4, temperature: float = 0.7
    ) -> None:
        self.client = client
        self.samples = samples_per_round
        self.temperature = temperature

    def _program(self, source: str) -> Program:
        return (Op("py_transform", (source,)),)

    def _failure_notes(self, source: str, task: Task) -> list[str]:
        notes: list[str] = []
        prog = self._program(source)
        for i, pair in enumerate(task.train):
            if pair.output is None:
                continue
            got = run(prog, pair.input)
            if got != pair.output:
                notes.append(f"- example {i + 1}: expected {pair.output}, got {got}")
        return notes

    def propose(
        self, task: Task, *, budget: Budget, feedback: Feedback | None = None
    ) -> Iterator[Program]:
        user = P.render_task(task)
        best_source: str | None = None
        rounds = max(1, budget.max_refine_rounds + 1)

        for round_idx in range(rounds):
            if round_idx == 0:
                completions = self.client.complete(
                    P.SYSTEM, user, n=self.samples, temperature=self.temperature
                )
            else:
                if best_source is None:
                    break
                fb = P.render_feedback(
                    best_source, self._failure_notes(best_source, task)
                )
                completions = self.client.complete(
                    P.SYSTEM, fb, n=self.samples, temperature=self.temperature
                )

            solved = False
            for completion in completions:
                source = P.extract_code(completion)
                if not source:
                    continue
                program = self._program(source)
                yield program
                if consistent(program, task):
                    solved = True
                elif best_source is None:
                    best_source = source  # keep a near-miss to refine from
            if solved:
                return
