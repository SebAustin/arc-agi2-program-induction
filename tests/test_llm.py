"""Tests for the LLM proposer using a MockClient (no network)."""

from __future__ import annotations

from arc_solver.config import Budget
from arc_solver.search.llm_client import MockClient
from arc_solver.search.llm_proposer import LLMProposer
from arc_solver.search.prompt import extract_code, render_task
from arc_solver.search.verifier import consistent
from conftest import make_task

_TRANSPOSE = (
    "Here's the rule:\n```python\n"
    "def transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"
)


def test_extract_code_from_fence():
    assert extract_code("```python\ndef transform(grid):\n    return grid\n```")
    assert extract_code("no code here") is None


def test_render_task_includes_grids(flip_task):
    text = render_task(flip_task)
    assert "Example 1 input" in text
    assert "def transform" in text


def test_llm_proposer_solves_with_mock():
    task = make_task(
        [(((1, 2), (3, 4)), ((1, 3), (2, 4)))],  # transpose
        [(((5, 6), (7, 8)), ((5, 7), (6, 8)))],
    )
    proposer = LLMProposer(MockClient([_TRANSPOSE]), samples_per_round=1)
    programs = list(proposer.propose(task, budget=Budget(max_refine_rounds=1)))
    assert any(consistent(p, task) for p in programs)


def test_llm_proposer_refines_on_failure():
    # First completion is wrong, second (refinement round) is correct.
    wrong = "```python\ndef transform(grid):\n    return grid\n```"
    task = make_task(
        [(((1, 2), (3, 4)), ((1, 3), (2, 4)))],
        [(((5, 6), (7, 8)), ((5, 7), (6, 8)))],
    )
    client = MockClient([wrong])  # round 0 wrong; round 1 reuses same -> still wrong
    proposer = LLMProposer(client, samples_per_round=1)
    list(proposer.propose(task, budget=Budget(max_refine_rounds=1)))
    # The proposer must have issued a second (refinement) call.
    assert len(client.calls) == 2
