"""Integration tests: engine, harness, and submission writer."""

from __future__ import annotations

import json

import pytest

from arc_solver.config import Budget
from arc_solver.eval.harness import evaluate, pass_at_2
from arc_solver.io.submission import SubmissionWriter
from arc_solver.search.engine import SolveEngine
from arc_solver.search.enumerative import EnumerativeProposer
from arc_solver.search.fitted import FittedProposer


def _engine():
    return SolveEngine(
        [FittedProposer(), EnumerativeProposer()],
        Budget(max_seconds=5, max_programs=8000),
    )


def test_engine_requires_a_proposer():
    with pytest.raises(ValueError):
        SolveEngine([])


@pytest.mark.parametrize(
    "fixture", ["identity_task", "flip_task", "recolor_task", "tile_task"]
)
def test_engine_solves_trivial_tasks(fixture, request):
    task = request.getfixturevalue(fixture)
    result = _engine().solve(task)
    assert result.solved_train
    assert result.attempts[0][0] == task.test[0].output


def test_pass_at_2_either_attempt():
    truth = ((1, 2),)
    assert pass_at_2((((1, 2),), ((9, 9),)), truth)  # attempt_1 matches
    assert pass_at_2((((9, 9),), ((1, 2),)), truth)  # attempt_2 matches
    assert not pass_at_2((((9, 9),), ((8, 8),)), truth)
    assert not pass_at_2((((1, 2),), ((1, 2),)), None)


def test_evaluate_report(flip_task, recolor_task):
    report = evaluate(_engine(), [flip_task, recolor_task], dataset_name="tiny")
    assert report.n_tasks == 2
    assert report.pass_at_2 == 1.0  # both solvable
    assert "tiny" in report.to_markdown()


def test_submission_schema(flip_task):
    result = _engine().solve(flip_task)
    writer = SubmissionWriter()
    writer.add(flip_task.task_id, list(result.attempts))
    d = writer.to_dict()
    assert flip_task.task_id in d
    entry = d[flip_task.task_id][0]
    assert "attempt_1" in entry and "attempt_2" in entry
    assert isinstance(entry["attempt_1"], list)
    # serialises cleanly
    assert json.loads(json.dumps(d))


def test_submission_rejects_duplicate(flip_task):
    writer = SubmissionWriter()
    writer.add("x", [(((1,),), ((1,),))])
    with pytest.raises(ValueError):
        writer.add("x", [(((1,),), ((1,),))])
