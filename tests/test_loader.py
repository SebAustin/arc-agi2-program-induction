"""Tests for task loading and strict grid validation."""

from __future__ import annotations

import json

import pytest

from arc_solver.io.loader import (
    load_challenges,
    load_dataset,
    load_task,
    validate_grid,
)
from arc_solver.io.submission import SubmissionWriter


def test_validate_grid_accepts_valid():
    assert validate_grid([[0, 1], [2, 3]]) == ((0, 1), (2, 3))


@pytest.mark.parametrize(
    "bad",
    [
        [],  # empty
        [[]],  # empty row
        [[1, 2], [3]],  # ragged
        [[1, 10]],  # colour out of range
        [[1, -1]],  # negative
        [[1, True]],  # bool not int
        [[1, "2"]],  # str
        "notalist",
    ],
)
def test_validate_grid_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_grid(bad)


def test_load_task_round_trip(tmp_path):
    payload = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }
    p = tmp_path / "abc123.json"
    p.write_text(json.dumps(payload))
    task = load_task(str(p))
    assert task.task_id == "abc123"
    assert task.train[0].input == ((1, 2),)
    assert task.train[0].output == ((2, 1),)
    assert task.test[0].output is None


def test_load_task_missing_output_in_train(tmp_path):
    payload = {"train": [{"input": [[1]]}], "test": [{"input": [[1]]}]}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_task(str(p))


def test_load_dataset_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path))


def test_load_challenges_parses_combined_file_with_hidden_test_output(tmp_path):
    payload = {
        "b222": {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}],
        },
        "a111": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[5]]}, {"input": [[6]]}],
        },
    }
    p = tmp_path / "arc-agi_test_challenges.json"
    p.write_text(json.dumps(payload))

    tasks = load_challenges(str(p))

    # Sorted by task_id for determinism, regardless of JSON key order.
    assert [t.task_id for t in tasks] == ["a111", "b222"]

    a111 = tasks[0]
    assert a111.train[0].input == ((0,),)
    assert a111.train[0].output == ((1,),)
    assert len(a111.test) == 2
    # Hidden test pairs must not require (or invent) an output.
    assert all(pair.output is None for pair in a111.test)
    assert a111.test[0].input == ((5,),)
    assert a111.test[1].input == ((6,),)

    b222 = tasks[1]
    assert b222.test[0].output is None
    assert b222.test[0].input == ((3, 4),)


def test_load_challenges_is_deterministic_regardless_of_key_order(tmp_path):
    task_a = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}
    task_b = {"train": [{"input": [[2]], "output": [[2]]}], "test": [{"input": [[2]]}]}

    p1 = tmp_path / "order1.json"
    p1.write_text(json.dumps({"zzz": task_b, "aaa": task_a}))
    p2 = tmp_path / "order2.json"
    p2.write_text(json.dumps({"aaa": task_a, "zzz": task_b}))

    tasks1 = load_challenges(str(p1))
    tasks2 = load_challenges(str(p2))

    ids1 = [t.task_id for t in tasks1]
    ids2 = [t.task_id for t in tasks2]
    assert ids1 == ids2 == ["aaa", "zzz"]


def test_load_challenges_rejects_non_object_top_level(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_challenges(str(p))


def test_load_challenges_rejects_empty_object(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({}))
    with pytest.raises(ValueError):
        load_challenges(str(p))


def test_load_challenges_rejects_task_missing_train_or_test(tmp_path):
    payload = {"only_train": {"train": [{"input": [[1]], "output": [[1]]}]}}
    p = tmp_path / "bad_task.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_challenges(str(p))


def test_load_challenges_rejects_train_pair_missing_output(tmp_path):
    payload = {
        "t1": {
            "train": [{"input": [[1]]}],  # missing output: invalid for train
            "test": [{"input": [[1]]}],
        }
    }
    p = tmp_path / "bad_train.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_challenges(str(p))


def test_load_challenges_rejects_malformed_grid(tmp_path):
    payload = {
        "t1": {
            "train": [{"input": [[1, 10]], "output": [[1]]}],  # colour out of range
            "test": [{"input": [[1]]}],
        }
    }
    p = tmp_path / "bad_grid.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_challenges(str(p))


def test_submission_writer_round_trips_two_test_input_schema(tmp_path):
    """SubmissionWriter output must match the exact competition schema:

    ``{"<task_id>": [{"attempt_1": grid, "attempt_2": grid}, ...]}`` with one
    entry per test input, for a task with two (hidden) test inputs loaded via
    ``load_challenges``.
    """
    payload = {
        "multi_test_task": {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}, {"input": [[5, 6]]}],
        }
    }
    p = tmp_path / "combined.json"
    p.write_text(json.dumps(payload))

    tasks = load_challenges(str(p))
    task = tasks[0]
    assert task.n_test == 2

    writer = SubmissionWriter()
    attempts = [(pair.input, pair.input) for pair in task.test]
    writer.add(task.task_id, attempts)

    out_path = tmp_path / "submission.json"
    writer.write(str(out_path))

    with open(out_path, encoding="utf-8") as fh:
        submission = json.load(fh)

    assert set(submission.keys()) == {"multi_test_task"}
    entries = submission["multi_test_task"]
    assert len(entries) == 2  # one object per test input
    for entry, pair in zip(entries, task.test):
        assert set(entry.keys()) == {"attempt_1", "attempt_2"}
        expected_grid = [list(row) for row in pair.input]
        assert entry["attempt_1"] == expected_grid
        assert entry["attempt_2"] == expected_grid
        # Every value is a list of lists of ints — the exact grid schema.
        assert all(isinstance(row, list) for row in entry["attempt_1"])
        assert all(isinstance(v, int) for row in entry["attempt_1"] for v in row)
