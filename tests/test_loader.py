"""Tests for task loading and strict grid validation."""

from __future__ import annotations

import json

import pytest

from arc_solver.io.loader import load_dataset, load_task, validate_grid


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
