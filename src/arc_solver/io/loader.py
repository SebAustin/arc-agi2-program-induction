"""Loading and strict validation of ARC task JSON.

Never trust external data: every grid is validated at the boundary (rectangular,
ints in 0-9, non-empty) and a clear ``ValueError`` is raised on malformed input.
The same loader normalises both ARC-AGI-1 and ARC-AGI-2 files to the ``Task``
type, since the two datasets share the same on-disk schema.
"""

from __future__ import annotations

import json
import os
from glob import glob

from arc_solver.io.task import Grid, Pair, Task

_MIN_COLOR = 0
_MAX_COLOR = 9


def validate_grid(raw: object) -> Grid:
    """Validate and freeze a raw grid into an immutable ``Grid``.

    Raises:
        ValueError: if ``raw`` is not a rectangular list of lists of ints 0-9.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("grid must be a non-empty list of rows")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for r, row in enumerate(raw):
        if not isinstance(row, list) or not row:
            raise ValueError(f"row {r} must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"row {r} has width {len(row)}, expected {width}")
        cells: list[int] = []
        for c, val in enumerate(row):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"cell ({r},{c}) is not an int: {val!r}")
            if not (_MIN_COLOR <= val <= _MAX_COLOR):
                raise ValueError(f"cell ({r},{c}) colour {val} out of range 0-9")
            cells.append(val)
        rows.append(tuple(cells))
    return tuple(rows)


def _parse_pair(raw: dict, *, require_output: bool) -> Pair:
    if "input" not in raw:
        raise ValueError("pair missing 'input'")
    grid_in = validate_grid(raw["input"])
    grid_out: Grid | None = None
    if raw.get("output") is not None:
        grid_out = validate_grid(raw["output"])
    elif require_output:
        raise ValueError("train pair missing 'output'")
    return Pair(input=grid_in, output=grid_out)


def _parse_task_entry(task_id: str, raw: object) -> Task:
    if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
        raise ValueError(f"task {task_id!r} must have 'train' and 'test'")
    train_raw = raw["train"]
    test_raw = raw["test"]
    if not isinstance(train_raw, list) or not isinstance(test_raw, list):
        raise ValueError(f"task {task_id!r}: 'train' and 'test' must be lists")
    train = tuple(_parse_pair(p, require_output=True) for p in train_raw)
    test = tuple(_parse_pair(p, require_output=False) for p in test_raw)
    if not train:
        raise ValueError(f"task {task_id!r} has no train pairs")
    if not test:
        raise ValueError(f"task {task_id!r} has no test pairs")
    return Task(task_id=task_id, train=train, test=test)


def load_task(path: str) -> Task:
    """Load and validate a single task JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    task_id = os.path.splitext(os.path.basename(path))[0]
    try:
        return _parse_task_entry(task_id, raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_dataset(directory: str) -> list[Task]:
    """Load every ``*.json`` task in a directory, sorted by id for determinism."""
    paths = sorted(glob(os.path.join(directory, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no task JSON found in {directory}")
    return [load_task(p) for p in paths]


def load_challenges(path: str) -> list[Task]:
    """Load the competition's combined-challenges JSON into a list of ``Task``.

    This is the format Kaggle mounts for ARC Prize 2026 — a single JSON file
    shaped ``{"<task_id>": {"train": [...], "test": [...]}}`` rather than one
    file per task. Test pairs never carry an ``"output"`` key (the answer is
    hidden at scoring time), so it is never required here; train pairs still
    require one. Tasks are returned sorted by ``task_id`` for determinism.

    Raises:
        ValueError: if the top-level JSON is not an object, or any task entry
            is malformed (missing train/test, empty, or fails grid validation).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: combined challenges file must be a JSON object")
    if not raw:
        raise ValueError(f"{path}: combined challenges file has no tasks")
    tasks: list[Task] = []
    for task_id in sorted(raw):
        try:
            tasks.append(_parse_task_entry(task_id, raw[task_id]))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return tasks
