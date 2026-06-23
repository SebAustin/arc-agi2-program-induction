"""Kaggle submission writer.

The ARC-AGI-2 competition expects a ``submission.json`` mapping every evaluation
``task_id`` to a list (one entry per test input) of ``{"attempt_1", "attempt_2"}``
objects, where each attempt is a grid (list of lists of ints). Scoring is pass@2:
a test input counts as solved if *either* attempt matches the hidden truth.

Every task **must** be present, so callers should always supply a fallback
(e.g. the identity grid) for tasks the solver could not crack in time.
"""

from __future__ import annotations

import json

from arc_solver.io.task import Grid


def _grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


class SubmissionWriter:
    """Accumulates per-task attempts and serialises the competition format."""

    def __init__(self) -> None:
        self._entries: dict[str, list[dict[str, list[list[int]]]]] = {}

    def add(self, task_id: str, attempts: list[tuple[Grid, Grid]]) -> None:
        """Record the (attempt_1, attempt_2) pairs for each test input of a task."""
        if task_id in self._entries:
            raise ValueError(f"duplicate task_id in submission: {task_id}")
        self._entries[task_id] = [
            {"attempt_1": _grid_to_lists(a1), "attempt_2": _grid_to_lists(a2)}
            for (a1, a2) in attempts
        ]

    def to_dict(self) -> dict[str, list[dict[str, list[list[int]]]]]:
        return dict(self._entries)

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    def __len__(self) -> int:
        return len(self._entries)
