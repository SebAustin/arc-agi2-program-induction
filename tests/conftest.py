"""Shared test fixtures: tiny synthetic tasks and helpers."""

from __future__ import annotations

import pytest

from arc_solver.io.task import Pair, Task


def make_task(train_pairs, test_pairs, task_id="t") -> Task:
    """Build a Task from ``[(in, out), ...]`` lists."""
    train = tuple(Pair(i, o) for i, o in train_pairs)
    test = tuple(Pair(i, o) for i, o in test_pairs)
    return Task(task_id=task_id, train=train, test=test)


@pytest.fixture
def identity_task() -> Task:
    return make_task(
        [(((1, 2), (3, 4)), ((1, 2), (3, 4))), (((5, 0), (0, 5)), ((5, 0), (0, 5)))],
        [(((7, 8), (9, 0)), ((7, 8), (9, 0)))],
    )


@pytest.fixture
def flip_task() -> Task:
    return make_task(
        [(((1, 2), (3, 4)), ((2, 1), (4, 3))), (((5, 6), (7, 8)), ((6, 5), (8, 7)))],
        [(((9, 0), (1, 2)), ((0, 9), (2, 1)))],
    )


@pytest.fixture
def recolor_task() -> Task:
    return make_task(
        [(((3, 0), (0, 3)), ((5, 0), (0, 5))), (((0, 3), (3, 3)), ((0, 5), (5, 5)))],
        [(((3, 3), (0, 0)), ((5, 5), (0, 0)))],
    )


@pytest.fixture
def tile_task() -> Task:
    return make_task(
        [(((1, 2),), ((1, 2, 1, 2), (1, 2, 1, 2)))],
        [(((3, 4),), ((3, 4, 3, 4), (3, 4, 3, 4)))],
    )
