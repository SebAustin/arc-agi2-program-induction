"""Immutable task model.

A ``Grid`` is a tuple-of-tuples of ints 0-9. Tuples are chosen deliberately:
they are hashable, so predicted outputs can be used directly as dict keys when
the selector buckets candidates by output grid (see ``select.selector``), and
immutable, so a grid can never be mutated in place by a primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A grid is an immutable, hashable rectangular matrix of colour indices 0-9.
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Pair:
    """A single demonstration or test pair.

    ``output`` is ``None`` for test pairs whose answer is hidden.
    """

    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class Task:
    """An ARC task: a handful of train pairs and one or more test inputs."""

    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]

    @property
    def n_test(self) -> int:
        return len(self.test)


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a grid."""
    return (len(grid), len(grid[0]) if grid else 0)
