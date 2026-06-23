"""Primitive registry and task context.

Every primitive maps a numpy grid to a numpy grid (or ``None`` when it does not
apply). Primitives advertise:

* ``cost_bits`` — its contribution to the description length (MDL prior).
* ``param_space`` — a function from a :class:`TaskContext` to the concrete
  parameter tuples worth trying. Conditioning the parameter space on the task
  (colours actually present, observed scale factors, ...) is what keeps
  enumeration tractable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from arc_solver.grid.grid_ops import color_counts, to_np
from arc_solver.io.task import Task

#: A primitive function: numpy grid (+ concrete params) -> numpy grid or None.
PrimFn = Callable[..., "np.ndarray | None"]
ParamSpace = Callable[["TaskContext"], Iterable[tuple]]


@dataclass(frozen=True)
class TaskContext:
    """Precomputed task statistics used to condition the parameter spaces."""

    colors: tuple[int, ...]  # colours present across all train inputs+outputs
    input_colors: tuple[int, ...]  # colours present in train inputs only
    scale_factors: tuple[tuple[int, int], ...]  # (out_h/in_h, out_w/in_w) seen
    same_shape: bool  # all train outputs share their input's shape

    @staticmethod
    def from_task(task: Task) -> "TaskContext":
        colors: set[int] = set()
        in_colors: set[int] = set()
        scales: set[tuple[int, int]] = set()
        same = True
        for pair in task.train:
            ai = to_np(pair.input)
            in_colors.update(color_counts(ai))
            colors.update(color_counts(ai))
            if pair.output is not None:
                ao = to_np(pair.output)
                colors.update(color_counts(ao))
                ih, iw = ai.shape
                oh, ow = ao.shape
                if (ih, iw) != (oh, ow):
                    same = False
                if ih and iw and oh % ih == 0 and ow % iw == 0:
                    scales.add((oh // ih, ow // iw))
        return TaskContext(
            colors=tuple(sorted(colors)),
            input_colors=tuple(sorted(in_colors)),
            scale_factors=tuple(sorted(scales)),
            same_shape=same,
        )


@dataclass(frozen=True)
class Primitive:
    name: str
    fn: PrimFn
    cost_bits: float
    param_space: ParamSpace


PRIMITIVES: dict[str, Primitive] = {}


def register(
    name: str, *, cost_bits: float = 3.0, param_space: ParamSpace | None = None
):
    """Decorator registering a primitive function under ``name``."""

    def _wrap(fn: PrimFn) -> PrimFn:
        if name in PRIMITIVES:
            raise ValueError(f"primitive already registered: {name}")
        ps: ParamSpace = param_space if param_space is not None else (lambda ctx: [()])
        PRIMITIVES[name] = Primitive(
            name=name, fn=fn, cost_bits=cost_bits, param_space=ps
        )
        return fn

    return _wrap


def op_selection_bits() -> float:
    """Uniform code length for choosing one primitive name (-log2(1/N))."""
    n = max(len(PRIMITIVES), 1)
    return math.log2(n)


def load_primitives() -> None:
    """Import every primitive module so the registry is fully populated.

    Importing the package is enough in normal use, but tests and the notebook
    call this to be explicit and order-independent.
    """
    from arc_solver.dsl import (  # noqa: F401  (import for side effects)
        primitives_geom,
        primitives_color,
        primitives_object,
        primitives_shape,
        primitives_arith,
    )
