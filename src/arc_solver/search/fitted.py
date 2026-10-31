"""Fitted proposer — analytic MAP shortcuts to the posterior.

Rather than search blindly, this proposer *reads the demonstrations* and emits a
small set of high-probability program hypotheses directly:

* every parameter-free geometric / shape op (the single-transform hypotheses);
* a full colour remapping fitted from a consistent cell-wise colour bijection;
* tiling / scaling fitted from the input/output dimension ratio;
* a constant-output program when all demonstrations share one output;
* symmetry repair for each candidate noise colour;
* a handful of depth-2 ``transform ∘ recolour`` compositions.

Each proposal is cheap to construct and is *verified* downstream, so wrong
guesses are harmless. This is the amortised analytic counterpart to the
enumerative proposer and accounts for most offline solves.
"""

from __future__ import annotations

from typing import Iterator

from arc_solver.config import Budget
from arc_solver.dsl.program import Op, Program
from arc_solver.dsl.registry import load_primitives
from arc_solver.grid.grid_ops import to_np
from arc_solver.search.fit_panels import fit_panels
from arc_solver.search.fit_recolor import fit_object_recolor
from arc_solver.search.proposer import Feedback
from arc_solver.io.task import Task

_GEOM = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)
_PARAMFREE_SHAPE = (
    "crop_to_content",
    "mirror_concat_h",
    "mirror_concat_v",
    "trim_border",
)


def _fit_color_map(task: Task) -> dict | None:
    """Infer a consistent input->output colour bijection, if one exists.

    Requires every train pair to share its input shape and to define a single,
    conflict-free colour map. Returns the non-trivial mapping or ``None``.
    """
    mapping: dict[int, int] = {}
    for pair in task.train:
        if pair.output is None:
            return None
        ai, ao = to_np(pair.input), to_np(pair.output)
        if ai.shape != ao.shape:
            return None
        for ci, co in zip(ai.flatten().tolist(), ao.flatten().tolist()):
            if ci in mapping:
                if mapping[ci] != co:
                    return None  # conflicting mapping -> not a pure recolour
            else:
                mapping[ci] = co
    nontrivial = {k: v for k, v in mapping.items() if k != v}
    return nontrivial or None


def _fit_scale(task: Task) -> tuple[int, int] | None:
    """Infer a consistent integer (ky, kx) output/input size ratio."""
    factor: tuple[int, int] | None = None
    for pair in task.train:
        if pair.output is None or not pair.input or not pair.output:
            return None
        ih, iw = len(pair.input), len(pair.input[0])
        oh, ow = len(pair.output), len(pair.output[0])
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        f = (oh // ih, ow // iw)
        if factor is None:
            factor = f
        elif factor != f:
            return None
    if factor in (None, (1, 1)):
        return None
    return factor


def _const_output(task: Task) -> tuple | None:
    """Return the single shared output grid, if all demos agree."""
    outs = {p.output for p in task.train if p.output is not None}
    return next(iter(outs)) if len(outs) == 1 else None


def _input_colors(task: Task) -> list[int]:
    cols: set[int] = set()
    for pair in task.train:
        cols.update(int(v) for v in to_np(pair.input).flatten())
    return sorted(cols)


class FittedProposer:
    """Emits analytically-fitted program hypotheses, simplest first."""

    def __init__(self) -> None:
        load_primitives()

    def propose(
        self, task: Task, *, budget: Budget, feedback: Feedback | None = None
    ) -> Iterator[Program]:
        # 1) single-transform hypotheses
        geom_ops = [Op(name) for name in _GEOM]
        for op in geom_ops:
            yield (op,)
        for name in _PARAMFREE_SHAPE:
            yield (Op(name),)

        # 2) fitted colour remap (and its compositions with geometry)
        cmap = _fit_color_map(task)
        color_op: Op | None = None
        if cmap is not None:
            pairs = tuple(sorted(cmap.items()))
            color_op = Op("color_map", (pairs,))
            yield (color_op,)
            for gop in geom_ops:
                if gop.name == "identity":
                    continue
                yield (gop, color_op)  # transform then recolour
                yield (color_op, gop)  # recolour then transform
            yield (Op("crop_to_content"), color_op)

        # 3) fitted tiling / scaling
        factor = _fit_scale(task)
        if factor is not None:
            yield (Op("tile", factor),)
            yield (Op("scale", factor),)

        # 4) symmetry repair, one hypothesis per candidate noise colour
        for c in _input_colors(task):
            yield (Op("repair_symmetry", (c,)),)

        # 5) constant output
        const = _const_output(task)
        if const is not None:
            yield (Op("const_grid", (const,)),)

        # 6) grid-partition / panel ops (F2)
        yield from fit_panels(task)

        # 7) per-object recolour / transform (F1)
        yield from fit_object_recolor(task)
        # Note: an object selection/counting family (F4) was built and measured
        # (search/fit_select.py, primitives select_object/count_grid) but added 0
        # solves on ARC-1 eval, so it is not emitted here. See SPEC.md.
