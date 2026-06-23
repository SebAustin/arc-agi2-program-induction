"""Minimum Description Length prior over programs.

The description length of a program is the number of bits needed to encode it:
for each op, the cost of *naming* the primitive (a uniform code over the
registry) plus the primitive's intrinsic ``cost_bits`` plus the cost of encoding
its concrete parameters, with a terminating STOP symbol.

This is the Occam pressure at the heart of the method: among programs that fit
the demonstrations equally well, shorter ones receive exponentially more prior
mass and therefore generalise better (the Solomonoff/MDL argument).
"""

from __future__ import annotations

import math
from typing import Any

from arc_solver.dsl.program import Program
from arc_solver.dsl.registry import PRIMITIVES, op_selection_bits

_LN2 = math.log(2.0)
_STOP_BITS = 1.0
_COLOR_BITS = math.log2(10)  # 10 ARC colours


def _param_bits(param: Any) -> float:
    """Bits to encode one concrete parameter value."""
    if isinstance(param, bool):
        return 1.0
    if isinstance(param, int):
        return _COLOR_BITS
    if isinstance(param, str):
        return 2.0  # small enum (e.g. a direction)
    if isinstance(param, dict):
        return len(param) * (2 * _COLOR_BITS)
    if isinstance(param, (tuple, list)):
        return sum(_param_bits(p) for p in param)
    return _COLOR_BITS


def description_length(program: Program) -> float:
    """Total code length of a program in bits (lower = simpler = higher prior)."""
    bits = _STOP_BITS
    sel = op_selection_bits()
    for op in program:
        prim = PRIMITIVES.get(op.name)
        cost = prim.cost_bits if prim is not None else 8.0
        bits += sel + cost
        for p in op.params:
            bits += _param_bits(p)
    return bits


def prior_logp(program: Program) -> float:
    """Unnormalised log-prior: ``-ln(2) * description_length``."""
    return -_LN2 * description_length(program)
