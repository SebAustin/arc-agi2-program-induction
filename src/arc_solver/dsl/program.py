"""Program representation.

A program is a *linear pipeline* of typed ops, applied left to right. Linearity
(rather than a full AST) keeps enumeration tractable and the description length
trivial to compute, while still composing — objectness is handled inside
primitives (e.g. ``keep_largest``) rather than via a lambda calculus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Op:
    """A single named operation with concrete (hashable) parameters."""

    name: str
    params: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        # Guarantee hashability so Programs can live in sets / dict keys.
        hash(self.params)


#: A program is an immutable sequence of ops.
Program = tuple[Op, ...]


def _param_repr(p: Any) -> str:
    if isinstance(p, tuple):
        return "[" + ",".join(_param_repr(x) for x in p) + "]"
    return str(p)


def pretty(program: Program) -> str:
    """Human-readable one-line rendering, e.g. ``rotate(90) | recolor(1,2)``."""
    if not program:
        return "<identity>"
    parts = []
    for op in program:
        if op.params:
            parts.append(f"{op.name}({','.join(_param_repr(p) for p in op.params)})")
        else:
            parts.append(op.name)
    return " | ".join(parts)


def serialize(program: Program) -> str:
    """Canonical JSON serialisation (round-trips with ``deserialize``).

    Parameters are plain values or (possibly nested) tuples — JSON renders tuples
    as arrays, and ``deserialize`` converts arrays back into tuples so every Op
    remains hashable.
    """
    payload = [[op.name, list(op.params)] for op in program]
    return json.dumps(payload, separators=(",", ":"))


def _to_hashable(value: Any) -> Any:
    """Recursively convert JSON lists into tuples so params stay hashable."""
    if isinstance(value, list):
        return tuple(_to_hashable(v) for v in value)
    return value


def deserialize(text: str) -> Program:
    """Inverse of ``serialize``."""
    payload = json.loads(text)
    return tuple(
        Op(name=name, params=tuple(_to_hashable(p) for p in params))
        for name, params in payload
    )
