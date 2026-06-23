"""The ``py_transform`` primitive — a bridge for LLM-emitted programs.

This wraps a sandboxed Python ``transform(grid)`` as a single DSL op, so an
LLM-authored solution flows through the *same* verifier, posterior, and selector
as a symbolic program. Its description length is large (it encodes whole source
code), so whenever a short symbolic program is also admissible the posterior
prefers the symbolic one — exactly the desired Occam behaviour.

This module is imported only by :class:`LLMProposer`; the offline core never
registers ``py_transform``.
"""

from __future__ import annotations

import numpy as np

from arc_solver.dsl.registry import register
from arc_solver.dsl.sandbox import SandboxError, run_transform
from arc_solver.grid.grid_ops import to_grid, to_np


@register("py_transform", cost_bits=64.0, param_space=lambda ctx: [])
def py_transform(arr: np.ndarray, source: str) -> np.ndarray | None:
    """Run an LLM-emitted ``transform`` on the grid via the sandbox."""
    try:
        result = run_transform(source, to_grid(arr))
    except SandboxError:
        return None
    if result is None:
        return None
    return to_np(result)
