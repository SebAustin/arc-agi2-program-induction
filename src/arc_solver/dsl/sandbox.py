"""Restricted execution of LLM-emitted ``transform`` functions.

The LLM proposer asks a model to emit a Python ``def transform(grid)`` that maps
a list-of-lists grid to a list-of-lists grid. Executing model-authored code is
inherently risky, so this sandbox applies defence in depth:

1. **Static screening** — reject source containing imports, dunder access, or
   known-dangerous names *before* compiling.
2. **Restricted builtins** — execute with a small allow-list of builtins; no
   ``open``, ``eval``, ``exec``, ``__import__``, file or network access.
3. **Wall-clock timeout** — abort runaway code via ``SIGALRM`` (POSIX).

This is a pragmatic guard for a developer running their own key locally, not a
hard security boundary. The offline solver never imports this module.
"""

from __future__ import annotations

import ast
import signal
from typing import Callable

from arc_solver.config import MAX_GRID_CELLS
from arc_solver.io.task import Grid

# Tokens that must never appear in model-emitted transform source.
_FORBIDDEN = (
    "import",
    "__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "getattr",
    "setattr",
    "delattr",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "input(",
    "breakpoint",
)

_SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "int": int,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}


class SandboxError(Exception):
    """Raised when source fails static screening or compilation."""


# Names that must never be referenced, even though they are absent from the
# restricted builtins (defence in depth — an AST reference is rejected outright).
_FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "memoryview",
        "bytearray",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "help",
        "exit",
        "quit",
    }
)

# Attribute names that expose frames / introspection internals. Note these do not
# start with an underscore, so the dunder rule below would miss them.
_FORBIDDEN_ATTRS = frozenset(
    {
        "gi_frame",
        "gi_code",
        "f_back",
        "f_globals",
        "f_builtins",
        "f_locals",
        "f_code",
        "tb_frame",
        "tb_next",
        "cr_frame",
        "ag_frame",
        "func_globals",
    }
)


def _screen(source: str) -> None:
    """Fast substring pre-filter (cheap rejection of obvious bad source)."""
    for token in _FORBIDDEN:
        if token in source:
            raise SandboxError(f"forbidden token in transform source: {token!r}")


def _ast_screen(source: str) -> None:
    """Robust AST screen: the real boundary alongside the restricted builtins.

    Rejects imports, every dunder/private or frame-introspection attribute
    access, and references to dangerous identifiers — catching escapes that a
    substring filter cannot (e.g. ``getattr`` reached via aliasing).
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"could not parse transform: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("imports are not permitted in transform source")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") or attr in _FORBIDDEN_ATTRS:
                raise SandboxError(f"forbidden attribute access: {attr!r}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise SandboxError(f"forbidden name reference: {node.id!r}")


class _Timeout(Exception):
    pass


def _compile_transform(source: str) -> Callable:
    _screen(source)
    _ast_screen(source)
    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    try:
        code = compile(source, "<transform>", "exec")
        exec(code, namespace)  # noqa: S102 — guarded by screening + builtins
    except Exception as exc:  # pragma: no cover - exercised via run_transform
        raise SandboxError(f"failed to compile transform: {exc}") from exc
    fn = namespace.get("transform")
    if not callable(fn):
        raise SandboxError("source did not define a callable 'transform'")
    return fn


def _to_grid(obj) -> Grid | None:
    """Validate a transform's return value as a small int grid."""
    if not isinstance(obj, (list, tuple)) or not obj:
        return None
    rows: list[tuple[int, ...]] = []
    width = None
    cells = 0
    for row in obj:
        if not isinstance(row, (list, tuple)) or not row:
            return None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None
        out_row = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= 9):
                return None
            out_row.append(v)
        cells += len(out_row)
        if cells > MAX_GRID_CELLS:
            return None
        rows.append(tuple(out_row))
    return tuple(rows)


def run_transform(source: str, grid: Grid, *, timeout_s: float = 2.0) -> Grid | None:
    """Compile and run ``transform`` on ``grid`` under the sandbox.

    Returns the validated output grid, or ``None`` if the code errors, times out,
    or returns an invalid grid. Raises :class:`SandboxError` only for source that
    fails static screening / compilation (a programming error, not a bad guess).
    """
    fn = _compile_transform(source)
    grid_arg = [list(row) for row in grid]

    def _handler(signum, frame):  # pragma: no cover - timing dependent
        raise _Timeout()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        result = fn(grid_arg)
    except Exception:
        return None
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    return _to_grid(result)
