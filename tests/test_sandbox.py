"""Tests for the restricted Python sandbox."""

from __future__ import annotations

import signal
import time

import pytest

from arc_solver.dsl.sandbox import SandboxError, run_transform


def test_valid_transform_runs():
    src = "def transform(grid):\n    return [list(reversed(r)) for r in grid]"
    assert run_transform(src, ((1, 2), (3, 4))) == ((2, 1), (4, 3))


@pytest.mark.parametrize(
    "src",
    [
        "import os\ndef transform(grid):\n    return grid",
        "def transform(grid):\n    return open('x').read()",
        "def transform(grid):\n    return __import__('os')",
        "def transform(grid):\n    return eval('1')",
    ],
)
def test_screening_rejects_dangerous_source(src):
    with pytest.raises(SandboxError):
        run_transform(src, ((1,),))


def test_missing_transform_raises():
    with pytest.raises(SandboxError):
        run_transform("x = 1", ((1,),))


def test_invalid_output_returns_none():
    # Returns a non-grid -> None
    assert run_transform("def transform(grid):\n    return 5", ((1,),)) is None
    # Out-of-range colour -> None
    assert run_transform("def transform(grid):\n    return [[42]]", ((1,),)) is None


def test_runtime_error_returns_none():
    assert run_transform("def transform(grid):\n    return grid[99]", ((1,),)) is None


@pytest.mark.parametrize(
    "src",
    [
        # dunder / introspection pivots
        "def transform(grid):\n    return (1).__class__",
        "def transform(grid):\n    return [].__class__.__base__",
        # frame-introspection attrs (do not start with underscore)
        "def transform(grid):\n    g = (x for x in grid)\n    return g.gi_frame.f_builtins",
        # dangerous identifiers reached without a call
        "def transform(grid):\n    f = getattr\n    return grid",
        "def transform(grid):\n    t = type\n    return grid",
    ],
)
def test_ast_screen_blocks_escapes(src):
    with pytest.raises(SandboxError):
        run_transform(src, ((1,),))


def test_frozen_safe_builtins():
    """Regression guard: the allow-list must not silently gain risky builtins."""
    from arc_solver.dsl.sandbox import _SAFE_BUILTINS

    risky = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "vars",
        "type",
        "dir",
        "input",
        "object",
    }
    assert risky.isdisjoint(_SAFE_BUILTINS.keys())


@pytest.mark.parametrize(
    "src",
    [
        # str.format with a runtime-built dunder ("_"+"_"+...) traverses attributes
        # while hiding "__" from the substring screen and the dunder from the AST
        # attribute rule (it lives in a string constant). The format *vehicle* is
        # denied outright, so all of these must be rejected.
        (
            "def transform(grid):\n"
            " u='_'\n"
            " name=u+u+'class'+u+u\n"
            " spec='{0.'+name+'}'\n"
            " s=spec.format(grid)\n"
            " return grid\n"
        ),
        (
            "def transform(grid):\n"
            " u='_'\n"
            " c=u+u+'class'+u+u\n"
            " spec='{grid.'+c+'}'\n"
            " s=spec.format_map({'grid':grid})\n"
            " return grid\n"
        ),
        # Literal-template variant: dunder marker inside a string constant.
        "def transform(grid):\n s='{0.__class__}'.format(grid)\n return grid\n",
    ],
)
def test_ast_screen_blocks_format_string_smuggling(src):
    with pytest.raises(SandboxError):
        run_transform(src, ((1,),))


@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="wall-clock timeout needs POSIX SIGALRM"
)
def test_infinite_loop_is_killed_by_timeout():
    """The safety-critical claim: a runaway transform is aborted, not hung."""
    src = "def transform(grid):\n    while True:\n        pass\n    return grid"
    start = time.monotonic()
    result = run_transform(src, ((1,),), timeout_s=0.2)
    elapsed = time.monotonic() - start
    assert result is None  # killed and swallowed to a non-result
    assert elapsed < 2.0  # returned promptly rather than hanging
