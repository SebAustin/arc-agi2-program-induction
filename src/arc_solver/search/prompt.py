"""Prompt construction and code extraction for the LLM proposer.

Kept separate from the proposer so the prompt format is easy to test and tune.
Grids are rendered as compact digit rows; the model is asked for a single
``transform(grid)`` Python function.
"""

from __future__ import annotations

import re

from arc_solver.io.task import Grid, Task

SYSTEM = (
    "You are an expert at ARC (Abstraction and Reasoning Corpus) puzzles. "
    "Each task shows input/output grid pairs that share one transformation rule. "
    "Infer the rule and express it as a single Python function "
    "`def transform(grid):` that maps a list-of-lists of ints (0-9) to the "
    "output list-of-lists. Use only plain Python and pure list operations — no "
    "imports, no file or network access. Return ONLY one fenced ```python code "
    "block."
)

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _render_grid(grid: Grid) -> str:
    return "\n".join("".join(str(c) for c in row) for row in grid)


def render_task(task: Task) -> str:
    """Render train pairs (and the test input) as a compact text prompt."""
    parts: list[str] = []
    for i, pair in enumerate(task.train):
        parts.append(f"# Example {i + 1} input:\n{_render_grid(pair.input)}")
        if pair.output is not None:
            parts.append(f"# Example {i + 1} output:\n{_render_grid(pair.output)}")
    parts.append(
        "Write `def transform(grid):` implementing the rule shared by all "
        "examples. Return only the code block."
    )
    return "\n\n".join(parts)


def render_feedback(source: str, failures: list[str]) -> str:
    """Build a refinement prompt citing the previous attempt's failures."""
    detail = "\n".join(failures)
    return (
        "Your previous transform was incorrect on some examples.\n\n"
        f"```python\n{source}\n```\n\n"
        f"Failures:\n{detail}\n\n"
        "Return a corrected `def transform(grid):` as a single code block."
    )


def extract_code(completion: str) -> str | None:
    """Pull the first fenced code block (or treat the whole text as code)."""
    m = _CODE_RE.search(completion)
    if m:
        return m.group(1).strip()
    if "def transform" in completion:
        return completion.strip()
    return None
