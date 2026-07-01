"""Build a Kaggle ``submission.json`` from the offline solver.

Every task is solved with the offline engine (no network). A wall-clock guard
ensures the run finishes within a time budget, emitting an identity-grid
fallback for any task not solved in time so that *every* task is present in
the submission (a hard competition requirement).

``--data`` accepts EITHER form the solver may be asked to run against:

- A single combined-challenges JSON file (the real Kaggle rerun format:
  ``{"<task_id>": {"train": [...], "test": [...]}}``), parsed with
  ``load_challenges``.
- A directory of one-task-per-file JSON (the public ARC-AGI-1/-2 layout used
  for local development), parsed with ``load_dataset``.

The two are auto-detected from whether ``--data`` points at a file or a
directory, so the same command works locally and on Kaggle.

Usage:
    python -m arc_solver.cli.make_submission \\
        --data data/arc_agi_2/evaluation --out submission.json \\
        --task-seconds 10 --total-seconds 39600

    python -m arc_solver.cli.make_submission \\
        --data /kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json \\
        --out /kaggle/working/submission.json \\
        --task-seconds 10 --total-seconds 39600
"""

from __future__ import annotations

import argparse
import os
import time

from arc_solver.cli.run_eval import build_engine
from arc_solver.io.loader import load_challenges, load_dataset
from arc_solver.io.submission import SubmissionWriter
from arc_solver.io.task import Grid, Task


def load_tasks(data_path: str) -> list[Task]:
    """Load tasks from ``data_path``, auto-detecting the on-disk format.

    A file is treated as a combined-challenges JSON (``load_challenges``); a
    directory is treated as one-task-per-file (``load_dataset``).
    """
    if os.path.isfile(data_path):
        return load_challenges(data_path)
    if os.path.isdir(data_path):
        return load_dataset(data_path)
    raise FileNotFoundError(f"--data path does not exist: {data_path}")


def _fallback(task: Task) -> list[tuple[Grid, Grid]]:
    """Identity attempt_1 plus a mirror attempt_2 for every test input."""
    out: list[tuple[Grid, Grid]] = []
    for pair in task.test:
        g = pair.input
        mirror = tuple(tuple(reversed(row)) for row in g)
        out.append((g, mirror if mirror != g else g))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate an ARC submission.json.")
    ap.add_argument(
        "--data",
        required=True,
        help=(
            "a combined-challenges JSON file (Kaggle rerun format) OR a "
            "directory of one-task-per-file JSON (local ARC-AGI-1/-2 layout)"
        ),
    )
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--task-seconds", type=float, default=10.0)
    ap.add_argument("--max-programs", type=int, default=20_000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument(
        "--total-seconds",
        type=float,
        default=0.0,
        help="overall wall-clock guard; 0 disables (fallback fills the rest)",
    )
    args = ap.parse_args(argv)

    dataset = load_tasks(args.data)
    engine = build_engine(args.task_seconds, args.max_programs, args.max_depth)
    writer = SubmissionWriter()
    start = time.monotonic()
    solved = 0

    for task in dataset:
        out_of_time = (
            args.total_seconds and (time.monotonic() - start) > args.total_seconds
        )
        if out_of_time:
            writer.add(task.task_id, _fallback(task))
            continue
        result = engine.solve(task)
        if result.solved_train:
            writer.add(task.task_id, list(result.attempts))
            solved += 1
        else:
            # Use the engine's attempts (already includes identity fallback).
            writer.add(task.task_id, list(result.attempts))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer.write(args.out)
    print(
        f"wrote {args.out}: {len(writer)} tasks "
        f"({solved} with an admissible program), "
        f"{time.monotonic() - start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
