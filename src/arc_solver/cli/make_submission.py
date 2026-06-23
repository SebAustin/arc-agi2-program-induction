"""Build a Kaggle ``submission.json`` from the offline solver.

Every task in the evaluation directory is solved with the offline engine (no
network). A wall-clock guard ensures the run finishes within a time budget,
emitting an identity-grid fallback for any task not solved in time so that
*every* task is present in the submission (a hard competition requirement).

Usage:
    python -m arc_solver.cli.make_submission --data data/arc_agi_2/evaluation \\
        --out submission.json --task-seconds 10 --total-seconds 39600
"""

from __future__ import annotations

import argparse
import time

from arc_solver.cli.run_eval import build_engine
from arc_solver.io.loader import load_dataset
from arc_solver.io.submission import SubmissionWriter
from arc_solver.io.task import Grid, Task


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
    ap.add_argument("--data", required=True)
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

    dataset = load_dataset(args.data)
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

    writer.write(args.out)
    print(
        f"wrote {args.out}: {len(writer)} tasks "
        f"({solved} with an admissible program), "
        f"{time.monotonic() - start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
