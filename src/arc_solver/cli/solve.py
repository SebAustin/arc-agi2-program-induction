"""Solve a single task and print the induced program and prediction.

Usage:
    python -m arc_solver.cli.solve --task data/arc_agi_1/training/007bbfb7.json
"""

from __future__ import annotations

import argparse

from arc_solver.cli.run_eval import build_engine
from arc_solver.dsl.program import pretty
from arc_solver.eval.harness import pass_at_2
from arc_solver.io.loader import load_task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve one ARC task.")
    ap.add_argument("--task", required=True, help="path to a task JSON file")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--max-programs", type=int, default=30_000)
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args(argv)

    task = load_task(args.task)
    engine = build_engine(args.max_seconds, args.max_programs, args.max_depth)
    result = engine.solve(task)

    print(f"task: {task.task_id}")
    print(f"admissible programs: {result.meta['n_admissible']}")
    if result.best_program is not None:
        print(f"best program: {pretty(result.best_program)}")
    for k, (att, pair) in enumerate(zip(result.attempts, task.test)):
        verdict = ""
        if pair.output is not None:
            verdict = "  CORRECT" if pass_at_2(att, pair.output) else "  wrong"
        print(f"test {k + 1} attempt_1: {att[0]}{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
