"""Run the offline solver over an ARC dataset and report normalised pass@2.

Usage:
    python -m arc_solver.cli.run_eval --data data/arc_agi_1/evaluation \\
        --max-seconds 15 --limit 100 --out reports/arc1_eval.json
"""

from __future__ import annotations

import argparse
import os

from arc_solver.config import Budget
from arc_solver.eval.harness import evaluate
from arc_solver.io.loader import load_dataset
from arc_solver.search.engine import SolveEngine
from arc_solver.search.enumerative import EnumerativeProposer
from arc_solver.search.fitted import FittedProposer


def build_engine(max_seconds: float, max_programs: int, max_depth: int) -> SolveEngine:
    """The offline engine: fitted analytic proposals first, then enumeration.

    Both proposers are offline and need no network.
    """
    budget = Budget(
        max_programs=max_programs, max_seconds=max_seconds, max_depth=max_depth
    )
    return SolveEngine([FittedProposer(), EnumerativeProposer()], budget)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the offline ARC solver.")
    ap.add_argument("--data", required=True, help="directory of task JSON files")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--max-programs", type=int, default=30_000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N tasks")
    ap.add_argument("--out", default="", help="optional JSON report path")
    ap.add_argument("--md", default="", help="optional markdown report path")
    args = ap.parse_args(argv)

    dataset = load_dataset(args.data)
    if args.limit:
        dataset = dataset[: args.limit]
    name = os.path.basename(os.path.normpath(args.data))

    engine = build_engine(args.max_seconds, args.max_programs, args.max_depth)
    report = evaluate(engine, dataset, dataset_name=name, progress=True)

    print(
        f"\n{name}: pass@2 = {report.pass_at_2:.4f} "
        f"({report.pass_at_2 * 100:.2f}%)  |  "
        f"train-solve = {report.train_solve_rate * 100:.2f}%  |  "
        f"{report.total_seconds:.1f}s total"
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        report.to_json(args.out)
        print(f"wrote {args.out}")
    if args.md:
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown())
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
