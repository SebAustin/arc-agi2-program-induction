#!/usr/bin/env python3
"""Fast, faithful ablation of the analytic fitter families over an ARC dataset.

The solve engine runs with ``early_stop_admissible=1``: if the fitters yield any
admissible program, the (expensive) enumerative proposer is skipped. So the final
engine's per-task result is exactly:

  * the pass@2 of the fitter's admissible programs (posterior mode + pass@2
    selector) when the enabled fitters produce at least one admissible program;
  * otherwise the committed *core* report's result (the enumerative outcome),
    which this tool reads from ``--core-report`` instead of re-running enumeration.

This reproduces the real engine's pass@2 in seconds and attributes each
newly-solved task to the family whose program won. Validate it against a known
full run (e.g. ARC-1 eval with F2,F1 enabled should reproduce 22/400) before
trusting a derived number.

Usage:
    PYTHONPATH=src python3 scripts/eval_ablation.py \\
        --data data/arc_agi_1/evaluation \\
        --core-report reports/arc1_eval.json \\
        --families F2,F1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

from arc_solver.config import Budget  # noqa: E402
from arc_solver.eval.harness import pass_at_2  # noqa: E402
from arc_solver.io.loader import load_dataset  # noqa: E402
from arc_solver.search.fitted import FittedProposer  # noqa: E402
from arc_solver.search.verifier import consistent  # noqa: E402
from arc_solver.select.posterior import posterior_weights  # noqa: E402
from arc_solver.select.selector import pick_top2  # noqa: E402

FAMILY_OF = {
    "panel_select": "F2", "panel_combine": "F2",
    "select_object": "F4", "count_grid": "F4",
    "per_object": "F1", "map_object_color": "F1",
}


def _program_families(prog):
    return {FAMILY_OF[op.name] for op in prog if op.name in FAMILY_OF}


def _fitter_result(task, proposer, budget, enabled):
    """(solved, winning_family) from the fitters alone, or (None, None) if the
    fitters produce no admissible program (caller falls back to the core report)."""
    admissible = []
    for prog in proposer.propose(task, budget=budget):
        if not _program_families(prog) <= enabled:
            continue
        if consistent(prog, task):
            admissible.append(prog)
    if not admissible:
        return None, None
    scored = posterior_weights(admissible)
    scored.sort(key=lambda sp: sp.weight, reverse=True)
    attempts = pick_top2(scored, [p.input for p in task.test])
    solved = all(pass_at_2(a, p.output) for a, p in zip(attempts, task.test))
    fam = next((f for o in scored[0].program if (f := FAMILY_OF.get(o.name))), "core")
    return solved, (fam if solved else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--core-report", required=True)
    ap.add_argument("--families", default="F2,F1,F4", help="comma-sep: F2,F1,F4")
    ap.add_argument("--max-seconds", type=float, default=10.0)
    args = ap.parse_args(argv)

    enabled = {f.strip() for f in args.families.split(",") if f.strip()}
    budget = Budget(max_seconds=args.max_seconds)
    rep = json.load(open(args.core_report))
    recs = rep.get("records", rep) if isinstance(rep, dict) else rep
    core = {r["task_id"]: float(r.get("score", 0)) >= 1.0 for r in recs}

    proposer = FittedProposer()
    tasks = load_dataset(args.data)
    solved, gains, regressions = [], {}, []
    for task in tasks:
        fit_solved, fam = _fitter_result(task, proposer, budget, enabled)
        core_solved = core.get(task.task_id, False)
        final = core_solved if fit_solved is None else fit_solved
        if final:
            solved.append(task.task_id)
        if final and not core_solved:
            gains[fam] = gains.get(fam, 0) + 1
        if core_solved and not final:
            regressions.append(task.task_id)

    n = len(tasks)
    out = {
        "data": os.path.basename(os.path.normpath(args.data)),
        "families": sorted(enabled),
        "n": n,
        "core_pass_at_2": round(sum(core.get(t.task_id, False) for t in tasks) / n, 4),
        "pass_at_2": round(len(solved) / n, 4),
        "n_solved": len(solved),
        "gains_by_family": gains,
        "regressions": regressions,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
