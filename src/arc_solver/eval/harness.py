"""Evaluation harness.

Scores a solver on a dataset using the competition's normalised pass@2: for each
task, the score is the mean over its test inputs of whether *either* of the two
attempts exactly matches the truth; the dataset score is the mean over tasks.
"""

from __future__ import annotations

import time
from typing import Sequence

from arc_solver.dsl.program import pretty
from arc_solver.eval.report import EvalReport, TaskRecord
from arc_solver.grid.metrics import grid_equal
from arc_solver.io.task import Grid, Task
from arc_solver.search.engine import SolveEngine


def pass_at_2(attempts: tuple[Grid, Grid], truth: Grid | None) -> bool:
    """True iff either attempt exactly matches the truth grid."""
    if truth is None:
        return False
    return grid_equal(attempts[0], truth) or grid_equal(attempts[1], truth)


def _task_score(attempts: Sequence[tuple[Grid, Grid]], task: Task) -> float:
    """Normalised pass@2 for one task (mean over its test inputs)."""
    if not task.test:
        return 0.0
    hits = sum(
        1.0 for att, pair in zip(attempts, task.test) if pass_at_2(att, pair.output)
    )
    return hits / len(task.test)


def evaluate(
    engine: SolveEngine,
    dataset: Sequence[Task],
    *,
    config: str = "offline-enumerative",
    dataset_name: str = "dataset",
    progress: bool = False,
) -> EvalReport:
    """Run ``engine`` over every task and aggregate normalised pass@2."""
    start = time.monotonic()
    records: list[TaskRecord] = []
    total_score = 0.0
    solved_train = 0

    for idx, task in enumerate(dataset):
        result = engine.solve(task)
        score = _task_score(result.attempts, task)
        total_score += score
        solved_train += int(result.solved_train)
        records.append(
            TaskRecord(
                task_id=task.task_id,
                score=score,
                solved_train=result.solved_train,
                n_admissible=result.meta.get("n_admissible", 0),
                seconds=result.meta.get("seconds", 0.0),
                best_program=(
                    pretty(result.best_program) if result.best_program else None
                ),
            )
        )
        if progress and (idx + 1) % 25 == 0:
            running = total_score / (idx + 1)
            print(f"  [{idx + 1}/{len(dataset)}] running pass@2 = {running:.3f}")

    n = len(dataset)
    return EvalReport(
        dataset=dataset_name,
        config=config,
        n_tasks=n,
        pass_at_2=total_score / n if n else 0.0,
        train_solve_rate=solved_train / n if n else 0.0,
        total_seconds=round(time.monotonic() - start, 1),
        records=records,
    )
