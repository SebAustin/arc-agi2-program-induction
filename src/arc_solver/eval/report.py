"""Evaluation report structures and serialisation.

The report distinguishes the *offline* configuration from any LLM-assisted run
(recorded in ``config``) so numbers are never conflated in the paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TaskRecord:
    """Per-task evaluation outcome."""

    task_id: str
    score: float  # normalised pass@2 in [0, 1] for this task
    solved_train: bool
    n_admissible: int
    seconds: float
    best_program: str | None


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    config: str  # e.g. "offline-enumerative" or "llm-assisted(anthropic)"
    n_tasks: int
    pass_at_2: float  # mean normalised pass@2 across tasks
    train_solve_rate: float  # fraction with >=1 admissible program
    total_seconds: float
    records: list[TaskRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation: {self.dataset}",
            "",
            f"- **Configuration:** {self.config}",
            f"- **Tasks:** {self.n_tasks}",
            f"- **pass@2:** {self.pass_at_2:.4f} ({self.pass_at_2 * 100:.2f}%)",
            f"- **Train-solve rate:** {self.train_solve_rate:.4f} "
            f"({self.train_solve_rate * 100:.2f}%)",
            f"- **Total time:** {self.total_seconds:.1f}s "
            f"({self.total_seconds / max(self.n_tasks, 1):.2f}s/task)",
            "",
            "## Solved tasks",
            "",
            "| task_id | score | program |",
            "| --- | --- | --- |",
        ]
        for rec in self.records:
            if rec.score > 0:
                lines.append(
                    f"| {rec.task_id} | {rec.score:.2f} | `{rec.best_program}` |"
                )
        return "\n".join(lines) + "\n"
