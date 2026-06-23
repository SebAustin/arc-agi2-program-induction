"""Generate the cover image and media-gallery figures for the write-up.

* Cover: a labelled diagram of the test-time Bayesian program-induction pipeline.
* Gallery: rendered solved tasks (train pairs + test input + the induced
  prediction and program), taken from the offline evaluation report.

Run:  PYTHONPATH=src python3 scripts/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from arc_solver.cli.run_eval import build_engine
from arc_solver.dsl.program import pretty
from arc_solver.io.loader import load_task
from arc_solver.viz.render import ARC_PALETTE, render_task_figure

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
GALLERY = os.path.join(ASSETS, "gallery")


def make_cover(path: str) -> None:
    """Draw the method pipeline as the cover image."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                linewidth=1.5,
                edgecolor=color,
                facecolor="#161b22",
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.32,
            title,
            ha="center",
            va="top",
            color="#e6edf3",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + h - 0.78,
            sub,
            ha="center",
            va="top",
            color="#9da7b3",
            fontsize=8.2,
            wrap=True,
        )

    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#58a6ff",
            )
        )

    ax.text(
        5.5,
        4.95,
        "Test-Time Bayesian Program Induction for ARC-AGI-2",
        ha="center",
        color="#e6edf3",
        fontsize=14,
        fontweight="bold",
    )

    box(0.2, 2.0, 2.0, 1.7, "Demonstrations", "few input/output\ngrid pairs", "#3fb950")
    box(2.6, 3.0, 2.3, 1.3, "Fitted proposals", "analytic MAP\nhypotheses", "#d29922")
    box(
        2.6, 1.1, 2.3, 1.3, "Enumerative search", "MDL-ordered\nDSL programs", "#d29922"
    )
    box(5.3, 2.0, 2.0, 1.7, "Verify", "consistent on\nALL train pairs", "#f85149")
    box(7.7, 2.0, 1.5, 1.7, "Posterior", "MDL prior\nweights", "#a371f7")
    box(9.4, 2.0, 1.4, 1.7, "Top-2", "mode +\ndisagree", "#58a6ff")

    arrow(2.2, 3.1, 2.6, 3.5)
    arrow(2.2, 2.6, 2.6, 1.9)
    arrow(4.9, 3.5, 5.3, 3.0)
    arrow(4.9, 1.9, 5.3, 2.6)
    arrow(7.3, 2.85, 7.7, 2.85)
    arrow(9.2, 2.85, 9.4, 2.85)

    ax.text(
        10.1,
        1.7,
        "attempt_1\nattempt_2",
        ha="center",
        va="top",
        color="#58a6ff",
        fontsize=8,
    )
    ax.text(
        5.5,
        0.35,
        "skill acquired per task from the demonstrations alone  •  "
        "minimal core-knowledge priors  •  Occam via description length",
        ha="center",
        color="#7d8590",
        fontsize=9,
        style="italic",
    )

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_gallery(report_path: str, data_dir: str, n: int = 6) -> list[str]:
    """Render up to ``n`` solved tasks from a report into the gallery."""
    os.makedirs(GALLERY, exist_ok=True)
    report = json.load(open(report_path))
    solved = [r for r in report["records"] if r["score"] > 0][:n]
    engine = build_engine(max_seconds=8.0, max_programs=20000, max_depth=4)
    written = []
    for rec in solved:
        task = load_task(os.path.join(data_dir, rec["task_id"] + ".json"))
        result = engine.solve(task)
        preds = [a[0] for a in result.attempts]
        out = os.path.join(GALLERY, f"{rec['task_id']}.png")
        render_task_figure(task, out, predictions=preds)
        written.append(out)
        print(f"  {rec['task_id']}: {pretty(result.best_program)}")
    return written


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    cover = os.path.join(ASSETS, "cover.png")
    make_cover(cover)
    print("wrote", cover)
    report = os.path.join(os.path.dirname(__file__), "..", "reports", "arc1_train.json")
    data = os.path.join(
        os.path.dirname(__file__), "..", "data", "arc_agi_1", "training"
    )
    if os.path.exists(report):
        files = make_gallery(report, data)
        print(f"wrote {len(files)} gallery figures to {GALLERY}")


if __name__ == "__main__":
    main()
