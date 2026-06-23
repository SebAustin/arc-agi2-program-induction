# Runbook & Deployment Guide

Operational guide for reproducing results, building a submission, and the gated
rollout of the optional LLM-amortised path.

## 1. Environment

- Python ≥ 3.10. Core path needs only **numpy**; figures need **matplotlib**.
- No GPU and no network are required for the offline solver.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
bash scripts/get_data.sh          # downloads public ARC-AGI-1 & -2 into data/
```

## 2. Reproduce the offline numbers

```bash
# Per-dataset evaluation (writes JSON + markdown into reports/)
python -m arc_solver.cli.run_eval --data data/arc_agi_1/training   --max-seconds 2 --out reports/arc1_train.json --md reports/arc1_train.md
python -m arc_solver.cli.run_eval --data data/arc_agi_1/evaluation --max-seconds 2 --out reports/arc1_eval.json  --md reports/arc1_eval.md
python -m arc_solver.cli.run_eval --data data/arc_agi_2/evaluation --max-seconds 2 --out reports/arc2_eval.json  --md reports/arc2_eval.md
```

Runs are deterministic (no randomness in the offline path), so numbers reproduce
exactly given the same budget. Increasing `--max-seconds` / `--max-programs`
expands search depth and can only raise coverage.

## 3. Build a Kaggle submission

```bash
python -m arc_solver.cli.make_submission \
  --data data/arc_agi_2/evaluation --out submission.json \
  --task-seconds 10 --total-seconds 39600
```

- `--total-seconds` is a wall-clock guard; on expiry every remaining task gets a
  safe identity/mirror fallback so **no task is missing** (a hard requirement).
- The notebook `notebooks/kaggle_submission.ipynb` does the same thing inside the
  Kaggle sandbox. Attach this repo's `src/` as a Kaggle Dataset (e.g.
  `arc-solver-src`); the setup cell finds it automatically.

### Recording the leaderboard submission (for the paper's Accuracy criterion)
1. Push the notebook + `src/` dataset to Kaggle; run to produce `submission.json`.
2. Submit; note the **submission ID** shown on the competition page.
3. Put that ID in the write-up so judges can match notebook → leaderboard score.

## 4. Optional: LLM-amortised path (gated rollout)

This path sends task grids to a third-party LLM API and **executes
model-authored Python in a sandbox**. Treat enabling it as a deliberate step.

**Pre-flight checklist**
- [ ] Read `SECURITY.md` — understand the sandbox threat model and residual risk.
- [ ] Set credentials in the environment only (never in code/notebooks):
      `export ARC_LLM_PROVIDER=anthropic` (or `openai`); `export ARC_LLM_KEY=...`
- [ ] Confirm a spend cap on the API account (the proposer samples N programs and
      refines, so cost scales with task count × samples × rounds).
- [ ] Run on a small slice first (`--limit 10` via a custom script) to estimate
      cost/accuracy before a full sweep.

**Note:** this path cannot run inside the Kaggle no-internet sandbox; it is for
the *verified / external* evaluation and for research, not the constrained
leaderboard.

## 5. Regenerate figures

```bash
PYTHONPATH=src python3 scripts/make_figures.py   # assets/cover.png + assets/gallery/*
```

## 6. Health checks

```bash
PYTHONPATH=src pytest --cov=arc_solver            # 88 tests, ~91% coverage
PYTHONPATH=src python -m arc_solver.cli.solve --task data/arc_agi_1/training/0d3d703e.json
```

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `arc_solver not found` in notebook | `src/` not attached on Kaggle | add it as a Dataset; check the path list in the setup cell |
| `no task JSON found` | data not downloaded | run `scripts/get_data.sh` |
| LLM path does nothing | no key in env | set `ARC_LLM_PROVIDER` + `ARC_LLM_KEY` |
| Eval slower than expected | large `--max-seconds` on hard sets | lower the per-task budget; fitted solves are unaffected |
