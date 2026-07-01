# Assumptions & Decisions Log

Documented assumptions made during the autonomous build (per the agency
workflow). Each is a deliberate, reversible choice, not an oversight.

## Scope & strategy
- **Track:** ARC-AGI-2 (confirmed with user). ARC-AGI-1 is used as a larger,
  easier development set (400 eval tasks vs ARC-AGI-2's 120) for tuning and for a
  second reported number.
- **Strategy:** balanced — a novel, theoretically-grounded method that is fully
  reproducible offline on CPU, with optional higher-accuracy LLM/GPU variants.
- **This session's deliverable:** a complete, runnable v1 (solver + paper +
  notebook + repo) producing an honest offline number. Scaling up (LLM/TTT runs)
  and the actual Kaggle leaderboard submission are the user's last mile.

## Technical assumptions
- **Grid type** is an immutable `tuple[tuple[int,...],...]` — hashable, so
  predicted outputs are usable as dict keys in the selector, and never mutated.
- **DSL is a linear pipeline** of `Grid -> Grid` primitives (objectness handled
  inside primitives), rather than a full lambda calculus — keeps enumeration and
  MDL scoring tractable while still composing.
- **Background colour** is heuristically the most frequent colour (ties toward
  black/0). A handful of ARC tasks violate this; affected primitives simply do
  not fire and the program is judged inadmissible.
- **Admissibility = exact match on ALL train pairs.** No partial credit is used
  to admit a program; `partial_score` only steers refinement.
- **pass@2 scoring** follows the competition's normalised form: per task, the
  mean over test inputs of whether *either* attempt matches; dataset score is the
  mean over tasks.
- **Datasets** are the public ARC-AGI-1 (`fchollet/ARC-AGI`) and ARC-AGI-2
  (`arcprize/ARC-AGI-2`) GitHub repos, downloaded into git-ignored `data/`.

## Reproducibility boundary
- The **offline** path (fitted + enumerative proposers) is the reproducible core
  and the headline number — it runs on CPU with only numpy and no network.
- The **LLM-amortised** path requires an API key (read from env only) and is
  validated with a `MockClient` in tests (no network). Its accuracy figures are
  to be reproduced by the user with their own key, and are reported separately —
  never conflated with the offline number.
- The optional **TTT** (GPU) variant is described as future work; it is not built
  in this session.

## Open items for the user
- Run the offline notebook on Kaggle to obtain the official ARC-AGI-2 leaderboard
  score and record the **submission ID** for the paper's Accuracy criterion.
- Optionally run the LLM-amortised path with an Anthropic/OpenAI key to report
  the higher-accuracy variant.

## Deployment assumptions (Kaggle rollout, `deploy/kaggle` branch)

- **Combined-challenges file paths are ground truth from the user, not
  independently re-derived by re-running Kaggle.** The exact mount path
  `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json` (and
  sibling `arc-agi_training-challenges.json`,
  `arc-agi_evaluation-challenges.json`, `sample_submission.json`) is taken as
  given; nothing in this session executed against Kaggle's actual
  infrastructure. All validation is local, against a same-shaped
  combined-challenges file built from public `data/arc_agi_1/training/*.json`.
- **`load_challenges` requires every train pair to carry an output and every
  test pair to omit one** (mirroring `load_task`'s existing contract) rather
  than trying to infer intent from a present-but-`null` `"output"` key. A
  `null` `"output"` on a test pair is treated the same as an absent key (both
  parse to `Pair(output=None)`); only a *present, non-null* `"output"` counts
  as revealed.
- **`load_challenges` sorts tasks by `task_id` for determinism**, matching
  `load_dataset`'s existing sort-by-filename behaviour. The competition does
  not document an implied ordering in the combined JSON's key order, so
  sorting is the safer, reproducible choice — it also means task processing
  order is stable across identical-content files with different key order
  (e.g. re-serialised by different JSON libraries).
- **`make_submission --data` auto-detects file vs. directory** via
  `os.path.isfile` / `os.path.isdir` rather than a new explicit flag, to keep
  the existing CLI surface and `docs/RUNBOOK.md` examples working unchanged
  for local per-task-file directories while adding the combined-file path the
  competition actually needs. A nonexistent path raises `FileNotFoundError`
  with an explicit message rather than silently doing nothing.
- **The notebook's data-locating cell tries, in order:** the real Kaggle test
  mount, then the Kaggle evaluation-challenges mount (a same-shaped dry run
  using data the notebook author can actually see the ground truth for), then
  a locally-built `data/kaggle_dry_run_challenges.json` for pure local
  iteration. This mirrors the existing pattern in the pre-existing notebook
  (which searched several `arc_solver` import locations the same way) rather
  than introducing a new convention.
- **The Kaggle Dataset name `arc-solver-src` is a recommendation, not an
  enforced constraint** — the notebook's import cell already searched this
  slug (and `arc-solver`) before this session's changes; that convention is
  kept as-is in `DEPLOYMENT.md` rather than invented fresh.
- **No production deploy step was executed.** No notebook was pushed to
  Kaggle, no dataset was uploaded, no competition submission was made, and the
  full ARC-AGI-2 evaluation sweep was not run (only a 3-task local dry run and
  the existing fast test suite). `DEPLOYMENT.md` documents the exact gated
  steps for the user to run themselves.
