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
