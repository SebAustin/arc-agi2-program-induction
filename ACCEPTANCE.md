# Acceptance

Mapping each committed requirement and success criterion to its evidence.

## Deliverables (paper-track package)

| Deliverable | Status | Evidence |
| --- | --- | --- |
| Runnable offline solver with a real pass@2 number | ✅ | `src/arc_solver/`, `reports/arc1_train.json` (9.25%), `reports/arc1_eval.json` (3.00%) |
| ≤1500-word write-up | ✅ | `WRITEUP.md` (~1.36k words) |
| Extended theory | ✅ | `docs/THEORY.md` |
| Kaggle notebook (offline, no network) | ✅ | `notebooks/kaggle_submission.ipynb` (validated; offline `EnumerativeProposer` only) |
| Cover image + media gallery | ✅ | `assets/cover.png`, `assets/gallery/*.png` (6 solved tasks) |
| README + runbook + ADRs | ✅ | `README.md`, `docs/RUNBOOK.md`, `docs/ADRs.md` |
| Security review | ✅ | `SECURITY.md` (0 CRITICAL / 0 HIGH; F1 hardening applied) |
| Assumptions log | ✅ | `ASSUMPTIONS.md` |
| LLM-amortised + refinement (optional path) | ✅ | `src/arc_solver/search/llm_proposer.py`, validated with `MockClient` (no network) |

## Success criteria

| Criterion | Target | Result |
| --- | --- | --- |
| Tests green | all pass | ✅ 94 tests pass |
| Coverage on core packages | ≥ 80% | ✅ ~91% total; dsl/search/select/io/grid/eval all ≥ 83% |
| Offline end-to-end solves trivial tasks | yes | ✅ identity/flip/recolor/tile/gravity (M5 gate) |
| Real accuracy number on public eval | yes | ✅ ARC-AGI-1 train 9.25%, eval 3.00%; ARC-AGI-2 eval 0% @2s (see below) |
| Schema-valid offline submission, every task present | yes | ✅ `make_submission` + notebook; verified schema |
| No hardcoded secrets; keys env-only | yes | ✅ confirmed by security audit |
| Sandbox safety (LLM code) | no CRITICAL/HIGH | ✅ AST + builtins + timeout; escape tests pass |

## Honest results summary

| Dataset | pass@2 | train→test generalisation |
| --- | --- | --- |
| ARC-AGI-1 training (400) | 9.25% (37/400) | 37/39 ≈ **95%** |
| ARC-AGI-1 evaluation (400) | 3.00% (12/400) | 12/12 = **100%** |
| ARC-AGI-2 evaluation (120) | 0.0% @2s/task | — |

The near-perfect train→test generalisation among solved tasks is the central
evidence for the thesis: the MDL prior selects *generalising* programs. The
ARC-AGI-2 evaluation result is honestly low for the offline DSL — consistent with
the benchmark's deliberate resistance to single-rule search — and is the
motivation for the LLM-amortised and TTT extension points.

## Out of scope this session (user's last mile)
- Running the LLM-amortised / GPU-TTT variants at scale with the user's own key.
- Submitting the offline notebook on Kaggle and recording the leaderboard
  **submission ID** for the paper's Accuracy criterion.
