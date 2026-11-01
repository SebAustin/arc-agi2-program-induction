# SPEC — Object-centric DSL fitters for ARC-1 eval

Status: approved 2026-10-31. Terms: [GLOSSARY.md](GLOSSARY.md). Rationale:
[docs/ADRs.md](docs/ADRs.md) (ADR-009).

## Goal
Raise **pass@2 on the public ARC-AGI-1 evaluation split** (currently 3.00%,
`reports/arc1_eval.json`) by a real, measured, no-regression margin. The training
split (9.25%, `reports/arc1_train.json`) is a secondary check only.

## Constraints
- **Additive & backward-compatible.** No regression to the paper's claims, the
  Kaggle scoring contract, the public API, or the 122-test suite.
- **Focused & landable** this session; every gain measured, not asserted.

## Lever: L1 — broaden the DSL via analytic fitters
Add new rule *families*, each recognized by a **fitter** that infers the rule from
the demonstrations and emits a concrete program. Every new primitive is
**fitter-only** (`param_space = lambda ctx: []`) so it adds zero enumerative cost
and stays off the search critical path (ADR-009).

## Evidence — structural profile of the 388 unsolved ARC-1 eval tasks
| Family | Share of unsolved | Decision |
|---|---|---|
| F5 drawing & connectivity | 34.0% | stretch (coarse super-family, hardest to fit) |
| F2 grid-partition & panel ops | 17.3% | core |
| F4 object selection / counting | 16.8% | core |
| F1 per-component mapping | 12.4% | core |
| F3 periodic / symmetric completion | 7.5% | deferred (overlaps existing strength) |

~63% of unsolved tasks (F1 + F4 + F5) rest on one missing substrate: an object
layer with per-object attributes and placement. 0/388 were one current-primitive
away — the gap is genuinely missing families, not an enumeration bug.

## Build order
0. **Object substrate** — `grid/objects.py` + a `per_object` combinator (keystone).
1. **F2** grid-partition & panel ops *(independent of the substrate)*.
2. **F4** object selection / counting *(uses the substrate)*.
3. **F1** per-component mapping *(uses the substrate)*.
Stretch: the fitter-friendly slice of **F5** (interior flood-fill, outline/bbox
fill, connect collinear dots). Deferred: **F3**.

## Object model (Q8)
Fitters search a small set of object readings —
`{single-colour, colour-spanning} × {4-adjacency, 8-adjacency} × background∈{most-frequent, 0}` —
and choose the reading under which the demonstrations are consistent,
MDL-tie-broken toward the current default (single-colour, 4-adjacency,
most-frequent background). Existing primitives keep today's default unchanged.

## Family fitters
- **F2 panels** → `partition_select` / `partition_combine`. Detect full-line
  separators (whole rows/cols of one non-background colour); split into panels;
  fit a rule from {select a panel by index / uniqueness / colour-count; combine
  panels cellwise by and / or / xor / overlay}.
- **F4 selection/counting** → `select_object` / `count_grid`. Fit {select the
  object chosen by size rank / uniqueness / frequency, optionally cropped to its
  bounding box; or emit a small count grid}.
- **F1 per-object mapping** → `per_object` / `map_object_color`. Fit a per-object
  recolour or transform keyed by an object attribute (size rank, colour, shape
  class) or by an indicator object.

## Measure-and-keep protocol (the gate)
- Reuse `eval/harness.py`. Inner loop on a fixed, **seeded ~100-task** ARC-1 eval
  subsample; **full 400-task** eval for the headline number.
- Keep a family **only if** it is net-positive on the subsample.
- Every step, hard gate: `pytest` stays green (122+ tests); **no drop** on the
  previously-solved tasks (diff against `reports/arc1_eval.json`); per-task time
  budget comparable to the offline Kaggle environment.

## Success
A genuine, measured gain over 3.00% on ARC-1 eval with no regression.
Mid-single-digits is a good session. Fitters match *structure*, not guaranteed
rules — expect to convert a fraction of each matched pool into real solves.

## Out of scope
No LLM / TTT path; no Kaggle leaderboard submission; the submission-path
`try/except` robustness gap is noted and parked as a separate follow-up.

## Result (measured 2026-10-31)
ARC-AGI-1, original engine vs. all three families enabled. Reproduced with
`scripts/eval_ablation.py` against the committed core reports, and validated
against a full `max_programs=30000, max_seconds=10` run, which it reproduces
exactly (22/400 for F2+F1 on eval).

- **Evaluation pass@2: 3.00% → 5.75%** — 12 → 23 tasks (**+11**); F2 +8, F1 +2,
  F4 +1. Report: `reports/arc1_eval_families.json`.
- **Training pass@2: 9.25% → 13.50%** — 37 → 54 tasks (**+17**); F2 +10, F1 +6,
  F4 +1. Report: `reports/arc1_train_families.json`.
- **Zero regressions on both splits.** ARC-AGI-2 evaluation unchanged (0.0%,
  `reports/arc2_eval_families.json`).
- **F4 kept.** Narrow criteria (largest/smallest/unique) added 0; richer criteria
  (position, bounding box, uniqueness) recovered +1, so F4 is emitted.
- **`shape`→colour recolour key excluded** from F1: it memorised a per-object
  bijection that was train-consistent but shadowed a correct enumerative solve via
  the engine's early-stop (the sole train regression, `9565186b`). Low-capacity
  keys (size, colour, size-rank) are kept — the Occam discipline the prior encodes.
- Tests: 122 → 157 (+35), all green; coverage ~94%. Per-task runtime unchanged —
  the families are fitter-only, so the enumerative search is untouched (ADR-009).
