# Code Review & Enhancement Pass

A structured review of the `arc-solver` codebase conducted by the AI Project
Agency, with the worthwhile improvements applied in place on the
`enhance/review-fixes` branch. Every fix preserves existing behaviour (the offline
pass@2 numbers) except where it strictly *adds* a hedge; see the regression check
at the bottom.

Three independent specialist reviews ran in parallel over the algorithmic core,
the security-sensitive sandbox, and the test suite. Each finding below was
re-verified against the code (and, where possible, reproduced) before being
accepted — confident-but-wrong findings were rejected.

## Applied fixes

| # | Severity | Area | What & why |
|---|----------|------|------------|
| 1 | High | `select/selector.py` | **`attempt_2` could equal `attempt_1`.** `_fallback_alt` returned the input unchanged for any grid invariant under both mirror and 180° rotation (a solid colour, a palindrome), silently duplicating the first guess and wasting the second pass@2 attempt. Reproduced directly. Fixed with a transform ladder that falls back to a deterministic one-cell perturbation for the fully-symmetric case; guaranteed distinct. Regression test added. |
| 2 | Medium | `select/selector.py` | **Posterior "mode" was order-dependent under exact ties.** `_output_mass` ranked tied-weight outputs by enumeration order, so `attempt_1` was not a well-defined function of the posterior. Added a canonical (lexicographically-smallest grid) tie-break for reproducibility. Determinism test added. |
| 3 | Medium | `dsl/sandbox.py` | **`str.format` dunder-smuggling defeated both sandbox screens** (re-audit finding). A model-emitted transform could build `"__class__"` from `"_"` pieces inside a format template and let `str.format` perform the attribute traversal — invisible to both the substring screen (no literal `"__"`) and the AST screen (dunder lives in an `ast.Constant`). Reproduced. Fixed by denying the `format`/`format_map`/`vformat` vehicle plus screening string literals for `"__"`/frame markers. Regression test added. See `SECURITY.md` F8. |
| 4 | Medium (hardening) | `dsl/primitives_shape.py`, `dsl/primitives_arith.py` | **Numpy views returned instead of copies** (`crop_to_content`, `trim_border`, `_halves`) violated `grid_ops.py`'s documented "returns new arrays" invariant. No active corruption today (no downstream primitive mutates in place), but a latent aliasing hazard; `.copy()` added (cheap — grids are capped at 2 500 cells). |
| 5 | Medium | `search/fitted.py` | **`_fit_scale` indexed `pair.input[0]` before its `ih == 0` guard**, raising `IndexError` on an empty grid instead of returning `None`. Reachable only via a directly-constructed `Task` (the loader rejects empties), but the defensive guard is free. |
| 6 | Low | `tests/` | **Lint gate was red** — three unused imports (`conftest.py`, `test_search.py`, `test_select.py`) meant `ruff check` failed even though `pytest` passed. Removed; the gate is now green (tests + lint + format). |
| 7 | — | `tests/` | **+19 tests (94 → 113).** Filled the highest-value behavioural gaps: selector fallback/tie-break/all-agree, engine dedup & early-stop & cross-run determinism, the sandbox timeout kill-switch (previously `pragma: no cover`), MDL per-type param formula, verifier `None`-output path, and the under-tested DSL primitives (`overlay_*` 36% → 93%, colour ops 60% → 96%). Total coverage 90% → 94%. |

## Deferred (documented, not applied) — with rationale

- **Engine train-signature dedup keeps only the cheapest program**
  (`search/engine.py`). A reviewer flagged that two programs identical on the
  train pairs but divergent on the *test* input are collapsed, losing a potential
  `attempt_2` hypothesis. This is a genuine deviation from the pure Bayesian
  ideal — but keeping the single cheapest per train-behaviour is a defensible
  MAP-style approximation: train-equivalent programs are unbounded and dominated
  by junk (append any input-noop suffix) that the MDL prior correctly
  down-weights exponentially. Keeping top-K per bucket is plausible but could
  inject test-divergent noise; it changes results either way and needs an
  empirical A/B on the eval set before adoption. **Recommend** as a measured
  experiment, not a blind change.
- **`op_selection_bits()` reads global mutable `PRIMITIVES` at scoring time**
  (`dsl/registry.py`). Description length shifts by one primitive's worth if
  `primitives_llm` is imported into the same process. The offline path never
  imports it, so offline reproducibility is intact; impact is confined to mixed
  offline/LLM processes. Low priority; fixing means freezing the registry count,
  which has import-order side effects.
- **`loader.validate_grid` doesn't enforce `MAX_GRID_CELLS`/`MAX_GRID_DIM`.** The
  size guard lives in the interpreter, not at the JSON boundary. Real ARC grids
  (≤30×30) are far under the cap, and inputs are trusted local files, so both
  specialist agents rated this INFO. Left as-is to avoid scope creep.
- **Sandbox residuals F2/F3/F4** (no memory rlimit; POSIX-only timeout; lenient
  fence extraction) and **F7** (no `pip-audit`/`bandit` in CI) — all LOW/INFO,
  unchanged from the prior audit and consistent with the local single-user,
  own-key deployment model. The sandbox is defence-in-depth, not a hard boundary.

## Regression check

The only applied change that can alter offline results is the canonical tie-break
(#2) — and only in the rare case of a ≥3-way exact MDL+mass tie, since pass@2
scores "attempt_1 **or** attempt_2 matches" and a 2-way tie leaves both outputs in
the top-2 regardless of order. Verified against the full ARC-AGI-1 training set
(400 tasks, `--max-seconds 5`) before vs. after:

- **Before (original code):** pass@2 = **9.50%** (38 tasks solved)
- **After (this branch):** pass@2 = **9.50%** (38 tasks solved)
- **Solved-set delta:** none lost, none gained; **zero per-task score changes**.

The applied fixes are confirmed non-regressing: `attempt_1` is untouched and the
fallback/tie-break changes altered no submission on the full training set. (The
independent `solution-verifier` separately returned **SOLID**: 114 tests green,
lint/format clean, 94% coverage, all four fix claims reproduced by execution.)
