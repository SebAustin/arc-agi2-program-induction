<!-- ARC Prize 2026 Paper Track — write-up. Target ≤ 1500 words (body). -->

# Test-Time Bayesian Program Induction for ARC-AGI-2

### Inducing a short, generalising program per task from the demonstrations alone — with a posterior selector built for pass@2

## 1. Problem and thesis

ARC-AGI-2 is designed to defeat memorisation: every evaluation task is novel, and
only *core-knowledge* priors (objectness, geometry, counting, symmetry) are
assumed. Following Chollet, the quantity that matters is not skill but
**skill-acquisition efficiency** — how much a system can learn about a brand-new
task from a handful of demonstrations and minimal priors.

We take that definition literally. We treat each task as **test-time Bayesian
program induction**: from the demonstration pairs alone, infer a short program
over a compositional grid DSL that reproduces *all* of them; weight the admissible
programs by a Minimum-Description-Length (MDL) prior; and, because the metric is
**pass@2**, emit the posterior-**mode** output as `attempt_1` and the most
probable **disagreeing** output as `attempt_2`. The system carries **no learned
weights and needs no network**: every task-specific fact is acquired at inference
from the demonstrations. Generalisation comes from Occam's razor, not from scale.

## 2. Relation to prior work

The strongest ARC approaches cluster into test-time training (MindsAI, the
ARChitects), evolutionary LLM program synthesis (SOAR), tiny recursive networks
(TRM), and pretraining-free compression (CompressARC). Greenblatt's GPT-4o
program sampling and Imbue's code-evolution show that *generate-verify-refine*
loops are powerful. Our contribution is not a new primitive but a **unifying
probabilistic frame**: (i) make the MDL prior the explicit object of inference, so
"which consistent program?" has a principled answer; (ii) treat symbolic search
*and* an LLM as two proposal distributions for the **same** posterior; and (iii)
derive the two-guess selector **from the pass@2 metric** rather than submitting
the top-2 programs. Where CompressARC compresses with gradient descent over a tiny
network, we compress by searching an explicit program space — and we add a
metric-aware selection step that the pass@2 objective specifically rewards.

## 3. Method

**Hypothesis space (DSL).** A program is a *linear pipeline* of typed
`grid → grid` primitives. ~30 primitives span the core-knowledge families:
geometry (rotations, reflections, transpose), colour (recolour, full colour-map,
swap), objectness (connected components → keep-largest/smallest, remove, gravity),
shape (crop-to-content, tile, scale, mirror-concatenate, symmetry repair), and
split-combine arithmetic (half-overlay AND/OR/XOR). Object reasoning lives *inside*
primitives, so the pipeline stays linear — enumerable and cheap to score — while
still composing. The interpreter threads a numpy grid through the pipeline and
returns `None` on any type/shape/runtime failure, with hard size guards.

**Posterior.** For demonstrations $D=\{(x_i,y_i)\}$ and program $p$,
$P(p\mid D)\propto P(D\mid p)\,P(p)$. The likelihood is **exact-match** — $1$ iff
$p(x_i)=y_i$ for *every* $i$, else $0$ — because ARC demonstrations are noise-free;
this makes admissibility a hard constraint. The prior is the **description
length**: $P(p)\propto 2^{-L(p)}$, with $L(p)$ summing an op-selection code, each
primitive's intrinsic cost, and its parameter bits. Hence **among admissible
programs the posterior is just the MDL prior** — the shortest consistent program
is the MAP explanation.

**Two proposers, one interface.** A `Proposer` yields candidate programs; the
engine verifies and selects, regardless of source.

- *Fitted proposer* — reads the demonstrations and emits MAP shortcuts directly:
  every single-transform hypothesis, a colour-map fitted from a consistent
  cell-wise bijection, tiling/scaling fitted from the size ratio, symmetry repair,
  constant output, and a few depth-2 `transform∘recolour` compositions. Each is
  $O(1)$ to construct and verified instantly.
- *Enumerative proposer* — best-first search expanding programs in **increasing
  description length**, so the cheapest hypotheses come first. The key to
  tractability is **behavioural de-duplication**: programs are keyed by the tuple
  of outputs they produce on the train inputs, so the search collapses from "all
  programs" to "all distinct behaviours", reaching depth-2/3 compositions within
  seconds on a CPU.

The **LLM proposer** (optional, behind the same interface) is an *amortised*
approximation of the same posterior: it emits Python `transform` functions —
sandboxed and wrapped as a single high-cost DSL op — refined with execution
feedback (which train pairs failed). Its large description length means a short
symbolic program always outranks it when both fit, so it extends coverage without
corrupting the Occam ranking.

**pass@2 selector.** Aggregate posterior mass by *predicted output grid*:
$W(g)=\sum_{p:p(x)=g}P(p\mid D)$. Then `attempt_1`$=\arg\max_g W(g)$ (the mode —
many distinct programs that agree on one grid reinforce it) and `attempt_2` the
heaviest grid that **disagrees** with `attempt_1`. Forcing disagreement is the
point: under pass@2 a second guess identical in spirit to the first wastes the
hedge; the optimal second guess covers the most probable *alternative* hypothesis.

## 4. Results

The offline path (fitted + enumerative) is single-threaded CPU, deterministic, no
pretraining, no network. We score the competition's normalised pass@2.

| Dataset | Tasks | pass@2 | train-solve |
| --- | --- | --- | --- |
| ARC-AGI-1 training | 400 | **13.50%**† | 13.50% |
| ARC-AGI-1 evaluation | 400 | **5.75%**† | 5.75% |
| ARC-AGI-2 evaluation | 120 | **0.0%** | 0.0% |

† With the object/panel fitter families (below); the enumerative-only core scores
9.25% (training) and 3.00% (evaluation). ARC-AGI-2 evaluation is unchanged, and
both re-measured ARC-AGI-1 splits are regression-free.

The most informative number is not the headline rate but the **gap between
train-solve and pass@2**: on ARC-AGI-1 training, 39/400 tasks admit a program and
**37 of those also solve the held-out test input — ~95% train→test
generalisation**. Admissibility-plus-brevity is, in this domain, already close to
correctness. Solved tasks span diverse families and include genuine depth-2
compositions discovered by search — e.g. `keep_largest | crop_to_content`,
`crop_to_content | tile`, and full colour bijections — not just single
transforms. The offline core solves **none** of the 120 ARC-AGI-2 evaluation
tasks at a 2 s/task CPU budget — expected and informative: ARC-AGI-2 was
engineered to resist single-rule search, so its tasks demand multi-step,
context-dependent programs whose shortest description exceeds a compact DSL in a
few seconds. This is the clearest motivation for the amortised LLM proposer and
the TTT extension, which attack exactly that depth/expressivity ceiling.

**Object and panel fitter families.** A structural audit of the unsolved
ARC-AGI-1 evaluation tasks found their dominant signature is *objectness and
in-place construction*, not the global whole-grid transforms the core favours.
Three analytic, *fitter-only* families close part of that gap — emitted
analytically and verified, never enumerated, so the search budget and per-task
runtime are unchanged. *Grid-partition / panel ops* (F2) detect separators and
either select a panel or combine panels cellwise via a truth table learned from
the demonstrations (generalising AND/OR/XOR to $k$ panels). *Per-object mapping*
(F1) recolours or transforms each object independently, keyed by a low-capacity
attribute (size, colour, size-rank). *Object selection / counting* (F4) picks an
object by a learned criterion (size, position, uniqueness, bounding box) and crops
or counts. Together they lift ARC-AGI-1 **evaluation** 3.00%→**5.75%** (12→23,
+11) and **training** 9.25%→**13.50%** (37→54, +17), with **zero regressions** on
both splits. One excluded case is instructive: a high-capacity *shape→colour* key
fit the training objects by memorisation and shadowed a correct enumerative solve
through the engine's early-stop — dropped in favour of the generalising keys,
exactly the Occam discipline the prior encodes.

**Ablations.** Removing the fitted proposer leaves the enumerator unable to reach
colour-bijection and exact-scale tasks within budget; removing behavioural
de-duplication prevents the enumerator from reaching depth-2 at all. Replacing the
disagreement rule for `attempt_2` with "second-best program" loses the hedge on
ambiguous tasks (the residual cases where two short rules fit train but differ on
test). Each component earns its place.

## 5. Why it works

The MDL/Solomonoff argument: demonstrations under-determine the rule, and the gap
between *fitting* and *predicting* is the generalisation problem. A program that
special-cases a memorised output pays for it in bits and is dominated by any
shorter rule; a program that captures the genuine regularity is short *because*
the regularity is real. Preferring the shortest consistent program therefore
prefers the explanation least able to encode coincidences — which is exactly why
the observed train→test generalisation is so high. The failures the prior cannot
fix are tasks with genuine demonstration **ambiguity**; those are precisely what
the pass@2 selector hedges. The method's ceiling is thus the **expressivity of the
DSL** and the **depth reachable under budget**, not the principle.

## 6. Universality, limitations, and progress

The frame is task-agnostic: *define a prior over programs, verify against
demonstrations, select for the metric.* Swap the DSL and the same machinery
applies to string/list transforms, Sudoku, mazes, or spreadsheet edits — the LLM
proposer needs no change at all. **Limitations:** the compact DSL misses
higher-order and heavily contextual rules; bounded CPU search caps composition
depth; the LLM path needs an API key and cannot run in the no-internet Kaggle
sandbox (so it serves the verified/external track, not the constrained
leaderboard). **For progress toward a top score**, the contribution is a clean,
reusable scaffold that separates *what to propose* (a growable prior — richer
primitives, learned proposal distributions, test-time training) from *how to
decide* (exact-match posterior + MDL + metric-aware selection), and a discipline
of reporting what was **acquired** at test time versus **pre-bought** by
pretraining. Both the offline core and the LLM/TTT extension points are open,
documented, and reproducible.

---

*Code & reproducibility:* public notebook (`notebooks/kaggle_submission.ipynb`,
offline) and repository with 157 tests; see `docs/THEORY.md` for the extended
derivation and `reports/` for per-task tables. Leaderboard submission ID: _to be
recorded on Kaggle run_.
