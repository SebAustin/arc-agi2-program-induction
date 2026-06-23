# Why it works: an extended theoretical account

The 1,500-word write-up (`WRITEUP.md`) states the thesis compactly. This appendix
gives the longer argument for *why* test-time Bayesian program induction is the
right shape of solution for ARC-AGI-2, and what each design choice buys.

## 1. ARC measures skill-acquisition efficiency, not skill

Chollet's *On the Measure of Intelligence* defines intelligence as
**skill-acquisition efficiency over a scope of tasks, relative to priors,
experience, and generalisation difficulty.** ARC is engineered so that:

- each task is novel to the solver's designer (developer-aware generalisation);
- only **core knowledge priors** are assumed (objectness, elementary geometry,
  counting, symmetry) — not acquired knowledge like language or web text;
- memorisation is useless because the evaluation tasks are disjoint from anything
  seen before.

A system that scores well by *pre-buying* skill — a large model pretrained on
millions of ARC-like examples — is, in Chollet's accounting, converting prior and
experience into apparent skill. It can be highly *capable* without being, by this
definition, *intelligent*. The interesting question ARC poses is therefore: **how
much can be acquired, per task, from a handful of demonstrations and minimal
priors?**

Our method answers that question literally. It carries no learned weights. Its
only prior is a small library of core-knowledge operations (the DSL) and a
preference for simplicity (the MDL prior). Everything task-specific is acquired at
test time by inference over that prior. The quantity it optimises *is*
skill-acquisition efficiency.

## 2. Induction as posterior inference

Let a task provide demonstrations $D = \{(x_i, y_i)\}_{i=1}^n$. We posit a latent
program $p$ (a transformation of grids) and seek the posterior

$$ P(p \mid D) \;\propto\; P(D \mid p)\, P(p). $$

Two modelling choices make this tractable and meaningful:

- **Likelihood is exact-match.** $P(D \mid p) = 1$ if $p(x_i) = y_i$ for *every*
  $i$, else $0$. ARC demonstrations are noise-free, so a correct rule must
  reproduce them exactly. This collapses the likelihood to a hard admissibility
  constraint: a program is either consistent with all demonstrations or it is out.

- **Prior is the description length.** $P(p) \propto 2^{-L(p)}$, where $L(p)$ is
  the number of bits to encode $p$ (op-selection bits + per-op cost + parameter
  bits). This is the universal/Occam prior restricted to our DSL: simpler
  programs are exponentially more probable a priori.

Consequently, **among admissible programs the posterior is just the MDL prior** —
the shortest consistent program is the maximum-a-posteriori explanation.

## 3. Why MDL generalises (and why it matters here)

The MDL/Solomonoff argument: a hypothesis that compresses the data well is, with
high probability, close to the data-generating process. In ARC terms, the
demonstrations under-determine the rule — many programs fit $D$ — and the gap
between *fitting* $D$ and *predicting* the held-out test output is exactly the
generalisation problem. Preferring the shortest consistent program is preferring
the explanation least able to encode coincidences:

- A program that "cheats" (e.g. memorises a train output and special-cases it)
  pays for that special case in bits and is dominated by any shorter rule.
- A program that captures the genuine regularity is short *because* the
  regularity is real.

Empirically we observe a striking consequence: on our development runs, almost
every program that is admissible (fits all train pairs) also produces the correct
held-out test output. The train→test generalisation rate among solved tasks is
very high. That is the MDL prior doing its job — admissibility plus brevity is
already, in this domain, close to correctness. The few failures are tasks with
**genuine demonstration ambiguity**, where two short rules fit the train pairs but
disagree on test — which is precisely what the pass@2 selector is designed for.

## 4. The selector is derived from the metric

ARC-AGI-2 scores **pass@2**: a test input counts as solved if *either* of two
submitted grids is correct. The Bayes-optimal pair is not "the top two programs"
but the two *outputs* that maximise the probability of covering the truth.

Aggregate posterior mass by predicted output grid:
$W(g) = \sum_{p:\,p(x)=g} P(p \mid D)$. Then

- `attempt_1` $= \arg\max_g W(g)$ — the posterior **mode** over outputs (many
  distinct programs that agree on the same grid reinforce it).
- `attempt_2` $= \arg\max_{g \neq \text{attempt\_1}} W(g)$ — the most probable
  **disagreeing** output.

Choosing the second attempt to *disagree* is the key: under pass@2, a second guess
identical-in-spirit to the first wastes the hedge. The right second guess covers
the most probable alternative hypothesis. This directly targets the marginal
distribution over test outputs rather than over programs — a small but principled
distinction that the metric rewards exactly when the task is ambiguous.

## 5. Neural and symbolic proposers as one posterior

The same posterior can be approached two ways, both implemented behind one
`Proposer` interface:

- **Enumerative search** walks the hypothesis space in increasing description
  length, so it *is* sampling from the prior in MAP order. Behavioural
  de-duplication (keying programs by their outputs on the train inputs) reduces
  the search from "all programs" to "all distinct behaviours", which is what makes
  bounded CPU search feasible.

- **An LLM is an amortised proposal distribution.** Trained on code and reasoning,
  it concentrates probability on the programs a competent solver would write —
  an importance sampler for the same posterior. Its proposals are sandboxed,
  wrapped as a single DSL op, and verified by the *identical* admissibility test;
  execution feedback (which train pairs failed) drives refinement, i.e. a few
  steps of posterior-guided resampling. Because an LLM program carries a large
  description length, whenever a short symbolic program is also admissible the
  posterior prefers the symbolic one — the neural proposer extends coverage
  without corrupting the Occam ranking.

This is the sense in which the method is *universal*: the framework is a
task-agnostic recipe — define a prior over programs, verify against
demonstrations, select for the metric — and ARC is one instantiation. The same
machinery applies to any few-shot program-induction problem (string and list
transformations, Sudoku, mazes, spreadsheet-style edits) by swapping the DSL.

## 6. What this predicts, and the limitations it implies

The account makes falsifiable predictions that match what we see:

1. **High train→test generalisation among solved tasks** — confirmed.
2. **Failures concentrate where the DSL lacks a primitive** (the regularity is not
   expressible in few bits) or where **demonstrations are genuinely ambiguous**
   (two short rules disagree on test) — the latter is the residual that pass@2,
   not the prior, must address.
3. **Adding the LLM proposer raises coverage on compositional tasks** the bounded
   enumerator cannot reach, without changing which program wins when both find a
   fit.

The ceiling of the offline core is therefore the *expressivity of the DSL* and the
*depth reachable under budget*, not the principle. ARC-AGI-2's deliberate
resistance to single-rule search shows up as exactly this: many of its tasks
require multi-step, context-dependent programs whose shortest description still
exceeds what a compact DSL and a few seconds of CPU can reach. The path forward is
to grow the prior (richer primitives, learned proposal distributions) while
keeping the inference and selection machinery — and the honesty about what was
acquired versus pre-bought — unchanged.
