# Architecture Decision Records

Short records of the load-bearing design choices and their rationale.

## ADR-001 — Linear pipeline DSL over a full lambda calculus
**Decision.** Programs are linear pipelines of typed `Grid -> Grid` ops;
object-level reasoning is encapsulated inside primitives (e.g. `keep_largest`),
not exposed as a separate value type.
**Why.** A linear program makes the description length trivial to compute and the
search space enumerable in MDL order. A full AST/lambda calculus is more
expressive but explodes the search and complicates the prior. We recover much of
the lost expressivity by letting primitives do object-internal work and by
allowing one analytic depth-2 composition tier in the fitted proposer.
**Trade-off.** Some genuinely higher-order tasks are unreachable; this is the main
expressivity ceiling and is documented as such.

## ADR-002 — Exact-match likelihood; admit only all-train-consistent programs
**Decision.** A program enters the posterior iff it reproduces **every** train
output exactly. `partial_score` exists only to steer refinement.
**Why.** ARC demonstrations are noise-free, so the Bayesian likelihood is a hard
0/1 constraint. Admitting partial fits would let overfit/special-cased programs
into the posterior and destroy the generalisation guarantee.

## ADR-003 — MDL prior as the posterior ranking
**Decision.** `P(program) ∝ 2^(-description_length)`; among admissible programs
the posterior is the prior. Enumeration expands in increasing description length.
**Why.** Occam/Solomonoff: the shortest consistent program is the one least able
to encode coincidences, hence most likely to generalise. Empirically the
train→test generalisation rate among solved tasks is very high, validating this.

## ADR-004 — Behavioural de-duplication in enumerative search
**Decision.** Programs are keyed by the tuple of outputs they produce on the train
inputs; only the first (cheapest) representative of each behaviour is kept and
expanded.
**Why.** This collapses the search from "all programs" to "all distinct
behaviours", which is the single change that makes bounded-CPU search find
depth-2+ compositions within seconds. Programs that map any train input to `None`
are never expanded (a `None` can't be repaired downstream).

## ADR-005 — One `Proposer` interface for offline search, LLM, and TTT
**Decision.** Every backend yields candidate `Program`s; the engine verifies and
selects identically regardless of source. LLM-emitted Python is wrapped as a
single high-cost `py_transform` op.
**Why.** Keeps the offline core pure and reproducible while letting an LLM or a
fine-tuned model act as an *amortised proposal distribution* for the same
posterior. The high MDL cost of `py_transform` means a short symbolic program
always wins when both are admissible — the neural proposer adds coverage without
corrupting the Occam ranking.

## ADR-006 — pass@2-aware selector (mode + disagreement)
**Decision.** `attempt_1` = posterior mode over *outputs*; `attempt_2` = most
probable output that disagrees with attempt_1.
**Why.** pass@2 rewards covering the truth with either of two grids. Aggregating
posterior mass by output grid (not by program) and forcing the second guess to
disagree maximises expected pass@2 on ambiguous tasks. Derived directly from the
metric rather than bolted on.

## ADR-007 — Sandboxed `exec` for LLM code, env-only API keys
**Decision.** LLM-emitted `transform` functions run under static token screening +
restricted builtins + a SIGALRM timeout. API keys are read only from environment
variables; SDKs are lazy-imported.
**Why.** Executing model-authored code needs defence in depth even in a
single-user local setting. The offline core never imports the sandbox or any
client, so the reproducible path has zero added attack surface. See `SECURITY.md`
for the threat model and residual risk.

## ADR-008 — Immutable, hashable grids (`tuple[tuple[int,...],...]`)
**Decision.** Grids are tuples of tuples; numpy is used only inside primitives.
**Why.** Hashability lets the selector bucket candidates by output grid directly;
immutability guarantees no primitive mutates a shared grid. The interpreter
converts to numpy once and back at the boundary.

## ADR-009 — Object layer via a `per_object` combinator (revisits ADR-001)
**Decision.** Introduce a first-class object layer — connected-component
extraction with per-object attributes (colour, size, bounding box, position) and
a small search over *object models* (4- vs 8-adjacency, single-colour vs
colour-spanning, background = most-frequent or 0) — and expose it to the DSL
through a `per_object(inner_op, selector, model)` combinator plus a few
object-aware primitives (`select_object`, `count_grid`, per-object recolour). The
combinator keeps the program's external contract `Grid -> Grid`; the object
collection is internal to it. All new object-aware and panel primitives are
**fitter-only** (`param_space = lambda ctx: []`), so the enumerative search space
and its budget are unchanged.
**Why.** A structural profile of the 388 unsolved ARC-AGI-1 evaluation tasks shows
the dominant signature is objectness and in-place construction: ~77% change
connected-component count, ~48% add foreground ink, and 0% admit a global colour
bijection, while the DSL is almost entirely global whole-grid transforms. ~63% of
unsolved tasks (per-object mapping + selection/counting + drawing) rest on the same
missing substrate. ADR-001 deliberately kept objectness inside individual
`Grid -> Grid` primitives to keep the pipeline flat; that is precisely the
expressivity ceiling ADR-001 anticipated. Promoting objects to a combinator-level
abstraction removes the ceiling for the highest-leverage families while preserving
the linear-pipeline prior.
**Trade-off.** This reopens ADR-001's "no separate object value type"
simplification. We bound the cost by keeping the object type **internal** to the
combinator (the external program stays `Grid -> Grid`, so serialization, the MDL
prior, and the selector are untouched), by keeping the new primitives off the
enumerative path, and by searching only a handful of object models per task.
Alternatives rejected: (a) stay fully flat — per-object composition stays
impossible, capping the per-object-mapping family; (b) a first-class object value
type threaded through the whole pipeline — larger surface, risks the
backward-compatibility this work requires. Each family is gated on a measured,
net-positive, no-regression result, so an abstraction that fails to pay for itself
is dropped rather than kept.
