# Glossary

The shared vocabulary of the solver's problem domain. Definitions here describe
*what a term means*, never how it is implemented — implementation lives in the
code and in [SPEC.md](SPEC.md); design rationale lives in [docs/ADRs.md](docs/ADRs.md).

- **Task** — An ARC problem: a small set of demonstrations plus one or more test
  inputs whose outputs must be produced.

- **Demonstration** — One input→output example within a task, showing the
  transformation to be inferred. (Also: *train pair*.)

- **Pass@2** — The scoring rule. A test input counts as solved if *either* of the
  two submitted output grids exactly matches the hidden answer. A task's score is
  the mean over its test inputs; a dataset's score is the mean over its tasks.

- **Attempt** — One of the two output grids submitted for a test input.
  *attempt_1* is the single most probable answer; *attempt_2* is a deliberately
  different hedge, so the two guesses cover distinct hypotheses.

- **Admissible program** — A candidate transformation that reproduces *every*
  demonstration's output exactly. ARC demonstrations are noise-free, so a single
  mismatch disqualifies a candidate; only admissible candidates are considered.

- **Fitter** — A recognizer that infers a candidate transformation directly from a
  task's demonstrations, rather than by blindly searching the space of
  transformations.

- **Background** — The colour treated as empty canvas and excluded from objects.
  By default the most frequent colour; plain black (0) is the common alternative
  reading.

- **Object** — A maximal connected region of cells that the solver treats as a
  single thing. A task fixes neither the adjacency (orthogonal only, or also
  diagonal) nor whether an object may span more than one colour; the correct
  reading is whichever one makes that task's demonstrations consistent.

- **Object model** — A specific choice of how objects are read from a grid: the
  adjacency rule, whether objects are single-colour or colour-spanning, and which
  colour is background.

- **Per-object mapping** — A transformation applied to each object independently
  (for example, recolour each object according to its size rank) rather than to
  the grid as a whole.

- **Panel** — One sub-grid produced when a grid is divided by separator lines into
  a regular arrangement of cells.

- **Separator** — A full row or column of a single non-background colour that
  partitions a grid into panels.
