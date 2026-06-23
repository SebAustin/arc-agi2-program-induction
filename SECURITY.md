# Security Review — ARC-AGI-2 Solver

**Scope:** `src/arc_solver/**`, `tests/`, `notebooks/`, `pyproject.toml`.
**Method:** STRIDE decomposition + static review + empirical sandbox-escape probing +
secret scan + dependency posture review. All findings are evidence-based with real
`file:line` citations; severities are calibrated to the actual deployment model
(local, single-user research tool run with the operator's own API key).
**Date:** 2026-06-23.

---

## 1. System decomposition

### Trust boundaries
| # | Boundary | Trusted side | Untrusted side |
|---|----------|--------------|----------------|
| TB-1 | Task JSON on disk → `load_task` | solver core | task files (`data/`, user-supplied paths) |
| TB-2 | LLM API response → proposer → sandbox | solver core | model-emitted `transform(grid)` source |
| TB-3 | Environment → `config`/`llm_client` | solver core | API keys, provider/model selectors |
| TB-4 | Solver → `submission.json` / reports on disk | solver core | output path argument |

### Entry points
- CLI: `cli/solve.py`, `cli/run_eval.py`, `cli/make_submission.py` (argparse; local args).
- Library: `io/loader.load_task` / `load_dataset` (parses task JSON).
- LLM path: `search/llm_proposer.LLMProposer.propose` → `dsl/primitives_llm.py` →
  `dsl/sandbox.run_transform` (**executes model-authored Python**).

### Data stores / sensitive data
- API keys: environment only (`ARC_LLM_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).
  Never written to disk, never logged, never embedded.
- Task JSON: public ARC datasets; no PII.
- Outputs: `submission.json`, `reports/*.json`; no secrets.

### The one real security surface
The **offline path is pure numpy program search and carries no notable risk.**
The sole meaningful surface is `dsl/sandbox.py`, which `compile()` + `exec()`s
LLM-emitted Python under (a) static token screening, (b) a restricted-builtins
namespace, and (c) a SIGALRM wall-clock timeout. It is reached only via
`primitives_llm.py` (the `py_transform` op) which is reached only via
`llm_proposer.py`, i.e. only when the LLM path is explicitly enabled.

---

## 2. STRIDE summary

| Threat | Surface | Assessment |
|--------|---------|------------|
| **S**poofing | LLM endpoint, env keys | N/A multi-user. Keys env-only; TLS handled by vendor SDKs. No auth surface of our own. |
| **T**ampering | Task JSON (TB-1) | Strictly validated in `loader.validate_grid` (rectangular, ints 0–9, non-empty). Malformed input → `ValueError`. **Low.** |
| **R**epudiation | n/a | No multi-user audit requirement for a local tool. Out of scope. |
| **I**nformation disclosure | Sandbox (TB-2), keys (TB-3) | Keys never logged/serialized. Sandbox could in principle read files IF builtins escape succeeds — see F1; empirically not reachable. **Low (residual).** |
| **D**enial of service | Sandbox (TB-2), interpreter | Pure-Python infinite loops are killed by SIGALRM (verified, 1.0s). Unbounded *intermediate* allocation is NOT bounded (F2). Output grid is capped by `MAX_GRID_CELLS`. **Low/Medium.** |
| **E**levation of privilege | Sandbox (TB-2) | The core question: can model code escape restricted-exec to run OS commands? Empirically **no** with current builtins (F1). Defense-in-depth only — see threat model. |

---

## 3. Sandbox threat model and residual risk (read this first)

**The sandbox is a defense-in-depth guard, not a hard multi-tenant boundary.**
The code it runs originates from an LLM that *the operator themselves prompted*,
executed locally on the operator's own machine with the operator's own key and
file permissions. There is no privilege gradient to cross: anything the sandbox
could leak (local files, the API key in the operator's own env) is already
available to the operator. The realistic risk is a *buggy/hallucinated* transform,
not an adversary. Accordingly, escape findings are rated **MEDIUM at most**, and a
clean escape would be flagged **HIGH** (not CRITICAL) given the single-user,
own-key, own-machine model.

**Empirical escape testing performed (all attempts failed to reach `os`/`subprocess`):**
- Dunder pivots (`[].__class__.__base__.__subclasses__()`, `transform.__globals__`,
  `e.__traceback__`) — **blocked** by the `"__"` token screen (`sandbox.py:28`).
- Introspection builtins (`type`, `vars`, `dir`, `chr`, `bytes`, `format`, `print`,
  `help`, `getattr`) — **absent** from `_SAFE_BUILTINS` (`sandbox.py:46-52`), all
  raise `NameError`.
- Generator/frame pivots without dunders (`g.gi_frame.f_back`, `.f_builtins`) —
  the frame attribute *names* are NOT screened, but the escape fails because
  (a) `next` is not allow-listed, (b) a *suspended* generator frame has
  `f_back is None` in CPython, and (c) a generator's own `f_builtins` resolves to
  the **restricted** sandbox builtins, not the real ones. No allow-listed builtin
  yields a reference to a frame executing in the real module namespace.

**Residual risk (why this is defense-in-depth, not a proof):** the screen is a
substring denylist and the builtins allow-list is the *real* boundary. The frame
attribute names (`gi_frame`, `f_back`, `f_globals`, `f_builtins`, `cr_frame`,
`tb_frame`, `f_code`) are not on the denylist, so the security property rests
entirely on no allow-listed builtin handing back a live foreign frame. That holds
today, but it is a brittle invariant: adding any of `getattr`, `vars`, `type`,
`next`, `iter`, or `__import__`-adjacent helpers to `_SAFE_BUILTINS` would likely
reopen an escape. See F1 remediation for hardening that removes this fragility.

---

## 4. Findings

| ID | Severity | Title | Location |
|----|----------|-------|----------|
| F1 | MEDIUM | Sandbox relies on a substring denylist; frame-attribute names not screened | `dsl/sandbox.py:26-44, 70-81` |
| F2 | LOW | No bound on intermediate memory allocation inside sandbox | `dsl/sandbox.py:110-135` |
| F3 | LOW | Timeout is POSIX/main-thread only; silently absent elsewhere | `dsl/sandbox.py:123-134` |
| F4 | LOW | Code-fence extractor falls back to executing the *entire* completion | `search/prompt.py:56-63` |
| F5 | LOW | Substring screen over-blocks benign identifiers (correctness, not security) | `dsl/sandbox.py:59-63` |
| F6 | INFO | No prompt-injection hardening on model output (by design) | `search/llm_proposer.py:74-83` |
| F7 | INFO | No dependency vulnerability scanning configured | `pyproject.toml` |

### F1 — Sandbox is a substring denylist; the real boundary is the builtins allow-list (MEDIUM)
**Evidence.** `_FORBIDDEN` (`sandbox.py:26-44`) is a substring denylist; `_screen`
(`sandbox.py:59-63`) does a plain `token in source` check. It blocks `"__"`, `import`,
`getattr`, `os.`, `sys.`, `subprocess`, `socket`, etc. — but it does NOT block frame
introspection attribute names (`gi_frame`, `f_back`, `f_globals`, `f_builtins`,
`f_code`, `cr_frame`, `tb_frame`). Empirically these *do* parse and access, e.g.
`g.gi_frame.f_builtins` runs and returns a dict; it only fails to escalate because
that dict is the restricted namespace and `next`/the real foreign frame are
unreachable with the current allow-list. **No working OS-command escape was found.**
The exposure is structural: security depends on the allow-list staying minimal, with
the denylist providing no backstop for the frame-walk class of attacks.
**Realistic impact.** Low today (own-machine, own-key). The concern is regression:
a future contributor adding a convenience builtin (`getattr`, `type`, `next`, `vars`)
reopens a real arbitrary-file-read / `os`-import escape with no failing test to catch it.
**Remediation (defense in depth, in priority order):**
1. Replace substring screening with an **AST allow-list** (`ast.parse` then walk):
   reject `ast.Attribute` whose `.attr` starts with `f_`/`gi_`/`cr_`/`tb_` or is a
   dunder, reject `ast.Import`/`ast.ImportFrom`, and optionally allow only a known set
   of node types. This kills the entire frame-walk class regardless of builtins.
2. Add `"__builtins__": {...}` *and* explicitly set unused dangerous builtins to absent
   (already done) — keep the allow-list frozen and add a test asserting its exact contents
   so additions are deliberate.
3. Add regression tests for the frame-walk vectors (`gi_frame.f_back`, `f_builtins`,
   `tb_frame`) so any future builtins change that reopens them fails CI.
4. For a true boundary (only if requirements change to untrusted multi-tenant): run the
   transform in a separate process with `resource` rlimits + `seccomp`/subprocess
   isolation. Not warranted for the current local single-user model.

### F2 — No cap on intermediate memory allocation (LOW)
**Evidence.** `_to_grid` (`sandbox.py:84-107`) bounds the *returned* grid by
`MAX_GRID_CELLS`, and SIGALRM bounds wall-clock time, but a transform can allocate
unbounded memory *before* returning, e.g. `x = [0]*99999999` — verified to run and
return a (bounded) grid in ~0.3s without tripping any guard. A pathological allocation
could OOM the process within the 2s window.
**Impact.** Local self-DoS only; no cross-user impact.
**Remediation.** Set an address-space rlimit around execution
(`resource.setrlimit(RLIMIT_AS, ...)` in a child process), or shorten `timeout_s`.
Low priority for a research tool.

### F3 — Timeout is POSIX/main-thread only (LOW)
**Evidence.** `run_transform` guards SIGALRM with `hasattr(signal, "SIGALRM")`
(`sandbox.py:123`). On Windows, or when `run_transform` is invoked off the main thread,
the timeout is silently skipped and a runaway transform runs unbounded.
**Impact.** Local hang/self-DoS on non-POSIX or threaded callers.
**Remediation.** Document the main-thread/POSIX requirement; for portability, run the
transform in a child process with a join-timeout, or add a watchdog-thread fallback.

### F4 — Fenced-code extractor executes the whole completion on fallback (LOW)
**Evidence.** `extract_code` (`prompt.py:56-63`): if no ```` ``` ```` fence is found but
the text contains `"def transform"`, it returns `completion.strip()` — i.e. the entire
model message (prose, multiple defs, trailing statements) is handed to `compile`/`exec`.
The sandbox still constrains it, so this is not an escape, but it widens what gets
executed beyond the intended single function.
**Remediation.** Require a fenced block, or after the fallback parse with `ast` and keep
only the `transform` function definition; reject module-level side-effecting statements.

### F5 — Substring screen over-blocks benign identifiers (LOW, correctness)
**Evidence.** `"import" in source` (`sandbox.py:61`) rejects identifiers like
`important`; `"__" in source` rejects any legitimate dunder a model might emit. Verified:
a transform using a variable named `important` is wrongly rejected with
`SandboxError: forbidden token 'import'`.
**Impact.** Reduces solve rate (false negatives); not a security issue.
**Remediation.** Folded into F1: AST-based screening eliminates substring false positives.

### F6 — No prompt-injection hardening on model output (INFO, by design)
**Evidence.** `llm_proposer.py:74-83` takes model completions and `extract_code` →
`exec`. A task crafted to make the model emit malicious code is the threat, but inputs
are public ARC grids rendered as digits (`prompt._render_grid`), and execution is
sandbox-constrained. The sandbox (F1) is the mitigating control. No additional action
needed for the stated threat model; revisit if task inputs ever become untrusted/third-party.

### F7 — No dependency vulnerability scanning configured (INFO)
**Evidence.** Runtime deps are minimal and reputable (`numpy`; optional `openai`,
`anthropic`, `torch`, `matplotlib` — `pyproject.toml`). No `pip-audit`/`safety`/`bandit`
in the toolchain (none installed). No pinned lockfile.
**Remediation.** Add `pip-audit` to the `dev` extra and a CI step; add `bandit -r src/`
(it would flag the `exec` at `sandbox.py:75`, which is intentional and `# noqa: S102`-annotated).

---

## 5. Secret & input-validation verification (completed)
- **No hardcoded secrets.** Repo-wide grep for key/token/`sk-`/`AKIA`/`ghp_` patterns
  returned only env-var reads and doc comments. Keys are read exclusively from
  `os.environ` in `config.py:60-66` and `llm_client.py:43,70`; `OpenAIClient`/
  `AnthropicClient` raise if no key is present. The notebook contains no secrets.
- **`.gitignore` correctly excludes** `.env`, `.venv/`, `/data/`, `/reports/*.json`.
- **Boundary validation is strong** at TB-1: `loader.validate_grid` (`loader.py:21-46`)
  enforces non-empty, rectangular, `int` (rejecting `bool`), 0–9. `sandbox._to_grid`
  re-validates transform output identically. `submission.SubmissionWriter.add` rejects
  duplicate task IDs (`submission.py:31-32`).
- **No dangerous deserialization or shell-out:** no `pickle`/`yaml.load`/`marshal`/
  `os.system`/`subprocess`/`shell=True` anywhere in `src/` (the only `subprocess`
  occurrence is the denylist literal in `sandbox.py:40`).

---

## 6. Remediation status
- **F1 — FIXED (post-audit).** `dsl/sandbox.py` now runs an **AST screen**
  (`_ast_screen`) in addition to the substring pre-filter: it rejects
  `ast.Import`/`ast.ImportFrom`, every `ast.Attribute` whose name starts with `_`
  or is a frame/introspection attribute (`gi_frame`, `f_back`, `f_globals`,
  `f_builtins`, `f_code`, `tb_frame`, `cr_frame`, ...), and any `ast.Name`
  referencing a dangerous identifier (`eval`, `exec`, `getattr`, `type`, `vars`,
  `globals`, ...). This kills the entire frame-walk class regardless of the
  builtins allow-list. A **frozen-builtins regression test** asserts
  `_SAFE_BUILTINS` contains none of the risky names, and `tests/test_sandbox.py`
  now includes escape-vector tests (`gi_frame.f_builtins`, dunder pivots,
  `getattr`/`type` references) — all asserted to raise `SandboxError`. F5
  (substring false positives) is partially subsumed: the substring filter remains
  as a cheap first pass but the AST screen is the real boundary.
- **Ranked backlog (remaining):** F2/F3 (LOW — rlimit + portable timeout) →
  F4 (LOW — strict fence extraction) → F7 (INFO — add `pip-audit`/`bandit` to CI)
  → F6 (INFO — revisit only if task inputs become untrusted).

## 7. Bottom line
For its actual deployment — a local, single-user research tool run with the operator's
own key — the security posture is **sound**. No hardcoded secrets, strict input
validation at every boundary, no dangerous deserialization, and **no demonstrable
arbitrary-OS-command escape** from the LLM-code sandbox despite targeted probing. The
one substantive item (F1) is that the sandbox's safety currently rests on the
minimalist builtins allow-list while the denylist leaves the frame-introspection class
unscreened; moving to AST-based screening would make that boundary robust to future
edits. All other findings are LOW/INFO hardening.
