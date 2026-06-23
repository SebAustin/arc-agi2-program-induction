# Retrospective

A blameless review of how the build went, and the generalisable lessons.

## What went well
- **Thesis-first framing paid off.** Anchoring on a single defensible idea
  (test-time Bayesian program induction + MDL prior + pass@2-aware selection)
  kept every component motivated and gave the paper a clean spine across all six
  rubric criteria, not just accuracy.
- **The fitted proposer was the highest-leverage decision.** Pure enumeration
  hit a 2% wall (it could not reach depth-2 within budget). Adding analytic MAP
  proposals lifted ARC-AGI-1 training to 9.25% and — crucially — made runs fast
  enough to sweep all datasets.
- **Behavioural de-duplication** turned an intractable search into one that
  reaches real compositions, validated by solved depth-2 programs.
- **Strict separation of the offline core from the LLM path** kept the
  reproducible artifact pure and the tests network-free (MockClient).
- **Honesty discipline** (offline vs LLM-assisted numbers never conflated;
  ARC-AGI-2's low offline number reported plainly) makes the work credible.

## What was harder than expected
- **ARC-AGI-2 evaluation is brutal for compact symbolic search** — 0% at a 2s
  CPU budget. This is consistent with the benchmark's design, but it means the
  offline core's headline accuracy on the *target* set is near-zero; real
  ARC-AGI-2 accuracy needs the LLM/TTT extensions (the user's last mile).
- **Search budget vs sweep wall-clock** was the main throughput tension; the eval
  sweeps dominated the timeline. Smaller per-task budgets + the fitted fast-path
  were the mitigation.

## Lessons for the agency's shared practice
1. **For search/optimisation projects, build the cheap analytic shortcut before
   the general search.** It both raises coverage and unblocks fast iteration.
   Worth encoding in the solution rubric as a "fast-path before general-path"
   check.
2. **Make the reproducible/offline core a hard architectural boundary from day
   one** (no client imports, env-only keys, mock-tested) — it kept the security
   surface tiny and tests deterministic.
3. **Report the metric the grader uses, exactly.** Implementing normalised pass@2
   verbatim avoided any later reconciliation and made the selector design fall
   out of the metric.
4. **Security hardening should ship with regression tests.** The auditor's F1
   (AST screening) was only worth applying *with* escape-vector tests that fail
   if a future builtin reopens the hole.

## Follow-ups for the user
- Run the offline notebook on Kaggle; record the submission ID for the paper.
- Run the LLM-amortised path with a key to report the higher-accuracy variant.
- Grow the DSL against measured ARC-AGI-2 failure clusters; add the GPU-TTT
  proposer behind the existing `Proposer` interface.
