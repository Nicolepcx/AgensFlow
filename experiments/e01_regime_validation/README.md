# Experiment 01 — Regime validation

## Hypothesis

The AgensFlow policy layer (regime detection + activation planning) makes
*meaningfully different* orchestration choices for *meaningfully different*
task types, and those choices produce *measurably better* (success, cost)
trade-offs than (a) a naive single-call baseline, and (b) the wrong
forced-regime configuration.

This is the chunk-3 first empirical test. It is small (12 tasks, single
trial per cell) and explicitly preliminary — the goal is to demonstrate that
regime-conditioning does real work, not to produce a paper-grade benchmark.

## What this experiment is *not*

- It is not a test of agent or LLM capability. The same models (claude-haiku-4.5
  for solver/verifier, gpt-5.4-nano for planner/memory/evaluator) are used
  across all configurations, isolating the policy as the variable under test.
- It is not a test of branching coalitions. Branching runtime is a later
  chunk; the configurations here are linear plans only.
- It is not statistically powered. With N=12 and single trials, we report
  honest descriptive statistics, not significance.

## Design

### Benchmark

12 tasks across 3 categories:

- **Category A — Simple Q&A, no evidence needed (4 tasks)**
  Should map to `straightforward` regime. Running `evidence_heavy` here
  should waste tokens with no quality benefit.
- **Category B — Document-grounded Q&A, evidence needed (4 tasks)**
  Should map to `evidence_heavy` regime. Running `straightforward` should
  fail on grounding.
- **Category C — Adversarial / missing evidence (4 tasks)**
  Should map to `evidence_heavy` regime. Documents are provided but do not
  contain the answer. The verifier should catch this; naive baseline is
  expected to confabulate.

### Configurations

For each task, four configurations are run:

1. `naive` — single LLM call (claude-haiku-4.5), task and documents inlined.
2. `agensflow_forced_straightforward` — full pipeline, regime forced.
3. `agensflow_forced_evidence_heavy` — full pipeline, regime forced.
4. `agensflow_auto` — full pipeline, regime detected from task features.

### Metrics

Per (task, configuration):
- `total_tokens` — sum across all calls (including failed validation retries).
- `validation_retries` — count of trace events with `error` set.
- `latency_seconds` — wall-clock total.
- `judgment` — grader's verdict: `success`, `partial`, or `failure`.
- `flagged_missing_evidence` — for Category C: did the answer correctly say
  the documents do not contain the answer?

### Headline metric

**Tokens per successful task** = (sum of total_tokens) / (count of successful
runs), reported per (configuration, category) cell.

### Policy-specific finding

The most policy-centric measurement: **does `agensflow_auto` match the best
forced configuration in each category?** If yes, the policy is choosing the
right plan. If no, the policy is mis-routing.

## Reproducibility

- All tasks, documents, prompts, and rubrics are in this directory and
  committed to the repo.
- The grader uses claude-haiku-4.5 (same family as the agents under test —
  acknowledged limitation; future work should use a stronger out-of-mix
  grader).
- Single trial per cell. LLM variance is real and not quantified here.
- Predictions are pre-registered in `RESULTS.md` before any run.

## How to run

From the repo root:

```
python -m experiments.e01_regime_validation.run
```

The run takes a few minutes and costs roughly $1-3 in OpenRouter credits.
Results are written into `RESULTS.md` (the predictions section is preserved;
the results section is overwritten).
