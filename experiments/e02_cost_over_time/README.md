# Experiment 02 — Cost-over-time learning trajectory

## Hypothesis

When AgensFlow is run with policy-driven dynamic routing (chunk 4.5) and a
shared persistent policy graph, **tokens-per-successful-task descends as a
function of run number** as the graph accumulates value estimates and the
router begins overriding the rule-based prior at confident signatures.

This is the experiment that directly tests the framework's distinguishing
claim — *the orchestration policy improves from its own traces, reducing
unproductive coordination overhead over many runs*. Experiment 01 tested
the rule-based prior; this one tests the learning loop.

## Design

### Benchmark

4 Category B tasks from experiment 01 (the evidence-grounded ones —
TCP/UDP, battery chemistry, oil crisis, SQL/NoSQL). Category B is the
sweet spot for chunk 5: tasks where the policy has real routing decisions
to optimize and where the success/failure judgement is unambiguous.

Each task is run **15 times sequentially**, threading a shared
`PolicyGraph` instance through every run. After each run the graph is
saved to disk so the experiment can resume if interrupted.

### Configuration

- Single configuration: `agensflow_dynamic` (the chunk 4.5 path with
  `policy_graph` provided).
- Models: same as experiment 01 (gpt-5.4-nano for planner/memory/evaluator,
  claude-haiku-4.5 for solver/verifier).
- `confidence_threshold = 3`: minimum visits to a signature before the
  graph's recommendation overrides the rule-based prior.
- `max_steps = 12`: budget cap on routing decisions per run.

### Metrics captured per (task, run_index) cell

- `total_tokens` — including failed validation retries (honest cost).
- `n_calls` — total agent invocations including the router node firings.
- `validation_retries` — failed Instructor validations.
- `latency_seconds`.
- `regime_used`.
- `done` — evaluator's done flag.
- `judgement` — grader's verdict (success/partial/failure).
- `reward` — the policy graph's reward signal.
- `path` — the (signature, action) sequence taken.
- `routing_sources` — for each routing decision: rule_based_prior,
  graph_recommendation, or termination reason.
- `policy_graph_size` — node count after this run.
- `n_confident_nodes` — nodes with ≥3 visits.

### Falsification criteria (pre-registered)

Reject the cost-over-time claim if:

1. **Primary**: mean tokens-per-successful-task across runs 11-15
   (averaged across the 4 tasks) is **not at least 15% lower** than the
   mean across runs 1-5. This is the headline test.
2. **Secondary**: the router never overrides the rule-based prior with a
   graph recommendation across the 60 total runs. This would mean either
   confidence_threshold is too high or signature folding is not producing
   enough reuse.
3. **Tertiary**: success rate in the last third of runs is materially lower
   than the first third (< -10%). This would mean the policy is
   *degrading* the system, not improving it.

Any of these failing is reported verbatim. The experiment commits to its
predictions before the run.

## How to run

From the repo root, with the package installed in editable mode and
`OPENROUTER_API_KEY` in `.env`:

```
# Smoke version (5 runs × 1 task = 5 cells, ~$1)
python -m experiments.e02_cost_over_time.run --tasks B1_tcp_udp --runs 5

# Full benchmark (15 runs × 4 tasks = 60 cells, ~$3-5)
python -m experiments.e02_cost_over_time.run

# Resume from a saved policy graph
python -m experiments.e02_cost_over_time.run --resume

# Reset and start fresh
python -m experiments.e02_cost_over_time.run --reset
```

The graph persists to `experiments/e02_cost_over_time/policy_graph.pkl`.
Trace dumps to `results_raw.jsonl`. Markdown summary to `RESULTS.md`.

## Limitations acknowledged before running

- Single trial per (task, run_index) cell. Each run is independent of
  noise variance; the curve is a single trajectory rather than an
  averaged trajectory across multiple seeds.
- Same models as experiment 01 — same-family grader bias still present.
- Sequential execution: runs happen in order, so any model-side drift
  during the experiment window is conflated with policy learning. Mitigation:
  the experiment runs in a single session, typically under 30 minutes.
- Category B only. Category A (no documents) is too easy to need a
  policy graph; Category C (missing evidence) showed in experiment 01
  that frontier models already refuse to confabulate, masking the
  framework's value-add. B is where the routing optimization has room.
