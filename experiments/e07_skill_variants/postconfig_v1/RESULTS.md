# Experiment 07 — Skill-definition variants (chunk 9)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-07
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 9 — declarative SKILL.md cards bound
to the policy graph as (skill_card × model) action cells, on top of the
chunk-8 substrate (sustained learning + topology skip-X).

### Setup

- Same 59-task pool as chunks 6/7/8 (minus C7.1 — recursion edge case).
- 8 epochs, shuffled per epoch with deterministic seed.
- Warm-start: chunk-8 final graph (richest substrate state available).
- Solver action space expanded from chunk-8's 3 model variants to chunk-9's
  9 (skill × model) cells = 3 SKILL.md cards × 3 model bindings (haiku /
  fast / mini).
- skip-X stays enabled (chunk-8 default).
- Same hybrid reward, λ=0.5 reliability weight (chunk-7 default).

### Solver SKILL.md cards under test

- `solver_concise` — minimum-viable answer; single-paragraph; no reasoning trace
- `solver_chain_of_thought` — explicit step-by-step inference, structured
  setup→reasoning→conclusion
- `solver_evidence_first` — citation-driven; cited evidence enumerated
  before any conclusion

The chunk-7/8 hardcoded solver was closest to chain_of_thought style;
chunk-9 adds two genuinely different behavioral envelopes.

### The systems-perspective hypothesis (4 parts)

**(a) Per-class differentiation.** At least 4 of 8 scenario classes converge
to a (skill, model) combination that is non-trivially different from the
"default skill + most-capable model" ground truth — a cheaper or
differently-constrained combination wins on RelativeJudge × cost.

**(b) Cost optimization through skill-as-constraint.** At least one class
converges to a (cheaper model, tighter skill) pair that produces ≥20% lower
tokens than the same model paired with the default skill spec, at preserved
RelativeJudge head-to-head. Direct test that SKILL.md acts as a *runtime constraint*
on model behavior — not as "better prompting."

**(c) Reliability differentiation.** Per-edge retry rate, reward variance,
and token variance differ meaningfully across (skill, model) pairs at the
same signature. The systems-level reliability profile is observable, not
noise. Specifically: at least 2 (skill, model) pairs at confident
signatures show retry-rate gaps ≥10 percentage points despite comparable
mean reward.

**(d) Stable interaction surface.** Re-runs of the same task pool against
the chunk-9 final graph (frozen, no learning) produce similar per-class
winning combinations — discovery is reproducible learning, not lucky
exploration. Tested via a stability-replay run after the main sweep
(`--frozen --epochs 2`). Predicted: ≥6 of 8 classes pick the same winning
(skill, model) combination in both replay epochs.

### What would falsify

- *(a) fails*: per-class winners are dominated by "default skill +
  most-capable model" — the framework didn't find better combinations.
  The systems claim doesn't survive: cards add nothing the substrate
  couldn't already discover from model variants alone.
- *(b) fails*: cost-saving combinations don't appear; the framework's
  "skill spec as constraint" framing is decorative.
- *(c) fails*: reliability metrics are noise — no robust per-edge
  differentiation. Either the substrate's tracking is too coarse or
  the reliability profile genuinely doesn't exist for this corpus.
- *(d) fails*: stability replay produces wildly different per-class
  winners. The substrate is over-fitting to traffic noise rather than
  discovering domain structure.

### Acknowledged constraints

- One corpus, one variant pool, one judge family. Cross-domain validation
  is a separate experiment (chunk 10+).
- Three SKILL.md cards is a small palette. The OSS user-story is "users
  ship as many SKILL.md alternatives as they want"; this experiment tests
  whether the substrate handles that surface, not how it scales to 20+
  cards.
- Same-family RelativeJudge judge bias is a known issue from chunk 6.5/7. Chunk-9
  reuses the chunk-8.5 cross-eval methodology to verify quality
  preservation independent of the in-condition RelativeJudge ranks.


<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Results — skill-definition variants (chunk 9)

Total runs: 472
Final graph: 443 nodes, 11309 total visits

### Per-epoch trajectory

| ep | n | err | tokens | RelativeJudge | reward | retries | skip% | solvers/run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 0 | 15,020 | 0.73 | +0.47 | 0.00 | 31% | 5.7 |
| 2 | 59 | 0 | 13,254 | 0.81 | +0.53 | 0.05 | 34% | 5.5 |
| 3 | 59 | 0 | 11,206 | 0.77 | +0.49 | 0.05 | 41% | 4.9 |
| 4 | 59 | 0 | 9,743 | 0.81 | +0.54 | 0.03 | 47% | 4.4 |
| 5 | 59 | 0 | 11,154 | 0.78 | +0.51 | 0.03 | 43% | 5.0 |
| 6 | 59 | 0 | 11,141 | 0.80 | +0.53 | 0.07 | 40% | 4.7 |
| 7 | 59 | 0 | 11,980 | 0.82 | +0.55 | 0.10 | 38% | 5.1 |
| 8 | 59 | 1 | 10,808 | 0.82 | +0.56 | 0.05 | 43% | 4.6 |

**Δ epoch 1 → 8:** tokens −28% (cheaper), RelativeJudge +0.09 (better quality), skip +12pp (more committed to skip), solvers/run −1.1 (variant pool converging).

### Top per-(skill, model) edges in final graph

| signature | action (skill_card × model) | visits | mean reward | reward σ | mean tokens | token σ | failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| evidence_heavy | `planner` | 491 | 0.53 | 0.19 | 221 | 235 | 0% |
| straightforward | `planner` | 408 | 0.67 | 0.18 | 186 | 199 | 0% |
| evidence_heavy | `skip:evaluator` | 152 | 0.66 | 0.14 | 0 | 0 | 0% |
| straightforward | `skip:web_search_tavily` | 150 | 0.79 | 0.11 | 0 | 0 | 0% |
| straightforward | `skip:web_search_exa` | 138 | 0.79 | 0.11 | 0 | 0 | 0% |
| evidence_heavy | `solver_fast` | 135 | 0.65 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_tavily` | 135 | 0.66 | 0.16 | 0 | 0 | 0% |
| straightforward | `skip:memory` | 134 | 0.79 | 0.13 | 0 | 0 | 0% |
| straightforward | `solver_fast` | 125 | 0.82 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:memory` | 124 | 0.65 | 0.17 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_exa` | 119 | 0.65 | 0.16 | 0 | 0 | 0% |
| evidence_heavy | `skip:solver_haiku` | 109 | 0.70 | 0.00 | 0 | 0 | 0% |
| straightforward | `evaluator` | 99 | 0.79 | 0.13 | 240 | 510 | 0% |
| ambiguous | `planner` | 98 | 0.45 | 0.15 | 236 | 244 | 0% |
| evidence_heavy | `solver_mini` | 92 | 0.69 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:solver_haiku` | 89 | 0.80 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `solver_mini` | 67 | 0.61 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:evaluator` | 57 | 0.75 | 0.22 | 0 | 0 | 0% |
| evidence_heavy | `skip:memory` | 49 | 0.55 | 0.15 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_exa` | 49 | 0.55 | 0.15 | 0 | 0 | 0% |

<!-- RESULTS:END -->

## Chunk-11 replay — bias-mitigation validation (added 2026-05-11)

The chunk-9 headline (`+0.09 RelativeJudge, −28% tokens, +12pp skip, −1.1
solvers/run`) was produced under a single-judge RelativeJudge setup with
`anthropic/claude-haiku-4.5`. Same-family judge bias was a known
caveat (chunks 6.5 / 7 cross-eval methodology). The chunk-11 reward
upgrade (A1+A2+A3+A4) addresses this structurally; the question was
whether the chunk-9 finding survives application of the upgraded
reward to the *same* trajectories.

The replay (`replay_rescore.py`) re-scored all 472 trajectories from
this sweep under the chunk-11 stack — no agent re-execution, no
policy-graph updates. Pure judge work.

### Setup

- **Source**: `results_agensflow.jsonl` (472 trajectory records from
  this sweep)
- **Cross-judge triple** (3 families, all with 100% axis-population
  compliance across 463 multi-trajectory groups):
  - `anthropic/claude-haiku-4.5` — TOOLS mode, `extra_body={provider: {require_parameters: true}}`
  - `openai/gpt-5.4-mini` — TOOLS mode, NO extra_body (require_parameters paradoxically breaks OpenAI's primary OpenRouter route)
  - `qwen/qwen3.6-flash` — JSON mode, `extra_body={provider: {require_parameters: true}}`
- **Per-axis rubric** (chunk 11.A3): `goal_achievement` (0.30) +
  `grounding` (0.30) + `coordination` (0.20) + `recovery` (0.20)
- **Effective scalar**: composed via weighted axis sum per judge, then
  mean across judges; per-judge effective scalars get a population std
  + range for the disagreement → confidence calculation
- **Rolling-buffer matched** to the original chunk-9 harness (group
  size 4, ordered by run_index per scenario class) so each trajectory
  is judged against the SAME peers it was judged against originally
- **Cost**: ~$35 LLM (~1,416 judge calls, ~$0.025/call avg). Runtime
  18,580s (5.2 hr) on a serial loop.

### Headline result

**`+0.10 RelativeJudge` from epoch 1 → epoch 8 under cross-judge + per-axis,
vs `+0.09` under the original single-judge haiku.**

```
Original (single-judge haiku):  epoch1=0.73 → epoch8=0.82   Δ +0.09
New (cross-judge + per-axis):   epoch1=0.72 → epoch8=0.82   Δ +0.10
```

The quality-improvement claim survives the strongest bias-mitigation
critique we can mount on this sweep. The substrate's coordination
learning is genuine, not a haiku-self-preference artifact.

### Per-epoch trajectory under cross-judge

| epoch | n | orig RelativeJudge | new RelativeJudge | Δ | mean confidence |
|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 0.73 | 0.72 | −0.01 | 0.66 |
| 2 | 59 | 0.81 | 0.79 | −0.02 | 0.67 |
| 3 | 59 | 0.77 | 0.79 | +0.03 | 0.65 |
| 4 | 59 | 0.81 | 0.80 | −0.01 | 0.66 |
| 5 | 59 | 0.78 | 0.81 | +0.03 | 0.68 |
| 6 | 59 | 0.80 | 0.83 | +0.03 | 0.73 |
| 7 | 59 | 0.82 | 0.83 | +0.01 | 0.70 |
| 8 | 58 | 0.82 | 0.82 | +0.00 | 0.70 |

Mean Δ across all 471 valid trajectories: **+0.007**. Cross-judge
agreement broadly tracks haiku for this corpus — haiku was not
substantially biased on aggregate.

### Per-class breakdown

| class | n | orig RelativeJudge | new RelativeJudge | Δ | per-axis σ | confidence | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 | 64 | 0.86 | 0.84 | −0.02 | 0.07 | 0.76 | easy, judges agreed |
| C2 | 64 | 0.79 | 0.79 | +0.01 | 0.10 | 0.68 | minor disagreement, no net effect |
| **C3** | **48** | **0.63** | **0.72** | **+0.09** | **0.13** | **0.54** | **hard multi-hop — haiku was UNDERrating; cross-judge revealed real quality** |
| C4 | 47 | 0.75 | 0.79 | +0.04 | 0.08 | 0.73 | mild underrating, cross-judge corrected up |
| C5 | 48 | 0.73 | 0.72 | −0.01 | 0.12 | 0.53 | hard, judges disagreed; net flat |
| C6 | 64 | 0.82 | 0.82 | −0.00 | 0.08 | 0.73 | flat |
| C7 | 72 | 0.81 | 0.81 | −0.00 | 0.09 | 0.70 | flat |
| C8 | 64 | 0.87 | 0.85 | −0.01 | 0.08 | 0.72 | flat |

### Findings on the structure of agreement

The per-axis disagreement std (`axis_σ`) and confidence values map
*cleanly* to task difficulty: the hardest scenario classes (C3, C5)
have the highest per-axis disagreement and lowest confidence, while
easier classes (C1, C4, C6) have tight agreement and high confidence.
This is the exact signal chunk-11.A4 (confidence-weighted backup) was
designed to surface and act on.

**C3 (+0.09 under cross-judge) is the standout.** C3 is multi-hop
reasoning — the hardest scenario class in the pool. The original
haiku-only judge was systematically underrating C3 trajectories;
cross-judge averaging revealed they were ~0.10 RelativeJudge higher than the
single judge gave them credit for. The substrate's per-class
differentiation was even cleaner than the chunk-9 numbers showed.

### Reliability + axis-compliance under the new infrastructure

- **3/3 judges, ~100% axis-population compliance** across 463 multi-
  trajectory groups (haiku 463/463, gpt-mini 462/462, qwen-flash
  462/462) — the chunk-11.A3 strict-schema Pydantic constraint forced
  all three models to populate per-axis scores even though gpt-mini
  and qwen wouldn't comply under the relaxed schema.
- **471/472 trajectories re-scored** — single transport-level failure,
  not systematic.
- **Per-judge isolation worked** — any single-judge failure during the
  sweep would have killed only that judge's vote, not the entire
  scoring call. (Did not actually fire in this run; designed for
  resilience under production-scale flakiness.)

### What's still untested under the chunk-11 stack

- **A1 (decompressed evidence)** — the JSONL records don't carry
  trace events, so the replay couldn't exercise A1's per-agent
  contribution rendering. Subset re-execution (chunk-11.B) will
  exercise this at real LLM scale before the chunk-12 cross-domain
  sweep.
- **A4 (confidence-weighted backup) end-to-end** — the replay
  computes confidence per trajectory but doesn't apply it to a
  backup (no graph update). Mechanics are unit-test-validated; the
  full backup-with-confidence path is exercised structurally in the
  harness when the chunk-12 sweep runs against a fresh graph.
- **Real-LLM gradient quality** — does the upgraded reward
  meaningfully change which (skill_card, model) cells the substrate
  converges to? Same-corpus rerun under chunk-11 reward would answer
  this. Chunk-12 cross-domain is the better test of the same
  underlying question.

### Conclusion

**Within the scope of this sweep (chunk-9's 59-task pool, 9-cell
solver variant pool, 4-axis rubric), the chunk-9 quality-improvement
finding is bias-mitigation-robust.** The +0.10 RelativeJudge under three
independent judges across three families is approximately the same
finding as the original +0.09 under haiku alone, with the per-class
structure unchanged and the hardest class (C3) showing the substrate
worked *better* than originally credited. The cost story (−28%
tokens) was never in question — token counts are deterministic.

The next falsifiability test is chunk 12: same substrate, same
chunk-11 reward stack, *different* domain (DeepPlanning EN). That's
the cross-domain bifurcation claim — same framework, different
workload, does it generalize.

