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

Total runs: 8
Final graph: 259 nodes, 4710 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0 | 24628 | 0.50 | +0.20 | 0.00 |
| 2 | 4 | 0 | 27853 | 0.74 | +0.44 | 0.00 |

**Δ epoch 1 → 2:** tokens -13% (positive = cheaper), RelativeJudge +0.24 (positive = better quality).

### Top per-(skill, model) edges in final graph

| signature | action (skill_card × model) | visits | mean reward | reward σ | mean tokens | token σ | failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| evidence_heavy | `planner` | 262 | 0.59 | 0.02 | 3 | 44 | 0% |
| straightforward | `planner` | 220 | 0.73 | 0.05 | 7 | 54 | 0% |
| evidence_heavy | `solver_fast` | 135 | 0.65 | 0.00 | 0 | 0 | 0% |
| straightforward | `solver_fast` | 125 | 0.82 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:web_search_tavily` | 123 | 0.82 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:evaluator` | 121 | 0.71 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:memory` | 113 | 0.81 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:web_search_exa` | 113 | 0.81 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_tavily` | 111 | 0.70 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:solver_haiku` | 109 | 0.70 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:memory` | 101 | 0.70 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_exa` | 101 | 0.69 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `solver_mini` | 92 | 0.69 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:solver_haiku` | 89 | 0.80 | 0.00 | 0 | 0 | 0% |
| straightforward | `evaluator` | 76 | 0.81 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `solver_mini` | 67 | 0.61 | 0.00 | 0 | 0 | 0% |
| ambiguous | `planner` | 52 | 0.45 | 0.06 | 18 | 93 | 0% |
| straightforward | `skip:solver_mini` | 48 | 0.83 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:evaluator` | 46 | 0.82 | 0.00 | 0 | 0 | 0% |
| straightforward | `solver_mini` | 45 | 0.82 | 0.00 | 0 | 0 | 0% |

<!-- RESULTS:END -->
