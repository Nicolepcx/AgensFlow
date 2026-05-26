# Experiment 09 — Cross-domain validation (security advisories)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-15
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk-9 substrate (sustained learning +
topology skip-X + (skill × model) variant pool), unchanged from e07. Only
the task distribution + corpus differ.

### Setup

- 60-task pool across the same 8 scenario classes as e03 (C1-C8),
  authored fresh against a synthetic 12-document security-advisory corpus.
- 8 epochs, shuffled per epoch with deterministic seed.
- **Cold-start** by default (warm-start from a distributed-systems graph
  would produce signatures the prior graph never saw — effectively
  cold-start with noise).
- Variant pool identical to e07: 9 (skill × model) solver cells +
  planner + memory + 2 web-search + 2 verifier + evaluator.
- skip-X enabled (matching the main e07/e08 condition).
- Same hybrid reward configuration.

### Ablation arc

1. **Main run** — `--cold-start`, skip-X enabled, 8 epochs.
2. **No-skip control** — `--cold-start --no-skip`, 8 epochs. Tests
   whether forced full-pipeline routing produces materially different
   per-class winners / worse Pareto frontier.
3. **Fixed-full-pipeline baseline** — one pass over 60 tasks with a
   hardcoded topology (all skills active, no learning). Provides the
   comparator for prediction #4.

### The cross-domain claim (4 parts)

**(a) Qualitative pattern transfers.** ≥4 of 8 scenario classes converge
to a (skill, model) combination that is non-trivially different from
"default skill + most-capable model." Same direction as e07's prediction
(a); we are testing that the discovery itself transfers cross-domain,
not just that some signal exists.

**(b) Skip choices emerge.** The substrate's skip-X distribution varies
by scenario class. Specifically: at least 2 classes (likely C1, C6, C8)
converge to skipping verifier; at least 2 classes (likely C3, C5, C7)
converge to keeping it. The per-class skip pattern is the cleanest
signal that topology learning transferred.

**(c) Per-edge reliability differentiation.** At minimum 2 (skill, model)
pairs at confident signatures show retry-rate gaps ≥10 percentage points
despite comparable mean reward — i.e., the systems-level reliability
profile observed on the original corpus is also observable on the
security corpus.

**(d) Cost-quality Pareto preserved.** Main run achieves ≥10% cost
reduction at RelativeJudge within −0.03 of the fixed-full-pipeline baseline's
mean RelativeJudge. Equivalently: tokens/run drops ≥10% with no more than 0.03
loss in mean RelativeJudge score vs the fixed baseline. This is the strict
operational definition of "the mechanism transfers without harming
quality."

### What would falsify

- *(a) fails*: per-class winners on the security corpus are dominated
  by "default skill + most-capable model" — the substrate didn't find
  better combinations on this domain. The cross-domain claim
  collapses to "the substrate doesn't generalize."
- *(b) fails*: skip distribution is uniform across classes (substrate
  either always skips or never skips). Topology learning didn't
  transfer cross-domain.
- *(c) fails*: reliability metrics are noise; no per-edge
  differentiation visible. Either the metric carries no domain-
  general signal or the security corpus produces too much variance
  for detection.
- *(d) fails*: cost reduction <10% OR RelativeJudge drops >0.03 vs the fixed
  baseline. The substrate either doesn't reduce cost on the new
  corpus or it does so by sacrificing quality.

### Acknowledged constraints

- Synthetic security corpus (12 documents) — by design, for containment
  and reproducibility. Real-CVE corpus would introduce judge-model
  contamination from solver pretraining.
- Same variant pool as e07. This experiment tests *cross-domain
  transfer of the substrate's mechanism*, not the impact of a different
  variant pool.
- Same RelativeJudge judge family. Cross-eval methodology (chunk-8.5 / 11) is
  available but optional for this experiment; reported only if
  in-condition RelativeJudge ranks look suspicious.


<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Results — cross-domain validation (security advisories)

Total runs: 480
Final graph: 583 nodes, 18456 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg | skip% | solvers/run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 0 | 10,887 | 0.69 | +0.45 | 0.00 | 45% | 5.2 |
| 2 | 60 | 0 | 12,768 | 0.70 | +0.43 | 0.03 | 40% | 5.4 |
| 3 | 60 | 0 | 10,790 | 0.74 | +0.48 | 0.03 | 46% | 4.9 |
| 4 | 60 | 0 | 12,380 | 0.73 | +0.46 | 0.03 | 44% | 5.1 |
| 5 | 60 | 0 | 11,877 | 0.73 | +0.47 | 0.00 | 43% | 5.1 |
| 6 | 60 | 0 | 13,904 | 0.79 | +0.52 | 0.02 | 40% | 5.2 |
| 7 | 60 | 0 | 13,124 | 0.78 | +0.49 | 0.07 | 40% | 5.1 |
| 8 | 60 | 0 | 13,085 | 0.71 | +0.45 | 0.03 | 40% | 5.2 |

**Δ epoch 1 → 8:** tokens -20% (positive = cheaper), RelativeJudge +0.02 (positive = better quality), skip -5% (positive = more committed to skip), solvers/run +0.0 (negative = pool converging).

### Top per-(skill, model) edges in final graph

| signature | action (skill_card × model) | visits | mean reward | reward σ | mean tokens | token σ | failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| evidence_heavy | `planner` | 731 | 0.51 | 0.19 | 320 | 241 | 0% |
| straightforward | `planner` | 592 | 0.61 | 0.22 | 262 | 201 | 0% |
| straightforward | `skip:web_search_tavily` | 232 | 0.67 | 0.22 | 0 | 0 | 0% |
| evidence_heavy | `skip:evaluator` | 219 | 0.61 | 0.18 | 0 | 0 | 0% |
| straightforward | `skip:web_search_exa` | 212 | 0.67 | 0.22 | 0 | 0 | 0% |
| straightforward | `skip:memory` | 206 | 0.67 | 0.24 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_tavily` | 188 | 0.60 | 0.19 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_exa` | 174 | 0.60 | 0.19 | 0 | 0 | 0% |
| evidence_heavy | `skip:memory` | 171 | 0.60 | 0.20 | 0 | 0 | 0% |
| ambiguous | `planner` | 154 | 0.45 | 0.18 | 309 | 217 | 0% |
| straightforward | `evaluator` | 153 | 0.68 | 0.23 | 399 | 481 | 0% |
| evidence_heavy | `solver_fast` | 135 | 0.65 | 0.00 | 0 | 0 | 0% |
| straightforward | `solver_fast` | 125 | 0.82 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:solver_haiku` | 109 | 0.70 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `solver_mini` | 92 | 0.69 | 0.00 | 0 | 0 | 0% |
| straightforward | `skip:evaluator` | 91 | 0.65 | 0.25 | 0 | 0 | 0% |
| straightforward | `skip:solver_haiku` | 89 | 0.80 | 0.00 | 0 | 0 | 0% |
| evidence_heavy | `skip:memory` | 71 | 0.50 | 0.19 | 0 | 0 | 0% |
| evidence_heavy | `skip:web_search_exa` | 71 | 0.50 | 0.19 | 0 | 0 | 0% |
| evidence_heavy | `solver_concise_mini` | 70 | 0.51 | 0.18 | 692 | 145 | 0% |

<!-- RESULTS:END -->
