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
Final graph: 482 nodes, 6805 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg | skip% | solvers/run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 0 | 31,500 | 0.67 | +0.37 | 0.03 | 0% | 8.7 |
| 2 | 60 | 0 | 27,278 | 0.71 | +0.39 | 0.10 | 0% | 8.9 |
| 3 | 60 | 0 | 25,457 | 0.68 | +0.37 | 0.08 | 0% | 8.4 |
| 4 | 60 | 0 | 24,104 | 0.65 | +0.35 | 0.00 | 0% | 8.2 |
| 5 | 60 | 0 | 25,348 | 0.66 | +0.35 | 0.12 | 0% | 8.3 |
| 6 | 60 | 0 | 24,374 | 0.68 | +0.38 | 0.05 | 0% | 7.9 |
| 7 | 60 | 10 | 25,296 | 0.67 | +0.36 | 0.12 | 0% | 8.3 |
| 8 | 60 | 14 | 26,168 | 0.62 | +0.31 | 0.13 | 0% | 8.7 |

**Δ epoch 1 → 8:** tokens +17% (positive = cheaper), RelativeJudge -0.05 (positive = better quality), skip +0% (positive = more committed to skip), solvers/run -0.0 (negative = pool converging).

### Top per-(skill, model) edges in final graph

| signature | action (skill_card × model) | visits | mean reward | reward σ | mean tokens | token σ | failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| evidence_heavy | `planner` | 229 | 0.35 | 0.23 | 524 | 46 | 0% |
| straightforward | `planner` | 174 | 0.34 | 0.24 | 430 | 29 | 0% |
| ambiguous | `planner` | 53 | 0.39 | 0.21 | 438 | 29 | 0% |
| evidence_heavy | `verifier_haiku` | 46 | 0.27 | 0.18 | 4128 | 1787 | 2% |
| evidence_heavy | `web_search_tavily` | 35 | 0.44 | 0.20 | 500 | 0 | 3% |
| evidence_heavy | `solver_concise_haiku` | 34 | 0.41 | 0.26 | 1402 | 147 | 0% |
| evidence_heavy | `verifier_fast` | 33 | 0.29 | 0.17 | 2375 | 554 | 0% |
| evidence_heavy | `solver_concise_mini` | 32 | 0.41 | 0.19 | 666 | 110 | 0% |
| evidence_heavy | `evaluator` | 29 | 0.52 | 0.16 | 1798 | 622 | 0% |
| evidence_heavy | `memory` | 28 | 0.43 | 0.23 | 1390 | 198 | 0% |
| evidence_heavy | `verifier_fast` | 28 | 0.27 | 0.21 | 2459 | 818 | 0% |
| evidence_heavy | `evaluator` | 25 | 0.34 | 0.22 | 1476 | 565 | 0% |
| straightforward | `evaluator` | 23 | 0.31 | 0.26 | 685 | 227 | 0% |
| evidence_heavy | `evaluator` | 22 | 0.20 | 0.14 | 2231 | 917 | 0% |
| straightforward | `memory` | 21 | 0.38 | 0.25 | 810 | 65 | 0% |
| evidence_heavy | `solver_cot_fast` | 21 | 0.26 | 0.14 | 3970 | 1279 | 0% |
| evidence_heavy | `memory` | 20 | 0.35 | 0.25 | 1334 | 234 | 0% |
| evidence_heavy | `solver_cot_fast` | 20 | 0.35 | 0.25 | 1581 | 666 | 0% |
| straightforward | `verifier_haiku` | 20 | 0.33 | 0.20 | 3003 | 767 | 0% |
| evidence_heavy | `solver_cot_fast` | 20 | 0.43 | 0.22 | 3419 | 831 | 0% |

<!-- RESULTS:END -->
