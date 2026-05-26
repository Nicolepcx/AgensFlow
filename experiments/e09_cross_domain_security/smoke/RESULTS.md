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

Total runs: 4
Final graph: 48 nodes, 64 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg | skip% | solvers/run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0 | 35,798 | 0.50 | +0.16 | 0.25 | 0% | 9.0 |

### Top per-(skill, model) edges in final graph

| signature | action (skill_card × model) | visits | mean reward | reward σ | mean tokens | token σ | failure rate |
|---|---|---:|---:|---:|---:|---:|---:|
| (no edges with ≥3 visits yet) |  |  |  |  |  |  |  |

<!-- RESULTS:END -->
