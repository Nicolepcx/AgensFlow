# Experiment 04 — Sustained-traffic reliability (chunk 7)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-05
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 7 — Mechanism A+C (per-edge failure
tracking + reliability-aware UCB) on top of chunk 6.5 substrate

### Setup

- Same 60-task chunk-6 pool minus C7.1 (recursion-limit edge case) → 59 tasks.
- 8 epochs, shuffled per epoch with deterministic seed.
- Warm-start: graph loaded from chunk 6.5's saved policy_graph.pkl.
- Same hybrid reward, same chunk-6 activation plan (full variant pool).
- λ (reliability_weight) = 0.5 (the default introduced in chunk 6.5).

### Primary prediction (reliability over time)

- **Tokens per RelativeJudge point should drop by ≥10% from epoch 1 → epoch 8.**
  Mechanism: as the policy graph accumulates per-edge data, the UCB-best
  legal action at each confident signature becomes a better choice than the
  rule-based prior.
- **At least 2 (signature, action) edges should accumulate ≥30% failure
  rate** by epoch 8 — and those edges should NOT be the dominant choice in
  later epochs. Mechanism: λ=0.5 reliability penalty downweights them; UCB
  routes around.
- **Per-class variant distribution should narrow.** For at least 4 of the
  8 classes, the top variant's share-of-runs should be higher in epoch 8
  than epoch 1.

### What would falsify the framework's claim

- *Primary fails*: tokens/RelativeJudge curve flat or rising. Online RL is not
  improving the system over sustained traffic.
- *Secondary fails*: no edges accumulate meaningful failure rates, even
  though chunk-6.5 logs showed retries. Failure attribution is wired wrong.
- *Tertiary fails*: variant distribution stays uniform. UCB+reliability
  can't differentiate variants under this reward signal.

### Acknowledged limitations before running

- Single-condition run (λ=0.5). The ablation (λ=0) is deferred to a
  follow-up so we can spend cost on the headline first.
- Same RelativeJudge judge as variant pool family — same-family bias is documented.
- N=8 epochs × 59 tasks = 472 runs. Convergence usually takes more.
- Warm-start means the curves don't show a true cold-start trajectory.
  Cold-start comparison is a follow-up condition.


<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Results — sustained traffic (chunk 7)

Total runs: 472
Final graph: 293 nodes, 3742 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 2 | 11933 | 0.75 | +0.47 | 0.09 |
| 2 | 59 | 2 | 11389 | 0.82 | +0.56 | 0.05 |
| 3 | 59 | 0 | 9056 | 0.84 | +0.62 | 0.00 |
| 4 | 59 | 2 | 8051 | 0.88 | +0.67 | 0.09 |
| 5 | 59 | 2 | 8808 | 0.86 | +0.63 | 0.02 |
| 6 | 59 | 0 | 11410 | 0.82 | +0.55 | 0.14 |
| 7 | 59 | 4 | 9739 | 0.86 | +0.62 | 0.02 |
| 8 | 59 | 2 | 9515 | 0.84 | +0.60 | 0.05 |

**Δ epoch 1 → 8:** tokens +20% (positive = cheaper), RelativeJudge +0.10 (positive = better quality).

### Top unreliable (signature, action) edges in final graph

| signature (regime) | action | visits | failures | failure rate |
|---|---|---:|---:|---:|
| evidence_heavy | `verifier_haiku` | 2 | 2 | 50.00% |
| evidence_heavy | `verifier_haiku` | 4 | 2 | 33.33% |
| ambiguous | `verifier_haiku` | 2 | 1 | 33.33% |
| ambiguous | `verifier_haiku` | 2 | 1 | 33.33% |
| straightforward | `verifier_haiku` | 5 | 2 | 28.57% |
| evidence_heavy | `verifier_haiku` | 5 | 2 | 28.57% |
| straightforward | `verifier_haiku` | 3 | 1 | 25.00% |
| evidence_heavy | `verifier_haiku` | 3 | 1 | 25.00% |
| straightforward | `verifier_haiku` | 3 | 1 | 25.00% |
| ambiguous | `verifier_haiku` | 4 | 1 | 20.00% |

<!-- RESULTS:END -->
