# Experiment 05 — Topology skip-X learning (chunk 8)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-06
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 8 — inline `skip:X` action mechanism
on top of chunk 7's validated substrate (UCB-on-folded-signatures policy
graph). Builds on the chunk 7 + ablation finding that the substrate itself
delivers the cost-quality improvement.

### What's new vs chunk 7

The router's action space now expands at every step to include `skip:X`
for every legal X. This makes coalition membership a learnable
coordination decision: the policy can choose to *exclude* a skill from
the topology rather than only re-ordering / re-binding it. Same substrate,
richer action space.

### Setup

- Same 59-task pool, same shuffle seed (deterministic A/B against chunk 7).
- 8 epochs, warm-start from chunk-6.5 graph (same as chunk 7).
- Same hybrid reward, same chunk-6 activation plan (full variant pool).
- λ (reliability_weight) = 0.5 (chunk-7 default; the ablation showed this
  doesn't materially affect outcome but we keep it for consistency).
- `enable_skip = True` (the experimental knob).

### Primary prediction (selective skip-learning)

- **In ≥2 of 8 scenario classes, `skip:X` becomes the dominant action at
  some signatures within that class** by epoch 8. Most likely candidates
  per chunk-6 expectations: skip-verifier in C1, C6, C8; skip-web-search
  in C1–C4, C6, C8 where the corpus already has the answer.

### Secondary prediction (cost reduction beyond chunk 7)

- **Mean tokens per task in epoch 8 drops ≥10% compared to chunk-7
  epoch 8** (~9,010 → ~8,100 or below). Mechanism: skipped skills cost
  zero tokens.

### Tertiary prediction (no quality regression)

- **RelativeJudge mean stays within 0.05 of chunk-7 epoch 8 baseline** (~0.84).
  UCB should route around the "skip everything → produce nothing →
  terminal low reward" pathology after a few exploration runs; the
  reward signal will catch quality degradation from over-skipping.

### What would falsify

- *(1) fails*: skip actions never become dominant. Either signature
  folding is too coarse to distinguish "X helps here" from "X doesn't",
  or exploration constant is too low to ever try skips enough times to
  learn their value.
- *(2) fails*: mean tokens unchanged from chunk-7 baseline. Either skips
  aren't being explored, or the policy correctly identifies that all
  currently-included skills are necessary at all signatures.
- *(3) fails*: tokens drop but RelativeJudge drops with them (cheap-but-bad).
  Reward signal isn't catching quality degradation from over-skipping —
  the chunk-7 hybrid reward's RelativeJudge weight may need re-tuning.

### Acknowledged limitations before running

- Single-condition run (skip enabled). A no-skip ablation isn't needed
  here because chunk 7's main run already serves as that baseline (same
  seed, same warm-start, same everything except `enable_skip`).
- Skip events count toward `max_steps` (per chunk-8 design decision),
  so a runaway skip-skip loop terminates cleanly via budget exhaustion.
- The activation plan still defines the *available* skills — skip can
  only refuse skills that were already legal. Plan-level coalition
  selection (the planner-level UCB sketched for chunk 9) is a separate
  layer, not in scope here.
- Same constraints as chunk 7: one corpus, one variant pool, one judge,
  no external baseline.


<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Results — topology skip-X (chunk 8)

Total runs: 8
Final graph: 126 nodes, 609 total visits

### Per-epoch trajectory

| epoch | n | errors | tokens/run | RelativeJudge avg | reward avg | retries avg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0 | 7089 | 0.50 | +0.34 | 0.00 |
| 2 | 4 | 0 | 6883 | 0.85 | +0.68 | 0.00 |

**Δ epoch 1 → 2:** tokens +3% (positive = cheaper), RelativeJudge +0.35 (positive = better quality).

### Top unreliable (signature, action) edges in final graph

| signature (regime) | action | visits | failures | failure rate |
|---|---|---:|---:|---:|
| (no failures recorded yet) |  |  |  |  |

<!-- RESULTS:END -->
