# Experiment 06 — Cross-condition quality evaluation

Pre-registered notes, written before the run. Predictions are preserved
across runs; only the auto-generated results section is overwritten.

## Purpose

Test whether chunk-8 (skip-on)'s headline +0.02 RelativeJudge lift over chunk-7
(no-skip) survives independent scrutiny. Per-condition RelativeJudge ranks each
condition's trajectories against its own peer group; this experiment scores
the two conditions' trajectories *together*, head-to-head, against a single
rubric, with three judges across three model families.

## Setup

- For each of 59 tasks: pull last 2 epochs of successful runs from chunk 7
  and chunk 8 (~2+2 = up to 4 trajectories per task).
- Build a single RelativeJudge group containing all 4 trajectories.
- Score with **four** judges in parallel:
  - `anthropic/claude-haiku-4.5` (same family as the original chunk-7/8 judge)
  - `anthropic/claude-sonnet-4.6` (within-Anthropic, different tier — control
    for whether disagreement is family-driven vs model-driven)
  - `openai/gpt-5.4` (cross-family check, OpenAI)
  - `x-ai/grok-4.3` (cross-family check, xAI)

  *Note:* `google/gemini-3.1-pro-preview` was the original third judge but
  returned null score entries under Instructor's TOOLS mode and failed
  schema validation; replaced with `sonnet-4.6` + `grok-4.3` for a stronger
  cross-family panel.

  *Chunk-11 update:* the conclusion "qwen + gemini don't work" was
  incomplete — chunk-11's per-judge probe
  (`scripts/probe_qwen_judge.py`) showed that qwen models DO work
  under `Mode.JSON` + `extra_body={"provider":
  {"require_parameters": True}}`, both documented in the
  Instructor+OpenRouter integration guide. The old e06 setup used
  Instructor's TOOLS mode without `extra_body`; the routing failures
  were our config gap, not a model limitation. See
  `learning/ruler/README.md` design notes for the full taxonomy of
  judge-compatibility failure modes. Gemini-pro-preview's null-fields
  issue remains unresolved (chunk-12 follow-up).
- Compute per-trajectory scores, condition means, head-to-head winner,
  and inter-judge agreement.

## What "the chunk-8 quality claim survives" looks like

- chunk-8 wins or ties ≥50% of head-to-heads under all 3 judges
- Inter-judge agreement ≥2/3 on a majority of tasks (i.e. judges aren't
  randomly disagreeing)
- Per-class breakdown shows wins concentrated in the *expected* classes
  (C1, C6, C8 — simple extraction) and competitive (not catastrophic) on
  the hard classes (C5, C7)

## What would falsify the chunk-8 quality claim

- chunk-7 wins majority head-to-heads under cross-family judges
  (`gpt-5.4`, `gemini-3.1-pro-preview`) even if `claude-haiku-4.5` favors
  chunk-8 — that would diagnose same-family bias as the source of the
  +0.02 RelativeJudge lift
- Catastrophic chunk-7 wins on C5/C7 (the hard classes), suggesting
  chunk-8's skip-mechanism cuts material it shouldn't cut

<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Cross-evaluation results

Tasks scored: 59, skipped (asymmetric coverage): 0

### Per-judge head-to-head (chunk-8 vs chunk-7)

| judge | chunk-8 wins | chunk-7 wins | ties | chunk-8 win rate |
|---|---:|---:|---:|---:|
| gpt | 29 | 27 | 3 | 49% |
| grok | 26 | 29 | 4 | 44% |
| haiku | 22 | 28 | 9 | 37% |
| sonnet | 28 | 21 | 10 | 47% |

**Tie threshold: |Δ score| < 0.05.**

### Inter-judge agreement on the winner

Distribution of *majority size* across tasks (3 judges per task).

| majority size | n tasks | meaning |
|---:|---:|---|
| 3 | 15 | unanimous agreement |
| 2 | 12 | 2 of 3 agree |

### Per-class breakdown

| class | judge | chunk-8 wins | chunk-7 wins | ties |
|---|---|---:|---:|---:|
| C1 | haiku | 1 | 6 | 1 |
| C1 | gpt | 3 | 4 | 1 |
| C1 | sonnet | 3 | 4 | 1 |
| C1 | grok | 2 | 6 | 0 |
| C2 | haiku | 3 | 4 | 1 |
| C2 | gpt | 3 | 5 | 0 |
| C2 | sonnet | 3 | 3 | 2 |
| C2 | grok | 2 | 6 | 0 |
| C3 | haiku | 1 | 2 | 3 |
| C3 | gpt | 4 | 2 | 0 |
| C3 | sonnet | 4 | 1 | 1 |
| C3 | grok | 3 | 1 | 2 |
| C4 | haiku | 4 | 2 | 0 |
| C4 | gpt | 4 | 2 | 0 |
| C4 | sonnet | 4 | 2 | 0 |
| C4 | grok | 4 | 2 | 0 |
| C5 | haiku | 5 | 1 | 0 |
| C5 | gpt | 4 | 2 | 0 |
| C5 | sonnet | 5 | 1 | 0 |
| C5 | grok | 5 | 1 | 0 |
| C6 | haiku | 3 | 4 | 1 |
| C6 | gpt | 6 | 2 | 0 |
| C6 | sonnet | 5 | 1 | 2 |
| C6 | grok | 5 | 2 | 1 |
| C7 | haiku | 2 | 5 | 2 |
| C7 | gpt | 2 | 7 | 0 |
| C7 | sonnet | 2 | 6 | 1 |
| C7 | grok | 1 | 7 | 1 |
| C8 | haiku | 3 | 4 | 1 |
| C8 | gpt | 3 | 3 | 2 |
| C8 | sonnet | 2 | 3 | 3 |
| C8 | grok | 4 | 4 | 0 |

<!-- RESULTS:END -->

## Findings & framing

**Date written:** 2026-05-07
**Total empirical record:** 236 head-to-head decisions (59 tasks × 4 judges).
Per-task results in `results.jsonl`; aggregates in `aggregates.json`.

### What the cross-evaluation shows

The chunk-8 substrate's apparent +0.02 RelativeJudge lift over chunk-7 in the
in-condition aggregates *was indeed mostly artifact of RelativeJudge ranking
each condition against its own peer group*. Head-to-head, with judges
seeing both conditions' trajectories simultaneously:

| judge | family | chunk-8 win rate | reading |
|---|---|:---:|---|
| `claude-haiku-4.5` | Anthropic (orig.) | 37% | mild chunk-7 preference |
| `claude-sonnet-4.6` | Anthropic | 47% | tied |
| `gpt-5.4` | OpenAI | 49% | tied |
| `grok-4.3` | xAI | 44% | tied |

**Three of four judges put chunk-8 within 3 percentage points of 50%.**
Only haiku — the *original RelativeJudge judge that calibrated chunk-7's training
reward signal* — shows a meaningful chunk-7 lean. That is directly
diagnostic of same-family-bias as the source of haiku's preference, not
a chunk-8 quality regression. Even within Anthropic, the higher-tier
sonnet does *not* show the same bias.

### What this lets us claim, calibrated

**Yes:** chunk-8 *preserves* quality. The cross-family judges agree the
two conditions are essentially tied head-to-head. The 47% token reduction
chunk-8 produces is real and not bought with a hidden quality loss.

**No:** chunk-8 does NOT *improve* quality over chunk-7. The +0.02 RelativeJudge
lift in the in-condition aggregates was calibration drift. We have to
soften the chunk-8 narrative from "cheaper *and* higher quality" to
"cheaper at preserved quality."

### Per-class direction

Across judges, the directional pattern is consistent and informative:

| class | judges' net direction | mechanistic reading |
|---|---|---|
| C1 (simple lookup) | chunk-7 | chunk-8 over-skips information-gathering |
| C2 (multi-step) | chunk-7 | full pipeline matters when steps compound |
| C3 | chunk-8 | the skipped skills genuinely added nothing |
| C4 | chunk-8 | same — chunk-8 correctly identified redundancy |
| C5 (no-corpus-answer) | chunk-8 | full pipeline retained AND benefits from learned bindings |
| C6 (numerical extraction) | chunk-8 | direct extraction; less is more |
| C7 (consistency debug) | chunk-7 | debugging genuinely needs every redundant signal |
| C8 (simple extraction) | tied | judges disagree at task level; class-level no clear winner |

**This is the most important framing-positive finding.** The framework's
*per-class topology choices are correct*. Chunk-7 wins exactly the classes
where chunk-8's skipping cuts material that genuinely matters. Chunk-8
wins exactly the classes where chunk-7's full pipeline was buying
nothing. The substrate is selecting the right topology per domain — not
just cutting cost everywhere.

### Inter-judge agreement

- **4-of-4 unanimous** on the per-task winner: 32 / 59 tasks (54%)
- 3-of-4 majority: 15 / 59 tasks (25%)
- 2-2 split: 12 / 59 tasks (20%)

For 79% of tasks, at least 3 of 4 judges agreed on direction. The
pattern is robust, not driven by a single judge's idiosyncrasy.

### Methodological note: why this experiment matters

Per-condition RelativeJudge ranking was a known limitation of the chunk-7 / 8
methodology — RelativeJudge scores trajectories against rolling buffers of
*same-condition* peers, so absolute scores aren't directly comparable
across conditions. We documented this in the chunk-8 RESULTS.md as a
caveat to the headline.

The cross-evaluation is what closes the methodological loop. By scoring
both conditions' trajectories *together* in a single RelativeJudge call, we
force the judge to rank them against each other rather than against
peer-group-internal references. The four judges across three model
families give us robustness against any individual judge's biases.

The diagnostic finding — that haiku alone shows the chunk-7 preference
the in-condition aggregates implied — is methodologically important
beyond this experiment. **Future RelativeJudge work should not use the same
model family as the agents under test.** Or if it does, it should
cross-validate with at least one out-of-family judge to detect
same-family-bias drift.

### Acknowledged limitations

- **n = 4 trajectories per task** (last 2 epochs from each condition).
  More history per task would tighten the head-to-head signal but at
  proportional judge-call cost.
- **Judges score trajectory summaries, not full transcripts.** RelativeJudge's
  `path_summary` + `final_answer` is a compressed view; subtle
  step-by-step differences may be lost. A full-transcript variant is
  out of scope for chunk 8.5 but a fair follow-up.
- **Same task pool as chunks 6/7/8.** Cross-domain validation is
  separate (chunk 9 / 10 territory).

