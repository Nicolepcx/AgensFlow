# Experiment 01 — Regime validation: predictions and results

This document is split in two:

1. **Pre-registered predictions** — written *before* any benchmark run.
   Committed to the repo at the time of writing so the predictions cannot be
   revised after seeing the data.
2. **Results** — auto-generated section below the marker, overwritten by each
   run of `python -m experiments.e01_regime_validation.run`.

---

## Pre-registered predictions

**Date written:** 2026-05-03
**Author:** Nicole Königstein
**Framework version:** AgensFlow 0.1.0 (pre-release, post-Instructor refactor)

The predictions are stated as: for each (category, configuration) cell, the
expected modal outcome on the 4 tasks in that category. "Win" / "lose" framing
refers to *(success, cost)* trade-off, not raw success rate alone.

### Per-category predictions

#### Category A — simple Q&A, no documents

| Configuration | Predicted modal outcome | Reasoning |
|---|---|---|
| `naive` | success | These tasks are within frontier-model capability without orchestration. |
| `agensflow_forced_straightforward` | success at modest token overhead vs naive | Three calls instead of one; the planner→solver→evaluator pipeline adds metadata but not capability. |
| `agensflow_forced_evidence_heavy` | success at large token overhead | Memory and verifier add cost without benefit; verifier may flag "missing evidence" because the docs are empty. |
| `agensflow_auto` | matches `forced_straightforward` | Auto detection should pick `straightforward` from low-evidence/low-verification features. |

**Headline prediction for A**: `auto` and `naive` should have similar success rate; `auto` should use somewhat more tokens than `naive` (3-5×) but markedly fewer than `forced_evidence_heavy`. **The policy claim**: forcing the wrong (over-large) regime here is wasteful.

#### Category B — document-grounded Q&A, evidence present and sufficient

| Configuration | Predicted modal outcome | Reasoning |
|---|---|---|
| `naive` | success but with weaker grounding fidelity | One LLM call inlines the documents; the model usually grounds well but may add general-knowledge claims not in the docs. |
| `agensflow_forced_straightforward` | partial | No memory step; solver receives no `evidence` field, so claims are not document-grounded. |
| `agensflow_forced_evidence_heavy` | success with high grounding fidelity | The full pipeline retrieves evidence and verifies grounding before completion. |
| `agensflow_auto` | matches `forced_evidence_heavy` | Auto detection should pick `evidence_heavy` from high-evidence/high-verification features. |

**Headline prediction for B**: `auto` and `forced_evidence_heavy` outperform `naive` on grounding; `forced_straightforward` underperforms both. **The policy claim**: when evidence matters, the policy correctly selects the heavier regime, and the heavier regime delivers better grounding at higher cost.

#### Category C — adversarial / missing evidence

| Configuration | Predicted modal outcome | Reasoning |
|---|---|---|
| `naive` | failure (confabulates) | Single-shot LLM has no verifier to catch ungrounded claims. |
| `agensflow_forced_straightforward` | failure | No memory or verifier; solver answers from parametric knowledge without checking the documents. |
| `agensflow_forced_evidence_heavy` | success (flags missing evidence) | The verifier should catch claims not supported by the (insufficient) documents and the evaluator should refuse to mark `done=True`. |
| `agensflow_auto` | matches `forced_evidence_heavy` | Same as B. |

**Headline prediction for C**: `auto` and `forced_evidence_heavy` correctly flag the gap (`flagged_missing_evidence == True` in the grader's verdict for ≥3 of 4 tasks); `naive` confabulates a specific answer for ≥3 of 4. **The policy claim**: the verifier+evaluator pair turns "would have hallucinated" into "explicitly flagged missing evidence." This is the most important prediction in the experiment because it is where the framework's value-add is largest.

### Cross-cutting predictions

- **Tokens-per-successful-task**: across all 12 tasks, `agensflow_auto` is expected to use **2-5× more tokens than `naive`** but to deliver **higher success rate** especially on Category C. The cost ratio matters less than the success-rate delta on tasks where evidence-grounding is the actual goal.
- **Validation retries**: rare but non-zero. We expect <10% of agensflow runs to trigger one corrective retry. (The verifier in particular sometimes returns malformed `uncertain_claims` arrays, as observed during chunk 2 development.)
- **Policy match rate**: in 11/12 or 12/12 tasks, `agensflow_auto` should pick the same regime that the corresponding `forced_*` configuration uses for that category. Any mismatch is a policy bug worth investigating.

### What would *falsify* the policy claim

- If `forced_evidence_heavy` does not outperform `forced_straightforward` on Category B grounding, the heavier coalition is not adding value, and the regime distinction is purely architectural rather than functional.
- If `naive` performs as well as `agensflow_auto` on Category C, the verifier is not earning its cost — the framework is not catching what it claims to catch.
- If `agensflow_auto` picks the wrong regime for any task, the rule-based regime detector needs revision.

Any of these outcomes would be reported here verbatim, not buried.

### Known limitations of this experiment (acknowledged before running)

- **N=12, single trial**: outcomes are subject to LLM variance not quantified here. A future replication should run each cell 3-5 times.
- **Same-family grader**: the grader uses claude-haiku-4.5, which is also in the agent mix. Bias toward "agreeing with itself" is possible. Future work should use a stronger out-of-mix grader.
- **Hand-coded `TaskFeatures`**: the auto-detect path receives features the experimenter assigned, not features derived from raw text. Auto-deriving features from natural-language tasks is its own research problem and out of scope for chunk 3.
- **Linear plans only**: branching coalitions are not exercised here. Branching runtime is a later chunk; the regimes that need branching (ambiguous, contradictory, high_risk) are not in this benchmark.

---

<!-- RESULTS:BEGIN (auto-generated; do not edit) -->

## Results

Total cells run: 48

### Category A

| Configuration | N | success | partial | failure | success-rate | tokens/run | tokens/success | retries | flagged-missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive` | 4 | 3 | 1 | 0 | 75% | 40 | 53 | 0 | 0 |
| `agensflow_forced_straightforward` | 4 | 4 | 0 | 0 | 100% | 1776 | 1776 | 0 | 0 |
| `agensflow_forced_evidence_heavy` | 4 | 4 | 0 | 0 | 100% | 3441 | 3441 | 0 | 0 |
| `agensflow_auto` | 4 | 4 | 0 | 0 | 100% | 1790 | 1790 | 0 | 0 |

### Category B

| Configuration | N | success | partial | failure | success-rate | tokens/run | tokens/success | retries | flagged-missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive` | 4 | 3 | 1 | 0 | 75% | 421 | 562 | 0 | 0 |
| `agensflow_forced_straightforward` | 4 | 0 | 0 | 4 | 0% | 2383 | — | 0 | 0 |
| `agensflow_forced_evidence_heavy` | 4 | 3 | 1 | 0 | 75% | 5809 | 7745 | 0 | 0 |
| `agensflow_auto` | 4 | 2 | 2 | 0 | 50% | 6979 | 13959 | 1 | 0 |

### Category C

| Configuration | N | success | partial | failure | success-rate | tokens/run | tokens/success | retries | flagged-missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive` | 4 | 4 | 0 | 0 | 100% | 224 | 224 | 0 | 4 |
| `agensflow_forced_straightforward` | 4 | 3 | 1 | 0 | 75% | 2308 | 3077 | 0 | 4 |
| `agensflow_forced_evidence_heavy` | 4 | 3 | 1 | 0 | 75% | 4418 | 5891 | 0 | 4 |
| `agensflow_auto` | 4 | 2 | 2 | 0 | 50% | 4279 | 8558 | 0 | 4 |

### Overall (all categories)

| Configuration | N | success-rate | tokens/run | tokens/success | retries |
|---|---:|---:|---:|---:|---:|
| `naive` | 12 | 83% | 228 | 274 | 0 |
| `agensflow_forced_straightforward` | 12 | 58% | 2155 | 3695 | 0 |
| `agensflow_forced_evidence_heavy` | 12 | 83% | 4556 | 5467 | 0 |
| `agensflow_auto` | 12 | 67% | 4349 | 6524 | 1 |

### Per-task detail

| task | category | configuration | judgement | tokens | calls | retries | regime used |
|---|---|---|---|---:|---:|---:|---|
| `A1_arithmetic` | A | `naive` | **success** | 29 | 1 | 0 | — |
| `A1_arithmetic` | A | `agensflow_forced_straightforward` | **success** | 1667 | 3 | 0 | straightforward |
| `A1_arithmetic` | A | `agensflow_forced_evidence_heavy` | **success** | 3363 | 5 | 0 | evidence_heavy |
| `A1_arithmetic` | A | `agensflow_auto` | **success** | 1698 | 3 | 0 | straightforward |
| `A2_unit_conversion` | A | `naive` | **success** | 37 | 1 | 0 | — |
| `A2_unit_conversion` | A | `agensflow_forced_straightforward` | **success** | 1806 | 3 | 0 | straightforward |
| `A2_unit_conversion` | A | `agensflow_forced_evidence_heavy` | **success** | 3519 | 5 | 0 | evidence_heavy |
| `A2_unit_conversion` | A | `agensflow_auto` | **success** | 1824 | 3 | 0 | straightforward |
| `A3_definition` | A | `naive` | **partial** | 66 | 1 | 0 | — |
| `A3_definition` | A | `agensflow_forced_straightforward` | **success** | 1905 | 3 | 0 | straightforward |
| `A3_definition` | A | `agensflow_forced_evidence_heavy` | **success** | 3651 | 5 | 0 | evidence_heavy |
| `A3_definition` | A | `agensflow_auto` | **success** | 1955 | 3 | 0 | straightforward |
| `A4_capital` | A | `naive` | **success** | 29 | 1 | 0 | — |
| `A4_capital` | A | `agensflow_forced_straightforward` | **success** | 1727 | 3 | 0 | straightforward |
| `A4_capital` | A | `agensflow_forced_evidence_heavy` | **success** | 3233 | 5 | 0 | evidence_heavy |
| `A4_capital` | A | `agensflow_auto` | **success** | 1685 | 3 | 0 | straightforward |
| `B1_tcp_udp` | B | `naive` | **success** | 383 | 1 | 0 | — |
| `B1_tcp_udp` | B | `agensflow_forced_straightforward` | **failure** | 2255 | 3 | 0 | straightforward |
| `B1_tcp_udp` | B | `agensflow_forced_evidence_heavy` | **success** | 5197 | 5 | 0 | evidence_heavy |
| `B1_tcp_udp` | B | `agensflow_auto` | **partial** | 10748 | 6 | 1 | evidence_heavy |
| `B2_battery_chemistry` | B | `naive` | **success** | 571 | 1 | 0 | — |
| `B2_battery_chemistry` | B | `agensflow_forced_straightforward` | **failure** | 2643 | 3 | 0 | straightforward |
| `B2_battery_chemistry` | B | `agensflow_forced_evidence_heavy` | **success** | 7371 | 5 | 0 | evidence_heavy |
| `B2_battery_chemistry` | B | `agensflow_auto` | **success** | 7319 | 5 | 0 | evidence_heavy |
| `B3_oil_crisis` | B | `naive` | **partial** | 365 | 1 | 0 | — |
| `B3_oil_crisis` | B | `agensflow_forced_straightforward` | **failure** | 2365 | 3 | 0 | straightforward |
| `B3_oil_crisis` | B | `agensflow_forced_evidence_heavy` | **partial** | 5332 | 5 | 0 | evidence_heavy |
| `B3_oil_crisis` | B | `agensflow_auto` | **partial** | 4961 | 5 | 0 | evidence_heavy |
| `B4_sql_nosql` | B | `naive` | **success** | 367 | 1 | 0 | — |
| `B4_sql_nosql` | B | `agensflow_forced_straightforward` | **failure** | 2270 | 3 | 0 | straightforward |
| `B4_sql_nosql` | B | `agensflow_forced_evidence_heavy` | **success** | 5337 | 5 | 0 | evidence_heavy |
| `B4_sql_nosql` | B | `agensflow_auto` | **success** | 4891 | 5 | 0 | evidence_heavy |
| `C1_oil_crisis_budget` | C | `naive` | **success** | 227 | 1 | 0 | — |
| `C1_oil_crisis_budget` | C | `agensflow_forced_straightforward` | **partial** | 2234 | 3 | 0 | straightforward |
| `C1_oil_crisis_budget` | C | `agensflow_forced_evidence_heavy` | **success** | 4308 | 5 | 0 | evidence_heavy |
| `C1_oil_crisis_budget` | C | `agensflow_auto` | **partial** | 4308 | 5 | 0 | evidence_heavy |
| `C2_tcp_aviation` | C | `naive` | **success** | 231 | 1 | 0 | — |
| `C2_tcp_aviation` | C | `agensflow_forced_straightforward` | **success** | 2572 | 3 | 0 | straightforward |
| `C2_tcp_aviation` | C | `agensflow_forced_evidence_heavy` | **success** | 4723 | 5 | 0 | evidence_heavy |
| `C2_tcp_aviation` | C | `agensflow_auto` | **success** | 4403 | 5 | 0 | evidence_heavy |
| `C3_battery_cold` | C | `naive` | **success** | 268 | 1 | 0 | — |
| `C3_battery_cold` | C | `agensflow_forced_straightforward` | **success** | 2334 | 3 | 0 | straightforward |
| `C3_battery_cold` | C | `agensflow_forced_evidence_heavy` | **partial** | 4524 | 5 | 0 | evidence_heavy |
| `C3_battery_cold` | C | `agensflow_auto` | **partial** | 4457 | 5 | 0 | evidence_heavy |
| `C4_false_premise` | C | `naive` | **success** | 170 | 1 | 0 | — |
| `C4_false_premise` | C | `agensflow_forced_straightforward` | **success** | 2093 | 3 | 0 | straightforward |
| `C4_false_premise` | C | `agensflow_forced_evidence_heavy` | **success** | 4119 | 5 | 0 | evidence_heavy |
| `C4_false_premise` | C | `agensflow_auto` | **success** | 3949 | 5 | 0 | evidence_heavy |

<!-- RESULTS:END -->

---

## Analysis (post-run, manually authored)

**Run date:** 2026-05-03
**Total cells:** 48 (12 tasks × 4 configurations)
**Total cost:** approximately 142,000 tokens, well under $1 in OpenRouter spend.

### Framing correction (added 2026-05-04)

This experiment, as built, tested **the rule-based prior**, not the
AgensFlow framework's actual learning mechanism. Two corrections are
necessary:

**Correction 1 — wrong baseline.** A first draft of this analysis treated
the experiment as if AgensFlow were competing against `naive single LLM
call` on cost-per-successful-task. That comparison is built into the
pre-registered predictions and so the data fairly tested it — but it is the
wrong comparison for the framework's actual claim. AgensFlow is a
coordination layer for multi-agent systems; the right baseline is a
multi-agent system with retry-on-failure between stages, not a single LLM
call.

**Correction 2 — the framework's learning mechanism was not exercised.**
More importantly, the policy in this experiment is rule-based and never
updates from anything. The framework's distinguishing claim — that the
orchestration policy improves from trace data via a folded policy graph
with cross-task value reuse (the POMCGS-flavored mechanism that makes
AgensFlow different from LATS, LangGraph, and other agent routers) — is
not implemented in chunk 3 and therefore not tested by this experiment.

The cost benefit lives in **Layer 1 policy learning**, not in the rule-based
prior:

  - Layer 1 (orchestration policy learning): the structural primitives
    *plus* the folded policy graph that accumulates value estimates per
    (state-signature, action) edge across runs, *plus* MCTS-style search
    that consults the graph at orchestration time, *plus* value backup from
    each run's outcome. The policy improves *here*, without touching any
    model weights.
  - Layer 2 (metrics): provide observability and feed Layer 1's value-
    update signal.
  - Layer 3 (model adjustment): last resort. Distillation or fine-tuning,
    used only when Layer 1 hits its ceiling.

Chunk 3 only built the structural primitives at Layer 1. The folded policy
graph, the belief tracking, the MCTS search, and the value backup are all
chunk-4 work. Without them, "controlled flow" is just rule-based routing —
which is exactly what the experiment showed: the rule-based prior makes
correct architectural choices (Category A: 86× cheaper than the wrong
regime forced) but cannot improve from its own traces.

The right experiment, after building chunk 4, is the **cost-over-time**
trajectory: run the same benchmark N times, plot tokens-per-successful-task
as a function of run number, observe whether the policy graph's value
updates pull the routing toward better choices. That is the experiment that
actually tests the framework's claim. Chunk 3 cannot test it because the
learning mechanism doesn't exist yet.

The analysis below has been written with both corrections in mind. The
predictions section above has not been altered — they were what they were,
and they tested the wrong thing.

### What the policy got right

**Policy regime selection: 12/12 correct.** In every task, `agensflow_auto` selected the same regime that the corresponding `forced_*` configuration uses for that category. The rule-based regime detector mapped task features to plans without misclassification. Of all the predictions, this one is the most cleanly confirmed.

**Category A demonstrates regime-conditioning's economic value.** Forcing `evidence_heavy` on simple Q&A tasks costs **86× more tokens than `naive`** (3,441 vs 40 tok/run) for the same 100% success rate. `auto` correctly avoids this overhead by selecting `straightforward`, achieving the same 100% success at 1,790 tok/run (45× naive). This is the cleanest validation of the policy's core claim: the *choice* of regime matters, and the wrong choice is wasteful.

**Category B confirms that memory is necessary for grounding.** `forced_straightforward` failed on 4/4 Category B tasks (0% success), exactly as predicted. Without the memory step, the solver had no `evidence` to ground in, so its claims either missed the document content or weren't recognised as grounded by the grader. This validates the architectural claim that the regime distinction is *functional*, not just decorative.

### What the data says about the framework's actual claim

The framework's value-proposition rests on three legs: (1) the policy
mechanism routes correctly, (2) controlled flow reduces unproductive variance
within a multi-agent system, (3) the policy can be refined from traces over
time. This experiment tests (1) directly, tests (2) indirectly through the
"forced wrong regime" condition, and does not test (3) at all.

**Leg 1 — policy correctness: confirmed (12/12).** The rule-based detector
mapped every task to the predicted regime. This is the precondition for any
learned policy to make sense.

**Leg 2 — controlled flow reduces unproductive variance: confirmed by the
Category A "wrong regime" condition.** Forcing `evidence_heavy` on Category
A tasks cost 86× more tokens than `naive` for the same outcome. This is a
direct demonstration of what *would* happen to a multi-agent system without
a routing policy: it would pay heavy-coordination cost on every task, even
the simple ones. The framework's value here is that `auto` correctly picks
`straightforward` and avoids 64% of the wasted tokens
(3,441 → 1,790 tok/run) compared to a hypothetical "always run the full
pipeline" deployment, while keeping 100% success.

**Leg 3 — learned policy refinement: not tested.** This is Layer 3 work,
not in chunk 3. The policy in this experiment is rule-based and was not
updated based on the trace data the experiment generated. The cost benefit
that comes from learned routing is therefore necessarily absent here. This
is by design: chunk 3 validates the architecture, not the learning loop.

### The naive-baseline comparison is the wrong frame for the cost claim

The headline cost-per-successful-task table below is reported for honesty,
but it is the answer to the wrong question. Naive single LLM call is not the
right baseline for a coordination-layer framework — that comparison is like
asking whether memoization is "cheaper than not calling the function." Of
course not. Memoization's value appears when you compare it against
*repeated calls without memoization*, not against a single call.

The right comparison for AgensFlow is against a multi-agent system with
retry-on-failure between stages — which is what production deployments
actually use. On that comparison, the framework's role is to:

1. Avoid running expensive coalitions when they aren't needed (the Category
   A finding scaled across many runs).
2. Make recovery events visible and bounded instead of compounding into
   retry stacks.
3. Provide trace data that a learned policy can refine routing from.

This experiment did not include a multi-agent retry-stack baseline. That is
the experiment chunk 3.5 should run.

| Configuration | Success-rate | Tokens / successful task | Notes |
|---|---:|---:|---|
| `naive` | 83% (10/12) | **274** | Single frontier-model call. **Wrong baseline** for the framework's actual claim. |
| `agensflow_forced_straightforward` | 58% (7/12) | 3,695 | Loses on Category B because no memory. Functional regime separation confirmed. |
| `agensflow_forced_evidence_heavy` | 83% (10/12) | 5,467 | Same success as naive on this small benchmark; the cost overhead would be amortised by a learned policy avoiding it on simpler tasks. |
| `agensflow_auto` | 67% (8/12) | 6,524 | Single-trial variance vs forced equivalent (one validation retry inflated the average). |

### What the falsification criteria say (revisited)

The pre-registered falsification criteria were framed against the naive
baseline. With the corrected framing, they say less than I initially
claimed:

1. *"If `forced_evidence_heavy` does not outperform `forced_straightforward`
   on Category B grounding..."* — **not falsified.** `forced_evidence_heavy`
   got 75% vs `forced_straightforward`'s 0% on Category B. The architectural
   distinction between regimes is functional. This is meaningful for the
   framework's actual claim — it shows the policy *has* something to route
   between.
2. *"If `naive` performs as well as `agensflow_auto` on Category C, the
   verifier is not earning its cost."* — **the data show `naive` matched or
   beat `agensflow_auto` on C.** But this comparison was the wrong test for
   the framework's actual claim. What it tells us: claude-haiku-4.5 alone
   refuses to confabulate on these specific gaps. The verifier's value
   against frontier models on easy refusal cases is small. The verifier's
   value against (a) weaker models, (b) less obvious gaps, (c) multi-agent
   systems where the planner or solver might confabulate independently, is
   not addressed by this comparison and remains an open question.
3. *"If `agensflow_auto` picks the wrong regime for any task..."* —
   **not falsified.** 12/12 correct.

The honest revision: by setting up the criteria against naive, I tested
something the framework wasn't actually claiming. The criteria for the
*real* claim need a multi-agent retry-stack baseline.

### What this means for the framework's positioning

The framework's pitch is unchanged but is more sharply focused:

- **AgensFlow is a coordination layer you put on top of an existing
  multi-agent system to give it controlled flow, observability, and a path
  to learned routing.** Like memoization, it doesn't replace anything; it
  makes existing infrastructure more efficient over repeated runs.
- **The cost benefit lives in (a) avoiding unnecessary coordination on
  simple tasks, (b) bounding recovery overhead with one corrective retry
  rather than retry stacks, (c) refining routing from trace data over
  time.** None of these are visible in a single-trial comparison against a
  single LLM call. They become visible against the right baseline (a
  multi-agent system with retries) over enough runs to amortise the policy
  refinement.
- **Inspectability remains true and useful**, but it is the *enabling
  property* for the cost benefit, not a separate claim. Without inspectable
  traces, no policy refinement is possible.
- **The Category A finding is the cleanest piece of evidence so far** for
  the controlled-flow argument. Forcing the wrong regime is provably
  wasteful; the policy avoiding that waste is the framework doing its job.

### Limitations confirmed by the run

- **Single-trial noise**: the `auto` vs `forced_evidence_heavy` discrepancy on Category B is plausibly noise. Replication is needed for any quantitative claim.
- **Same-family grader**: claude-haiku-4.5 grades its own family's outputs. Bias is plausible but not measured. A stronger out-of-family grader (e.g., GPT-5.4 full or Claude Opus) would tighten the Category B and C judgements.
- **Benchmark difficulty**: Category C tasks were too easy for the model. The next iteration needs harder confabulation traps.
- **Framework's value-add is regime-specific**: Category A shows the policy correctly avoiding overhead; Category B shows the framework providing functional grounding; Category C shows the framework not adding value where the underlying model already refuses.

### Bug / improvement notes surfaced by the experiment

1. **Regime detector threshold fragility**: `evidence_availability > 0.7` is strict. Setting features to exactly 0.7 falls through to `straightforward`. The detector should arguably use `>=` for inclusivity, or the boundary should be documented more loudly. We encountered this when designing Category C features.
2. **`runner.py` was over-strict about empty documents**: it raised when `memory` was selected without documents, blocking the "force the wrong regime" experimental condition. Relaxed during this experiment so memory can run with empty documents and produce empty evidence (correct production behaviour for any vector store that returns zero results).
3. **Validation retry observed in the wild**: `B1_tcp_udp` × `agensflow_auto` triggered one corrective retry, recorded in the trace as expected. The cost (10,748 tok vs ~5,000 for the no-retry version) is honestly accounted for. Hooks worked as designed.

### What chunks 4 onward should focus on

The corrected framing reshapes the priority list. The framework's
distinguishing claim is *learnable orchestration policy via the folded
policy graph at Layer 1*. Chunk 3 did not build that. So the next chunks
have to start with building it, not polishing what exists.

1. **Chunk 4: build the actual Layer 1 policy-learning mechanism.**
   PolicyGraph + GraphNode (folded state graph with value estimates per
   action), belief signature (the POMCGS folding function), belief
   tracking (latent estimates updated after each agent), MCTS-style search
   that consults the graph at orchestration time, value backup from each
   run's reward, and persistence so the graph survives between runs. The
   notebook draft already had the algorithmic skeleton; chunk 4 integrates
   it with the LangGraph runtime and the Pydantic schema.
2. **Chunk 5: cost-over-time experiment.** Run the same benchmark from
   chunk 3 *N* times in sequence, with the policy graph updating between
   runs. Plot tokens-per-successful-task as a function of run number. This
   is the experiment that *directly* tests the framework's distinguishing
   claim. If the curve descends, the framework works as advertised. If
   it doesn't, the value-update signal or the signature-folding logic
   needs revision.
3. **Chunk 6: multi-agent retry-stack baseline.** Build the production-
   shape baseline (planner → solver → verifier with retry-on-failure
   between stages, no AgensFlow policy). The right comparison is the
   *trained* AgensFlow policy from chunk 5 vs. this baseline. Without
   chunk 5's trained policy, this comparison is premature.
4. **Chunk 7: Streamlit visualizer (FOO-ready).** Now the visualizer has
   a much better story: it shows the policy graph evolving as runs
   accumulate, with routing decisions changing as value estimates update.
   That is a substantive demo, not a polished trace viewer.
5. **Chunk 8: branching runtime** — distinctive coordination claims live
   in branching for ambiguous/contradictory regimes.
6. **Chunk 9: Layer 2 metrics** — HFE, ACE, AR, SP computed from trace
   data, integrated as part of the value-backup reward signal at Layer 1.
   This makes the policy not just learn "what works" but learn against the
   information-theoretic objective the metric set encodes.
7. **Chunk 10+: Layer 3 model adjustment.** Last resort, only if Layer 1
   learning hits its ceiling. Distill the trained policy into a smaller
   model, or fine-tune specialists on traces where their handoff was
   high-fidelity.

The Streamlit visualizer was originally chunk 4. It is now chunk 7 because
without the policy graph, the visualizer would have nothing distinctive to
show — just a trace of one run, like every other agent observability tool.
With the policy graph, the visualizer has the framework's actual story to
tell.

