# `agensflow.learning.relative_judge`

**RelativeJudge** — the framework's peer-relative LLM-judge scoring
method. Anchors the quality component of `compute_hybrid_reward`.

## Naming attribution

The method is **inspired by** the external [RULER](https://github.com/OpenPipe/ART)
framework (Brown et al., distinct work). This package is an
*in-house reimplementation* of the relative-ranking-against-a-rubric
pattern — **not a wrapper around RULER itself**. We reimplemented
because:

- We need it to integrate with `OpenRouterClient` and Instructor-
  validated structured output (the typed Pydantic boundary catches
  judge schema drift early).
- We add cross-judge averaging (`cross_judge_models`), per-axis
  decomposition (`axis_weights`), and disagreement-derived confidence
  weighting on top of the basic peer-ranking idea.
- The framework's hybrid-reward + UCB-on-signatures loop relies on
  comparative scoring with a specific cost-penalty composition that's
  easier to maintain in-house than to graft onto an external library.

The previous name in this codebase was `ruler` (matching the
external framework's name). It was renamed to `relative_judge` to
avoid the misattribution. Note that the JSONL data field
`ruler_score` is preserved across all committed experimental results
as a backward-compatible legacy name; the method's class and
function identifiers all use `RelativeJudge` / `relative_judge`.

## Purpose

The chunk-5 finding pinned why **relative ranking against an explicit
rubric** matters: when the policy can route to its own internal
evaluator and that evaluator's `done=True` flag is worth +1.0 in v1
reward, the policy learns to game the evaluator rather than improve
the answer. RelativeJudge addresses this by:

- **Comparing trajectories side-by-side.** The judge sees N
  trajectories produced for the same task and ranks them relative to
  each other. Gaming the judge would require gaming all comparators
  simultaneously — much harder to optimize for.
- **Anchoring on an EXPLICIT rubric.** What "good" means lives in
  text the user can read, edit, and version-control — not buried in a
  learned scorer. The rubric is the operational anchor (the *q* in
  the framework's information-theoretic framing).
- **Producing graded comparative reward.** UCB exploration produces
  pairs/groups of trajectories at the same signature; RelativeJudge
  ranks them, the hybrid reward maps the rank to a scalar, and the
  substrate's value backup gets a clean comparative signal.

## Architecture

```
relative_judge_score_group(user_task, trajectories, client, config=RelativeJudgeConfig())
  └─ if 0 trajectories: return empty group
  └─ if 1 trajectory: return neutral_single_trajectory_score (relative N/A)
  └─ build judge prompt: user_task + rubric + trajectories
  └─ client.complete_typed(
         model=config.judge_model,
         system_prompt=_RULER_SYSTEM_PROMPT,
         user_prompt=...,
         output_model=_RelativeJudgement,
         temperature=config.temperature,
         max_tokens=config.max_tokens,
       )
  └─ map _RelativeJudgement → RulerScoreGroup
  └─ defensively fill missing trajectory_ids with neutral score
```

## Configuration knobs

| knob | default | what it controls | tune when |
|---|---|---|---|
| `judge_model` | `anthropic/claude-haiku-4.5` | which model judges | production-grade grading — `openai/gpt-5.4-mini` or `anthropic/claude-sonnet-4.5` (accept cost trade-off); cross-family bias mitigation — pick a family NOT used by your agents under test |
| `max_tokens` | 1500 | judge output token budget | groups >5 trajectories — raise to 2500 |
| `temperature` | 0.0 | judge sampling temperature | rarely tune — 0.0 is what makes scores reproducible across re-runs |
| `neutral_single_trajectory_score` | 0.5 | score for single-trajectory groups | rarely tune — change only if your reward shape needs a different neutral |
| `rubric` | `""` (use built-in `DEFAULT_RUBRIC`) | rubric text shown to the judge | workload-specific criteria — override with your domain's quality criteria |
| `evidence_mode` (chunk 11.A1) | `"budgeted"` | how `TrajectoryEvidence` is rendered into the prompt — `"off"` falls back to path_summary only, `"budgeted"` truncates, `"full"` keeps everything | calibration / cross-eval runs — set `"full"` (~5-10x judge cost); production sweeps — leave `"budgeted"` (~2-3x cost); chunk-2..10 reproductions — `"off"` |
| `evidence_topk` (chunk 11.A1) | 3 | top-K memory snippets shown in budgeted mode | high-grounding domains — raise to 5-8 |
| `solver_draft_max_chars` (chunk 11.A1) | 2000 | per-solver draft truncation | long-form solver outputs (research synthesis) — raise to 4000-6000 |
| `evidence_max_chars_per_agent` (chunk 11.A1) | 4000 | hard cap on planner/evaluator/verifier reasoning length | pathological long outputs — keep low; debugging long reasoning chains — raise |
| `cross_judge_models` (chunk 11.A2) | `[]` | judges to run for averaging + disagreement telemetry — empty = single-judge mode (chunk-2..10) | bias-mitigation / confidence-aware backup — set to a cross-family triple like `["anthropic/claude-haiku-4.5", "openai/gpt-5.4-mini", "qwen/qwen3.6-flash"]`; cost trade-off is linear in the number of judges |
| `disagreement_confidence_threshold` (chunk 11.A2) | 0.2 | judge-std value at which derived confidence collapses to 0; calibrates the disagreement-to-confidence map | tighter (lower) threshold = more sensitive to small judge disagreement; looser (higher) = only blatant disagreement downweights confidence |
| `axis_weights` (chunk 11.A3) | `{goal_achievement: 0.30, grounding: 0.30, coordination: 0.20, recovery: 0.20}` | per-axis weights for composing the scalar RelativeJudge score from the judge's per-axis dict; matches the four axes in `DEFAULT_RUBRIC` | research workloads — raise `grounding`; safety-critical — raise `recovery`; high-volume customer-support — raise `coordination`; chunk-2..10 reproduction — set to `{}` (disables composition, holistic scalar used) |

The built-in `DEFAULT_RUBRIC` ranks on four equally-weighted axes:
goal achievement, grounding, coordination quality, recovery cleanliness.

Defaults ship in `agensflow/configs/defaults/relative_judge.yaml`.

## Usage

### Default (built-in rubric, path-summary only):

```python
from agensflow.learning.relative_judge import (
    RelativeJudgeConfig, TrajectoryToScore, relative_judge_score_group,
)
group = relative_judge_score_group(
    user_task="What is the capital of France?",
    trajectories=[
        TrajectoryToScore(trajectory_id="t1", final_answer="Paris", path_summary="..."),
        TrajectoryToScore(trajectory_id="t2", final_answer="Lyon", path_summary="..."),
    ],
    client=client,
)
print(group.score_for("t1"))  # ~1.0
print(group.score_for("t2"))  # ~0.0
```

### With per-axis rubric scoring (chunk 11.A3):

`DEFAULT_RUBRIC` ships with four named axes: `goal_achievement`,
`grounding`, `coordination`, `recovery`. The judge returns per-axis
scores in [0, 1]; the framework composes the effective scalar via
`RelativeJudgeConfig.axis_weights` (weighted sum over the intersection of axes
returned by the judge and axes declared in the config).

```yaml
# my-config.yaml — research-style workload weighting grounding heaviest
relative_judge:
  axis_weights:
    goal_achievement: 0.30
    grounding:        0.45    # heaviest — fact-checking matters
    coordination:     0.15
    recovery:         0.10
```

When per-axis cross-judge is enabled together (recommended chunk-11+
default), the substrate sees:

- the **composed scalar** (cross-judge mean of per-judge composed
  scalars) → flows into the hybrid reward,
- the **per-axis cross-judge mean** → tells you which axis a
  trajectory was strong/weak on, averaged across judges,
- the **per-axis disagreement std** → tells you which axis judges
  disagreed about (a goldmine for diagnosing rubric ambiguity vs
  judge competence per axis).

Chunk-2..10 reproduction: set `axis_weights: {}` to disable
composition entirely and use the judge's holistic scalar (the
chunk-9-and-earlier behavior).

### With cross-judge averaging (chunk 11.A2 — bias mitigation + confidence telemetry):

```yaml
# my-config.yaml
relative_judge:
  # Three judges from three families — same-family bias on any single
  # judge is averaged away. Cost is 3x judge spend vs single-judge.
  cross_judge_models:
    - "anthropic/claude-haiku-4.5"
    - "openai/gpt-5.4-mini"
    - "qwen/qwen3.6-flash"
  # Tighter threshold = more sensitive to disagreement. 0.2 is the default;
  # 0.1 is stricter (downweights at smaller std), 0.3 more permissive.
  disagreement_confidence_threshold: 0.2
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
group = relative_judge_score_group(
    user_task=task, trajectories=trajectories,
    client=client,
    judge_model="ignored-when-cross-judge-set",
    config=cfg.relative_judge,
)
# Each result carries per-judge scores + disagreement metrics.
result = group.scores["t1"]
print(f"mean: {result.score:.2f}  std: {result.disagreement_std:.2f}  confidence: {result.confidence:.2f}")
print(f"per judge: {result.per_judge_scores}")
```

When `confidence < 1.0`, the chunk-9 harness multiplies the hybrid
reward by confidence before backing up to the policy graph (chunk
11.A4 — confidence-weighted backup). This means low-confidence runs
contribute less to the substrate's value estimates without introducing
a cost-only bias: both the quality bonus and the cost/retry penalties
shrink at the same rate, so gradient direction is preserved while
magnitude is reduced.

### With structured evidence (chunk 11.A1):

```python
from agensflow.learning.relative_judge import (
    TrajectoryToScore, build_trajectory_evidence, relative_judge_score_group,
)

# At scoring time, the harness has the trace. Build evidence from it:
evidence = build_trajectory_evidence(trace.events)

trajectory = TrajectoryToScore(
    trajectory_id="t1",
    final_answer=record.final_answer,
    path_summary=" → ".join(record.path),
    evidence=evidence,  # structured per-agent contributions
)

# The judge now sees: planner output, top-K memory snippets, each
# solver's draft, verifier verdicts, evaluator reasoning — budgeted
# per RelativeJudgeConfig defaults (~2-3x judge cost vs path-summary alone).
group = relative_judge_score_group(
    user_task=task,
    trajectories=[trajectory, ...],
    client=client,
    config=cfg.relative_judge,  # picks up evidence_mode + budgets from YAML
)
```

### YAML-driven with custom rubric + production judge:

```yaml
# my-config.yaml
relative_judge:
  judge_model: "openai/gpt-5.4-mini"
  rubric: |
    Score from 0 to 1 on:
    1. Factual correctness (40%)
    2. Citation grounding (30%)
    3. Conciseness (15%)
    4. Format compliance (15%)
```

```python
from agensflow.config import load_config
from agensflow.learning.relative_judge import relative_judge_score_group, DEFAULT_RUBRIC

cfg = load_config("my-config.yaml")
rubric = cfg.relative_judge.rubric or DEFAULT_RUBRIC
group = relative_judge_score_group(
    user_task=task,
    trajectories=trajectories,
    client=client,
    judge_model=cfg.relative_judge.judge_model,
    rubric=rubric,
    max_tokens=cfg.relative_judge.max_tokens,
)
```

(`relative_judge_score_group` accepts the knobs as kwargs; the runner threads
them in from `cfg.relative_judge`. We don't pass the whole config object to
keep the function signature explicit about what it consumes.)

## Required environment

- The judge call goes through `OpenRouterClient`, so
  `OPENROUTER_API_KEY` must be set. Costs accrue under the same
  attribution headers as agent calls.

## Design notes

- **`RelativeJudgeConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module config — see
  `web_search/README.md` for the rationale.

- **Judge cost is tracked in a separate trace.** `relative_judge_score_group`
  builds its own `TraceCollector` for the judge call rather than
  mixing judge events into the routed-agent trace. This keeps the
  per-agent rollups (in `RunReport`) clean — judge cost is grading
  overhead, not agent activity, and conflating them would confuse
  per-agent reward backup.

- **Defensive missing-trajectory fill.** If the judge's structured
  output omits a trajectory ID we asked about (rare, but possible),
  we fill that ID with a neutral score and an explanation flagging
  the omission. Better to keep the run going with a clearly-marked
  neutral than to crash on a missing key.

- **Rubric is text, not structured.** The judge sees raw markdown.
  This is intentional: the rubric is a human-edited artifact whose
  exact wording matters for judge interpretation. Structured rubrics
  (weighted axes as YAML) would be tempting but would lose the
  expressive control natural-language gives.

## Caveats

### Judge model compatibility — **always check before adding a new judge**

Not every model on OpenRouter works as a RelativeJudge judge. There are
three distinct failure modes, in three different layers — labelling
all of them "qwen doesn't work" loses information.

**Layer 1 — OpenRouter routing 404.** The router can't find an
upstream provider that accepts the `tool_choice` value Instructor
sends. Failure is instant (≤0.1s). Affects:

| model | confirmed |
|---|---|
| `qwen/qwen3.6-flash` | yes — chunk-6 + chunk-11 probe |
| `qwen/qwen3.6-max-preview` | yes — chunk-11 probe |
| `qwen/qwen3.6-235b-a22b` | yes — chunk-11 probe |
| `google/gemini-3.1-flash` | yes — chunk-11 probe |

Likely a provider-side `tool_choice` shape mismatch with how
Instructor forces specific function names. Could be unblocked by
changing Instructor's `Mode` for that specific judge (`JSON` instead
of `TOOLS`), or by setting OpenRouter `provider:` preferences.
Investigation deferred to chunk 12+.

**Layer 2 — Model-side validation failure.** The model accepts the
tool call, takes time to respond, but the structured output doesn't
satisfy the Pydantic schema. Affects:

| model | confirmed | note |
|---|---|---|
| `google/gemini-3.1-pro-preview` | yes — chunk-6 + chunk-11 probe | returns null score entries under TOOLS mode |

**Layer 3 — Schema compliance failure.** The model produces output
that *technically* validates under the relaxed schema (empty
`axis_scores={}` allowed) but won't populate axes even when prompted
to. Surfaces as a SILENT degradation under chunk-11.A2 cross-judge
mode and a HARD failure under the chunk-11.A3 strict schema. Affects:

| model | confirmed | note |
|---|---|---|
| `x-ai/grok-4.3` | yes — chunk-11 probe + smoke v1/v2 | empty axis_scores under relaxed; strict schema unmasks it as "won't comply" |

**Validated working with chunk-11.A3 strict schema (axes required) +
chunk-11.A2 per-judge config infrastructure:**

| family | model | mode | extra_body | typical latency |
|---|---|---|---|---:|
| anthropic | `claude-haiku-4.5` | `tools` | `{provider: {require_parameters: True}}` | ~3-4s |
| anthropic | `claude-sonnet-4.6` | `tools` | `{provider: {require_parameters: True}}` | ~6s |
| openai | `gpt-5.4-mini` | `tools` | **`{}`** (require_parameters BREAKS OpenAI route) | ~1-2s |
| openai | `gpt-5.4` | `tools` | **`{}`** (same) | ~3-5s |
| qwen | `qwen3.6-flash` | **`json`** | `{provider: {require_parameters: True}}` | ~8s |
| qwen | `qwen3.6-max-preview` | **`json`** | `{provider: {require_parameters: True}}` | ~50s (slow but works) |
| xAI | `grok-4.3` | **`json`** | `{provider: {require_parameters: True}}` | ~23s |

**Recommended chunk-11+ cross-family triple:**
`claude-haiku-4.5` (TOOLS) + `gpt-5.4-mini` (TOOLS, no extra_body) +
`qwen/qwen3.6-flash` (JSON). Three families, fast latencies,
all-axes-populated 100% in the smoke v3 validation.

**Smoke v3 result (35 trajectories, epoch-1 subset of chunk-9 corpus):**
3/3 judges achieved 100% axis-population compliance across 27 multi-
trajectory groups. Zero call failures. Per-axis disagreement std
ranged 0.04-0.19 across scenario classes, exposing per-class
same-family bias (notably C5/C6 dropping 0.12-0.18 under cross-judge
vs single haiku) — the exact signal chunk-11.A2 was designed to
surface.

**Configuration for the recommended triple:**

```yaml
relative_judge:
  cross_judge_models:
    - "anthropic/claude-haiku-4.5"
    - "openai/gpt-5.4-mini"
    - "qwen/qwen3.6-flash"
  cross_judge_modes:
    "qwen/qwen3.6-flash": "json"
  cross_judge_extra_body:
    "openai/gpt-5.4-mini": {}   # explicit disable — extra_body breaks OpenAI
  extra_body_default:
    provider:
      require_parameters: true
```

**Known failures (per chunk-11 probe — also documented in the failure-
layers section below):**

- `google/gemini-3.1-pro-preview` — model-side null-field issue,
  unresolved (chunk-12 follow-up)
- `google/gemini-3.1-flash` — OpenRouter routing 404 even under JSON
- `mistralai/mistral-large`, `deepseek/deepseek-v3.5`,
  `meta-llama/llama-3.4-405b-instruct`, `cohere/command-a-2025-09` —
  all failed both TOOLS and JSON in the chunk-11 probe; investigation
  deferred

**Always probe new judge candidates against `_StrictRelativeJudgement`
before adding them to `cross_judge_models`** — see
`scripts/probe_qwen_judge.py` for the canonical probe script.

### Epistemic note: validate AFTER resolving known incompleteness

When a chunk has multiple known structural improvements pending AND
expensive validation runs queued, the order has to be: **all
structural improvements first, validation last.** Validating on
infrastructure you suspect is incomplete:

- gives you a result that subsequent improvements will invalidate
- spends money you can't recover
- AND the first validation tells you nothing useful — you'll re-run
  it anyway after the improvements land

The cost asymmetry is large: one chunk's delay on validation is
cheap (the work is already known); validate-then-re-validate is
roughly 2x the LLM budget plus the cognitive overhead of explaining
why the first result was misleading.

Concrete chunk-11 example: I initially proposed running the chunk-9
replay before A3 (per-axis rubric scoring). The smoke v3 result
shows several scenario classes (C5, C6) had ~0.18 RelativeJudge inflation
from same-family judge bias — the very bias chunk-11.A2's
cross-judge averaging exists to mitigate. A pre-A3 replay would
have shown the original "+0.09 RelativeJudge quality improvement" headline,
which we'd have known was wrong as soon as A3 + cross-judge ran. ~$30
of LLM cost would have produced no usable data. The right order
(structural improvements → validation) costs the same total but
produces results that aren't superseded by the next chunk.

### Epistemic note: don't accept "platform fails" without evidence

The chunk-6 cross-eval documented "qwen doesn't work, gemini doesn't
work." Chunk-11 inherited that as ground truth. It was wrong: the
qwen failures were our own missing config (`extra_body={"provider":
{"require_parameters": True}}` + `Mode.JSON` for tool-routing-failed
models). Both are documented in the Instructor+OpenRouter integration
guide and would have unblocked qwen the first time if we'd checked.

When you see "platform feature X fails" but X is the platform's
headline integration, the prior on misconfiguration is much higher
than the prior on broken docs. Specifically:

- **Read the error message at the layer it came from.** A 404 with
  "no endpoints support tool_choice" + "to learn more visit
  /docs/guides/routing" is OpenRouter's routing layer telling you
  about routing config, not "the model is incompatible."
- **Check the integration docs of the platform you're using.** For
  Instructor+OpenRouter that's
  `python.useinstructor.com/integrations/openrouter` — both the
  `extra_body` pattern and the `Mode.JSON` fallback are documented
  there.
- **Probe the validated working set when documenting failures.**
  "qwen-flash failed" → "all qwen variants failed under TOOLS but
  qwen-flash + qwen-max work under JSON mode" is a much more useful
  characterization, AND it's the difference between blocking a chunk
  on bias mitigation and not.

This rule applies beyond judge models — any time we conclude "we
can't do X with provider Y" before reading their integration guide,
we're probably wrong.

When ANY judge in `cross_judge_models` raises, the entire scoring
call fails (Instructor's `complete_typed` propagates the exception),
which kills RelativeJudge for that group. **Never add a judge to
`cross_judge_models` without verifying it's already in the validated
set.**

The validated working set, maintained in
`experiments/e06_cross_eval/run.py:DEFAULT_JUDGES`:

```python
{
    "haiku":  "anthropic/claude-haiku-4.5",
    "gpt":    "openai/gpt-5.4",
    "sonnet": "anthropic/claude-sonnet-4.6",
    "grok":   "x-ai/grok-4.3",
}
```

When evaluating a candidate new judge: run `experiments/e06_cross_eval`
with the candidate added to `DEFAULT_JUDGES` against a 5-10
trajectory subset before rolling it into production sweeps. If it
returns valid scores end-to-end, add it to the validated set + this
table. If it 4xx's or returns nulls, document the failure here so
the next person doesn't rediscover it.

### Other caveats

- **Same-family bias.** When the judge's family overlaps with the
  agents under test, the judge can prefer family-specific output
  patterns over actual quality. Default mitigation: judge with haiku
  while most production agents under test mix nano + haiku — the bias
  is partial. The chunk-11 cross-judge mode (`cross_judge_models`)
  averages across families to mitigate this structurally; the
  per-judge disagreement telemetry surfaces residual bias.

- **Single-trajectory groups get a flat neutral score.** This is
  intentional — relative scoring needs comparators. If you find
  yourself frequently scoring single-trajectory groups, batch them
  into larger groups upstream; otherwise the policy gets no useful
  comparative signal.

- **Cost adds up.** Each scoring call is one judge invocation with
  N trajectories × their final_answer + path_summary in the prompt.
  Configure `max_tokens` and judge model to control spend; the
  hybrid reward's cost-penalty term won't see the judge cost (it's
  outside the routed trace), so monitor separately if cost matters.

## Tests

`tests/test_ruler.py` covers happy-path scoring of multi-trajectory
groups, single-trajectory neutral score, empty group, defensive
missing-id fill, judge-cost tracking on the separate trace.
