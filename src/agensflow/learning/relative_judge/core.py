"""
RelativeJudge — peer-relative scoring of agent trajectories.

A custom implementation of the relative-ranking-against-a-rubric
pattern: an LLM judge scores N trajectories produced for the same task,
on a 0–1 scale, against a rubric that defines what "good" looks like.
The scores become the quality component of the hybrid reward.

The method is *inspired by* the external RULER framework
(https://github.com/OpenPipe/ART, Brown et al.), but this is a separate
in-house implementation — not a wrapper around RULER itself. We
reimplemented because:

  - We need it to integrate with `OpenRouterClient` and Instructor-
    validated structured output (the typed Pydantic boundary catches
    judge schema drift early).
  - The framework adds cross-judge averaging (`cross_judge_models`),
    per-axis decomposition (`axis_weights`), and disagreement-derived
    confidence weighting on top of the basic peer-ranking idea — see
    `RelativeJudgeConfig`.
  - JSONL records and dashboard code use the legacy field name
    `ruler_score` (preserved for backward-compatible data; renaming
    would invalidate committed experimental results).

Why relative ranking matters here:

  - It resists internal-state hacking of the reward function.
    comparing trajectories side-by-side against an explicit rubric
    makes "evaluator marked done=True but the answer was wrong" much
    harder for the policy to optimize toward.
  - It gives gradient signal across UCB exploration: when the policy
    tries two different routes (e.g. solver_fast vs. solver_capable)
    at the same signature, the judge ranks the resulting trajectories
    relative to each other, producing a clean comparative reward.
  - The rubric is the operational anchor — *q* in the framework's
    information-theoretic framing — and stays explicit and editable
    rather than buried in a learned scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.trace import TraceCollector

if TYPE_CHECKING:
    from agensflow.learning.relative_judge.config import RelativeJudgeConfig
    from agensflow.runtime.trace import TraceEvent


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolverContribution:
    """One solver agent's contribution to a trajectory.

    Trajectories often invoke multiple solver variants (chunk-9's
    9-cell action space). Capturing each as a separate contribution
    lets the judge reason about which variant produced what — vs.
    seeing only an aggregate path.
    """

    skill_name: str            # e.g. "solver_concise_haiku"
    model: str                 # the model id that ran
    draft: str                 # the solver's draft answer (truncated per config)


@dataclass(frozen=True)
class VerifierContribution:
    """One verifier agent's verdict + concerns."""

    skill_name: str            # e.g. "verifier_haiku"
    verdict: str               # "supported" | "partially_supported" | "unsupported"
    reasoning: str             # the verifier's reasoning text


@dataclass(frozen=True)
class TrajectoryEvidence:
    """Structured per-agent contribution for the RelativeJudge judge (chunk 11.A1).

    Replaces the compressed `path_summary: str` for callers that want
    the judge to reason about WHAT each agent contributed (vs. just
    which agents fired). The dataclass stays optional on
    `TrajectoryToScore` — backward-compat path is to leave `evidence`
    None and rely on `path_summary`.

    Budgeted by default (see `RelativeJudgeConfig.evidence_mode`): top-K
    memory snippets, truncated solver drafts, full verifier concerns
    (usually short), full evaluator reasoning. Full-transcript mode is
    an expensive opt-in for debug / calibration runs.

    Built from a list of `TraceEvent`s — the harness has access to
    those at scoring time. See `build_trajectory_evidence` below.
    """

    # Planner contribution: typically short — goal, subproblem,
    # constraints. Stored as a dict so it's flexible across regime
    # variants (some plans have different planner outputs).
    planner: dict[str, Any] | None = None

    # Memory contribution: top-K evidence snippets after retrieval.
    # The full evidence list is usually 5-10 entries; budgeted mode
    # ships the top-K (RelativeJudgeConfig.evidence_topk).
    memory_evidence: list[str] = field(default_factory=list)

    # Solver contributions: one entry per solver variant invoked. Each
    # carries the variant's skill_name, model, and (truncated) draft.
    solvers: list[SolverContribution] = field(default_factory=list)

    # Verifier contributions: one entry per verifier variant invoked.
    # Verifier outputs are usually short, so verdict + reasoning ship
    # in full (no truncation by default).
    verifiers: list[VerifierContribution] = field(default_factory=list)

    # Evaluator's structured output — done flag, reasoning, final answer.
    # Optional dict so the structure can vary across schema versions.
    evaluator: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrajectoryToScore:
    """One trajectory in a group to be RelativeJudge-scored.

    Two ways to specify the trajectory's content:
      - `path_summary`: compact agent-name string (chunk-2..10 default).
      - `evidence`: structured per-agent contribution (chunk 11.A1+).
        When `evidence` is non-None, the judge prompt includes the
        structured contributions in addition to the path summary;
        when None, the judge sees only path_summary (legacy behavior).
    """

    trajectory_id: str
    final_answer: str
    # Compact summary of the orchestration path — agent names + outcomes.
    # Kept short so the judge isn't overwhelmed by a full trace dump.
    path_summary: str = ""
    # Optional structured evidence (chunk 11.A1). When supplied, the
    # judge prompt includes per-agent contributions in addition to the
    # path summary; when None, the judge sees only path_summary.
    evidence: TrajectoryEvidence | None = None


class _TrajectoryScore(BaseModel):
    """One trajectory's score, returned by the judge — relaxed schema.

    `axis_scores` defaults to {} so legacy rubrics (no axes named) work
    without provoking validation retries. Used when
    `RelativeJudgeConfig.axis_weights` is empty (chunk-2..10 reproduction path).
    """

    trajectory_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    axis_scores: dict[str, float] = Field(default_factory=dict)


class _StrictTrajectoryScore(BaseModel):
    """Strict variant of _TrajectoryScore (chunk 11.A3) — `axis_scores`
    is REQUIRED with at least one entry.

    Used when `RelativeJudgeConfig.axis_weights` is non-empty (the chunk-11+
    default). Pydantic's tool-schema generation marks `axis_scores`
    as required (no default) so models can't silently skip it — a
    judge that returns an empty dict triggers Instructor's retry
    loop with the validation error in the corrective context.

    This is the load-bearing fix for the smoke-v1/v2 finding that
    gpt-5.4 and grok-4.3 were ignoring the prompt's axis-population
    instruction because the relaxed schema told them they could.
    """

    trajectory_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    axis_scores: dict[str, float] = Field(min_length=1)


class _RelativeJudgement(BaseModel):
    """The judge's structured output (relaxed schema)."""

    scores: list[_TrajectoryScore]


class _StrictRelativeJudgement(BaseModel):
    """Judge output when axis_scores is required (chunk 11.A3)."""

    scores: list[_StrictTrajectoryScore]


@dataclass(frozen=True)
class RelativeJudgeScoreResult:
    """Per-trajectory result from a RelativeJudge scoring call.

    Chunk 11.A2 added cross-judge support: when `RelativeJudgeConfig.cross_judge_models`
    is non-empty, multiple judges score each trajectory and the result
    carries per-judge scores + disagreement telemetry + a derived
    confidence. In single-judge mode (the chunk-2..10 default),
    `per_judge_scores` has one entry, disagreement is 0, and confidence
    is 1.0.

    Chunk 11.A3 added per-axis rubric scoring: when judges return
    `axis_scores` (dict[axis, [0..1]]), the framework composes the
    final scalar via `RelativeJudgeConfig.axis_weights` and surfaces both the
    per-axis means (across judges) and the per-axis disagreement std.
    The `score` field is the COMPOSED scalar (axis-aware when axes are
    present; falls back to the judge's holistic scalar otherwise).
    """

    trajectory_id: str
    score: float                                # composed scalar (axis-weighted) or fallback
    explanation: str                            # first judge's explanation (for legacy callers)
    # Chunk 11.A2 cross-judge fields:
    per_judge_scores: dict[str, float] = field(default_factory=dict)
    per_judge_explanations: dict[str, str] = field(default_factory=dict)
    disagreement_std: float = 0.0               # population std across judges' composed scalars
    disagreement_range: float = 0.0             # max - min across judges
    confidence: float = 1.0                     # 1 - clamp(std / threshold, 0, 1)
    # Chunk 11.A3 per-axis fields (empty when the judge only returned scalar):
    # `axis_scores` is the cross-judge mean per axis (one entry per axis any
    # judge scored). `per_judge_axis_scores` is the full {judge → {axis →
    # score}} matrix for telemetry and per-axis bias detection.
    # `per_axis_disagreement_std` is the population std per axis across
    # judges — the goldmine signal for "which axis did judges disagree on?"
    axis_scores: dict[str, float] = field(default_factory=dict)
    per_judge_axis_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    per_axis_disagreement_std: dict[str, float] = field(default_factory=dict)


@dataclass
class RelativeJudgeScoreGroup:
    """All trajectory scores from one RelativeJudge scoring call, plus metadata."""

    scores: dict[str, RelativeJudgeScoreResult]
    judge_model: str                            # legacy single-judge name (or "+"-joined)
    judge_tokens: int                           # sum across judges
    judge_latency_seconds: float                # sum across judges
    # Chunk 11.A2: list of judges actually run (length 1 in single-judge
    # mode, length N in cross-judge mode).
    per_judge_models: list[str] = field(default_factory=list)

    def score_for(self, trajectory_id: str) -> float:
        """Return the score for a trajectory, or 0.0 if missing."""
        result = self.scores.get(trajectory_id)
        return result.score if result is not None else 0.0

    def confidence_for(self, trajectory_id: str) -> float:
        """Return the cross-judge confidence for a trajectory, or 1.0
        if missing (treated as fully confident — same as single-judge)."""
        result = self.scores.get(trajectory_id)
        return result.confidence if result is not None else 1.0


# --------------------------------------------------------------------------- #
# Rubric handling
# --------------------------------------------------------------------------- #


DEFAULT_RUBRIC = """\
Score each trajectory from 0.0 (worst) to 1.0 (best) on these four axes
INDEPENDENTLY, then provide an overall holistic score:

1. **goal_achievement**: did the trajectory's final answer actually
   answer the user's question? Higher score for fully and accurately
   answering.

2. **grounding**: are the answer's factual claims supported by the
   information available to the trajectory (provided documents,
   retrieved evidence)? Penalize confabulation. If the available
   information was insufficient, the trajectory should say so
   explicitly — that itself is a high-quality outcome, not a low one.

3. **coordination**: did the orchestration path make sense for the
   task? Penalize wasted agent calls (e.g. invoking expensive
   verification when the answer was obviously correct, or skipping
   verification when it was needed). Reward concise, purposeful
   coordination.

4. **recovery**: did the trajectory avoid validation retries and
   self-correction loops? Penalize recovery events; reward
   first-attempt success.

REQUIRED — for EVERY trajectory, `axis_scores` MUST contain all four
of these keys, each mapped to a float in [0.0, 1.0]:

  axis_scores = {
    "goal_achievement": <float>,
    "grounding":        <float>,
    "coordination":     <float>,
    "recovery":         <float>,
  }

The framework composes its own weighted-axis scalar from these
values; do not omit any axis. Also produce a holistic `score` in
[0.0, 1.0] reflecting your overall judgment — captured as a sanity
check on the composed scalar.

Use the relative comparison across the trajectories to anchor the
scale — the best in the group should be near 1.0, the worst near 0.0,
with the others spaced according to relative quality. If all
trajectories are similarly good, scores can be close together near
the top. If all are similarly bad, scores can be close together near
the bottom.
"""


# --------------------------------------------------------------------------- #
# Score group
# --------------------------------------------------------------------------- #


def relative_judge_score_group(
    *,
    user_task: str,
    trajectories: list[TrajectoryToScore],
    client: OpenRouterClient,
    judge_model: str = "anthropic/claude-haiku-4.5",
    rubric: str = DEFAULT_RUBRIC,
    max_tokens: int = 1500,
    config: "RelativeJudgeConfig | None" = None,
) -> RelativeJudgeScoreGroup:
    """
    Score a group of trajectories relative to each other against a rubric.

    Returns a `RelativeJudgeScoreGroup` with one score per trajectory. The judge
    receives the user task, the rubric, and per-trajectory content. Two
    content modes:

      - **path_summary** (chunk 2..10 default): when trajectories carry
        only `path_summary`, the judge sees the coordination path
        string + final answer.
      - **structured evidence** (chunk 11.A1): when trajectories carry
        a `TrajectoryEvidence`, the judge ALSO sees per-agent
        contributions (planner output, top-K memory evidence, each
        solver's draft, verifier verdicts + reasoning, evaluator
        reasoning) — budgeted per `config`.

    `config` controls evidence rendering: mode (budgeted vs full),
    top-K memory snippets, draft / per-agent character caps. When
    `config` is None, defaults from `RelativeJudgeConfig()` apply (budgeted
    mode with conservative caps).

    The judge model defaults to claude-haiku-4.5. The same-family-bias
    caveat from chunk 6.5/7 is real; chunk 11.A2 adds cross-judge
    averaging as the structural fix.
    """
    if not trajectories:
        return RelativeJudgeScoreGroup(scores={}, judge_model=judge_model, judge_tokens=0,
                               judge_latency_seconds=0.0)

    # Resolve config — defaults if not supplied. Imported here to avoid
    # the cyclic-import-at-module-load problem (config imports core to
    # re-export RewardConfig-style; we avoid that by lazy-importing).
    if config is None:
        from agensflow.learning.relative_judge.config import RelativeJudgeConfig
        config = RelativeJudgeConfig()

    if len(trajectories) == 1:
        # Single-trajectory groups can't be scored relatively. Give a neutral
        # score and a clear explanation. Caller should batch trajectories into
        # groups of ≥2 for meaningful relative ranking.
        only = trajectories[0]
        return RelativeJudgeScoreGroup(
            scores={only.trajectory_id: RelativeJudgeScoreResult(
                trajectory_id=only.trajectory_id,
                score=config.neutral_single_trajectory_score,
                explanation="Single-trajectory group; relative scoring not applicable.",
            )},
            judge_model=judge_model,
            judge_tokens=0,
            judge_latency_seconds=0.0,
        )

    user_prompt = _build_judge_prompt(user_task, rubric, trajectories, config)

    # Resolve the judge list. Single-judge mode (chunk-2..10 default)
    # has cross_judge_models empty → run only `judge_model`. Multi-judge
    # mode uses the list as-is; the `judge_model` arg is ignored.
    judges = (
        list(config.cross_judge_models) if config.cross_judge_models
        else [judge_model]
    )

    # We don't go through the policy graph trace collector for the judge call —
    # this is reward-signal infrastructure, not a routed agent. We do collect
    # judge cost separately via a fresh TraceCollector so the caller can
    # account for grading overhead.
    judge_trace = TraceCollector()
    state_snapshot: dict[str, object] = {"_ruler_judge_call": True}

    # Schema selection: when axis_weights is non-empty, the framework
    # WANTS per-axis scores → use the strict schema that makes
    # axis_scores required (no default). Models that try to skip axes
    # trigger Instructor's bounded retry. When axis_weights is empty
    # (legacy / custom rubrics with no axes), use the relaxed schema
    # so the empty-axes path is allowed.
    output_model = (
        _StrictRelativeJudgement if config.axis_weights else _RelativeJudgement
    )

    # Run each judge sequentially. Per-judge failures are isolated:
    # if one judge errors out (validation exhaustion, transport error,
    # provider 5xx), the others' votes are preserved and the
    # cross-judge mean is computed over the survivors. Without this
    # isolation, a single flaky judge would kill the entire group's
    # scoring — the opposite of bias mitigation.
    per_judge_parsed: dict[str, _RelativeJudgement] = {}
    per_judge_errors: dict[str, str] = {}
    total_tokens = 0
    total_latency = 0.0
    judge_models_actual: list[str] = []  # populated with what the API echoed back
    for judge in judges:
        # Per-judge mode: explicit override from config.cross_judge_modes
        # if set, otherwise client default (TOOLS). Required for
        # qwen/gemini routes that 404 under TOOLS (chunk-11 probe).
        judge_mode = config.cross_judge_modes.get(judge)

        # Per-judge extra_body: explicit override (including {} to
        # disable) if listed in config.cross_judge_extra_body, otherwise
        # the configured default. Required for OpenAI compatibility
        # (extra_body breaks gpt-5.4-* routing on OpenRouter).
        if judge in config.cross_judge_extra_body:
            judge_extra_body: dict | None = (
                config.cross_judge_extra_body[judge] or None
            )
        elif config.extra_body_default:
            judge_extra_body = dict(config.extra_body_default)
        else:
            judge_extra_body = None

        try:
            result = client.complete_typed(
                model=judge,
                system_prompt=_RELATIVE_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_model=output_model,
                agent_name=f"_ruler_judge:{judge}",
                trace=judge_trace,
                state_snapshot=state_snapshot,
                max_retries=2,
                temperature=0.0,
                max_tokens=max_tokens,
                mode=judge_mode,
                extra_body=judge_extra_body,
            )
        except Exception as exc:  # noqa: BLE001 — broad on purpose
            # Per-judge isolation. Capture, surface in telemetry, move on.
            # The cross-judge mean falls back to the surviving judges.
            per_judge_errors[judge] = f"{type(exc).__name__}: {exc!s}"[:300]
            judge_models_actual.append(judge)  # echo back input name on error
            continue
        parsed = result.parsed_output  # type: ignore[assignment]
        per_judge_parsed[judge] = parsed
        total_tokens += result.total_tokens
        total_latency += result.latency_seconds
        judge_models_actual.append(result.model)

    # If ALL judges failed, fall back to the neutral group result
    # rather than crashing — caller sees zero confidence and a clear
    # explanation, downstream stays alive.
    if not per_judge_parsed:
        neutral = config.neutral_single_trajectory_score
        explanation = (
            f"All {len(judges)} judges failed: " +
            "; ".join(f"{j}: {e}" for j, e in per_judge_errors.items())
        )[:500]
        return RelativeJudgeScoreGroup(
            scores={
                t.trajectory_id: RelativeJudgeScoreResult(
                    trajectory_id=t.trajectory_id,
                    score=neutral,
                    explanation=explanation,
                    confidence=0.0,
                )
                for t in trajectories
            },
            judge_model="+".join(judge_models_actual),
            judge_tokens=total_tokens,
            judge_latency_seconds=total_latency,
            per_judge_models=judge_models_actual,
        )

    # Aggregate per-trajectory across judges.
    scores = _aggregate_cross_judge_scores(
        trajectories=trajectories,
        per_judge_parsed=per_judge_parsed,
        config=config,
    )

    # Legacy single-name `judge_model` field on RelativeJudgeScoreGroup: when
    # one judge ran, surface its echoed name; when N judges, "+"-join
    # the list so the legacy field is human-readable in reports.
    legacy_name = (
        judge_models_actual[0] if len(judge_models_actual) == 1
        else "+".join(judge_models_actual)
    )

    return RelativeJudgeScoreGroup(
        scores=scores,
        judge_model=legacy_name,
        judge_tokens=total_tokens,
        judge_latency_seconds=total_latency,
        per_judge_models=judge_models_actual,
    )


def _compose_score_from_axes(
    axis_scores: dict[str, float],
    axis_weights: dict[str, float],
) -> float | None:
    """Compose a scalar from per-axis scores via configured weights.

    Returns None when composition isn't possible (axis_scores empty, no
    overlap with axis_weights, or weight sum ≤ 0). Falls back to the
    judge's holistic scalar in that case.

    The weighted sum is computed over the INTERSECTION of `axis_scores`
    and `axis_weights` keys, normalized by the intersection's weight
    total. This means: judges returning extra/unknown axes get those
    ignored; configs declaring axes the judge didn't return get those
    skipped from composition (rather than silently treating their
    score as 0). The final scalar reflects only the axes actually
    scored, weighted as configured.
    """
    if not axis_scores or not axis_weights:
        return None
    overlap = {k: axis_weights[k] for k in axis_scores if k in axis_weights}
    total_w = sum(overlap.values())
    if total_w <= 0.0:
        return None
    weighted_sum = sum(axis_scores[k] * w for k, w in overlap.items())
    return weighted_sum / total_w


def _aggregate_cross_judge_scores(
    *,
    trajectories: list[TrajectoryToScore],
    per_judge_parsed: dict[str, "_RelativeJudgement"],
    config: "RelativeJudgeConfig",
) -> dict[str, RelativeJudgeScoreResult]:
    """Aggregate per-trajectory scores across N judges.

    For each trajectory:
      - For each judge: extract its holistic scalar + per-axis dict.
        Compose a per-judge effective scalar via `axis_weights` when
        axes are present; fall back to holistic scalar otherwise.
      - Cross-judge mean: average of effective scalars.
      - disagreement_std/_range: population stats across effective scalars.
      - confidence: 1 - clamp(std / threshold, 0, 1).
      - axis_scores (cross-judge): per-axis mean across judges that
        scored that axis.
      - per_axis_disagreement_std: per-axis population std.
      - per_judge_axis_scores: full {judge → {axis → score}} matrix.

    Missing-trajectory defense (per-judge): if a judge omitted a
    trajectory, that judge's slot is filled with a neutral scalar
    (no axes contributed). Other judges' signals are preserved.

    Composition asymmetry tolerated: some judges may return axes,
    others not. Per-judge composition handles this — judges without
    axes use their holistic scalar; judges with axes use the
    axis-weighted scalar. The cross-judge mean is over the resulting
    per-judge effective scalars regardless of their origin.
    """
    threshold = max(config.disagreement_confidence_threshold, 1e-9)
    scores: dict[str, RelativeJudgeScoreResult] = {}
    for traj in trajectories:
        per_judge_effective: dict[str, float] = {}      # judge → effective scalar
        per_judge_holistic: dict[str, float] = {}       # judge → holistic scalar (unused for legacy callers)
        per_judge_axes: dict[str, dict[str, float]] = {}  # judge → {axis → score}
        per_judge_explanations: dict[str, str] = {}

        for judge_name, parsed in per_judge_parsed.items():
            score_obj = next(
                (s for s in parsed.scores if s.trajectory_id == traj.trajectory_id),
                None,
            )
            if score_obj is None:
                # Defensive: judge skipped this trajectory. Neutral
                # scalar; no axis contribution.
                per_judge_effective[judge_name] = config.neutral_single_trajectory_score
                per_judge_holistic[judge_name] = config.neutral_single_trajectory_score
                per_judge_axes[judge_name] = {}
                per_judge_explanations[judge_name] = (
                    "(judge omitted this trajectory; neutral score assigned)"
                )
                continue

            per_judge_holistic[judge_name] = score_obj.score
            per_judge_axes[judge_name] = dict(score_obj.axis_scores)
            per_judge_explanations[judge_name] = score_obj.explanation

            # Compose effective scalar: prefer axis-weighted when axes
            # are present; fall back to holistic when not.
            composed = _compose_score_from_axes(
                score_obj.axis_scores, config.axis_weights,
            )
            per_judge_effective[judge_name] = (
                composed if composed is not None else score_obj.score
            )

        # Cross-judge aggregation on the effective scalars.
        scores_arr = list(per_judge_effective.values())
        if not scores_arr:
            scores[traj.trajectory_id] = RelativeJudgeScoreResult(
                trajectory_id=traj.trajectory_id,
                score=config.neutral_single_trajectory_score,
                explanation="(no judge produced a score; neutral fallback)",
                per_judge_scores={},
                per_judge_explanations={},
                disagreement_std=0.0,
                disagreement_range=0.0,
                confidence=0.0,
                axis_scores={},
                per_judge_axis_scores={},
                per_axis_disagreement_std={},
            )
            continue

        mean = sum(scores_arr) / len(scores_arr)
        if len(scores_arr) > 1:
            var = sum((s - mean) ** 2 for s in scores_arr) / len(scores_arr)
            std = var ** 0.5
            score_range = max(scores_arr) - min(scores_arr)
            confidence = max(0.0, min(1.0, 1.0 - std / threshold))
        else:
            std = 0.0
            score_range = 0.0
            confidence = 1.0

        # Per-axis cross-judge mean + std. For each axis, gather scores
        # from judges that returned it; mean + std over those values.
        # Axes are union across judges (some judges may have missed an
        # axis the others scored; we still report the partial signal).
        all_axes: set[str] = set()
        for axes in per_judge_axes.values():
            all_axes.update(axes.keys())
        axis_means: dict[str, float] = {}
        axis_stds: dict[str, float] = {}
        for axis in all_axes:
            vals = [
                axes[axis] for axes in per_judge_axes.values() if axis in axes
            ]
            if not vals:
                continue
            ax_mean = sum(vals) / len(vals)
            axis_means[axis] = ax_mean
            if len(vals) > 1:
                ax_var = sum((v - ax_mean) ** 2 for v in vals) / len(vals)
                axis_stds[axis] = ax_var ** 0.5
            else:
                axis_stds[axis] = 0.0

        explanation = next(
            (e for e in per_judge_explanations.values() if e),
            "",
        )

        scores[traj.trajectory_id] = RelativeJudgeScoreResult(
            trajectory_id=traj.trajectory_id,
            score=mean,
            explanation=explanation,
            per_judge_scores=per_judge_effective,           # the effective scalars
            per_judge_explanations=per_judge_explanations,
            disagreement_std=std,
            disagreement_range=score_range,
            confidence=confidence,
            axis_scores=axis_means,
            per_judge_axis_scores=per_judge_axes,
            per_axis_disagreement_std=axis_stds,
        )
    return scores


# --------------------------------------------------------------------------- #
# Internal: prompt construction
# --------------------------------------------------------------------------- #


_RELATIVE_JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring multi-agent trajectory quality.

You will receive:
  - The user's task.
  - A rubric describing what makes a trajectory high or low quality,
    including the named axes the trajectory is evaluated on.
  - A set of N trajectories produced for the same task.

REQUIRED OUTPUT FORMAT — one entry per trajectory, in a `scores` array:

  - `trajectory_id`: the trajectory identifier (string).
  - `axis_scores`: a JSON object whose keys are the named axes from
    the rubric, and whose values are floats in [0.0, 1.0]. **You MUST
    populate `axis_scores` with every axis named in the rubric.** This
    is the load-bearing field — do not leave it empty when the rubric
    names axes. If the rubric truly names no axes, only then may
    `axis_scores` be `{}`.
  - `score`: a holistic scalar in [0.0, 1.0] reflecting your overall
    judgment. The framework computes its own weighted-axis composite
    from `axis_scores`; your holistic `score` is captured as a sanity
    check.
  - `explanation`: a brief justification (1-3 sentences) covering the
    per-axis reasoning AND the overall verdict.

Use the relative comparison across trajectories to anchor the scale —
the best in the group near 1.0, the worst near 0.0, others spaced
according to relative quality. Output STRICT JSON only."""


def _build_judge_prompt(
    user_task: str,
    rubric: str,
    trajectories: list[TrajectoryToScore],
    config: "RelativeJudgeConfig | None" = None,
) -> str:
    """Render the user prompt for the judge.

    When a trajectory carries `evidence` (TrajectoryEvidence), the
    structured per-agent contributions are rendered into the prompt —
    budgeted per `config.evidence_mode` and the related caps. When
    evidence is None, the legacy path-summary-only flow is used
    (chunk-2..10 behavior).
    """
    if config is None:
        from agensflow.learning.relative_judge.config import RelativeJudgeConfig
        config = RelativeJudgeConfig()
    sections: list[str] = []
    sections.append(f"User task:\n{user_task}\n")
    sections.append(f"Rubric:\n{rubric}\n")
    sections.append("Trajectories to score:\n")
    for traj in trajectories:
        sections.append(f"--- trajectory {traj.trajectory_id} ---")
        if traj.path_summary:
            sections.append(f"Coordination path: {traj.path_summary}")
        if traj.evidence is not None:
            sections.extend(_render_evidence_section(traj.evidence, config))
        sections.append("Final answer:")
        sections.append(traj.final_answer.strip() or "(empty)")
        sections.append("")
    sections.append(
        f"Produce the verdict as JSON with one entry per trajectory id "
        f"({', '.join(t.trajectory_id for t in trajectories)})."
    )
    return "\n".join(sections)


def _truncate(s: str, max_chars: int) -> str:
    """Soft-truncate a string with an ellipsis marker. `max_chars <= 0`
    means no truncation."""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + "… [truncated]"


def _render_evidence_section(
    evidence: TrajectoryEvidence,
    config: "RelativeJudgeConfig",
) -> list[str]:
    """Render a TrajectoryEvidence into prompt-section lines.

    Mode controlled by `config.evidence_mode`:
      - "off": return []; judge sees only path_summary (chunk-2..10
        reproduction). Defensive — also called by `_build_judge_prompt`.
      - "full": no truncation; ship every field as-is.
      - "budgeted" (default): top-K memory, draft truncated to
        `solver_draft_max_chars`, all other fields capped at
        `evidence_max_chars_per_agent`.
    """
    if config.evidence_mode == "off":
        return []
    is_full = config.evidence_mode == "full"
    per_agent_cap = 0 if is_full else config.evidence_max_chars_per_agent
    draft_cap = 0 if is_full else config.solver_draft_max_chars
    topk = 0 if is_full else config.evidence_topk

    lines: list[str] = []
    if evidence.planner is not None:
        # Render planner as a JSON-ish block; keeps structure visible
        # to the judge without losing field names.
        planner_str = json.dumps(evidence.planner, default=str, indent=None)
        lines.append(f"Planner: {_truncate(planner_str, per_agent_cap)}")

    if evidence.memory_evidence:
        snippets = (
            evidence.memory_evidence
            if topk == 0
            else evidence.memory_evidence[:topk]
        )
        lines.append(f"Memory (top-{len(snippets)} of {len(evidence.memory_evidence)}):")
        for snip in snippets:
            lines.append(f"  - {_truncate(snip, per_agent_cap)}")

    if evidence.solvers:
        lines.append(f"Solver attempts ({len(evidence.solvers)}):")
        for s in evidence.solvers:
            draft = _truncate(s.draft, draft_cap) if draft_cap > 0 else s.draft
            lines.append(f"  [{s.skill_name} ← {s.model}]: {draft}")

    if evidence.verifiers:
        lines.append(f"Verifier verdicts ({len(evidence.verifiers)}):")
        for v in evidence.verifiers:
            reasoning = _truncate(v.reasoning, per_agent_cap) if per_agent_cap > 0 else v.reasoning
            lines.append(f"  [{v.skill_name}] {v.verdict}: {reasoning}")

    if evidence.evaluator is not None:
        ev_str = json.dumps(evidence.evaluator, default=str, indent=None)
        lines.append(f"Evaluator: {_truncate(ev_str, per_agent_cap)}")

    return lines


# --------------------------------------------------------------------------- #
# Helper: build TrajectoryEvidence from trace events (harness-side)
# --------------------------------------------------------------------------- #


def build_trajectory_evidence(
    events: "list[TraceEvent]",
    *,
    config: "RelativeJudgeConfig | None" = None,
) -> TrajectoryEvidence:
    """Construct a `TrajectoryEvidence` from a list of trace events.

    Lives in the ruler module (not the harness) so any caller with a
    trace can produce evidence without re-implementing the per-agent
    extraction logic.

    The function is conservative: when an agent's `output_update`
    doesn't have the expected fields (schema variance, partial runs),
    that contribution is skipped silently. The judge sees what the
    framework actually produced, not a hand-curated subset.

    Skip events (`agent.startswith("skip:")`) are excluded — they have
    no content for the judge to reason about.

    `config` is currently used only via type hint (no per-event
    truncation here; truncation happens at render time so the same
    `TrajectoryEvidence` can be re-rendered under different budgets
    without re-extracting from events).
    """
    if config is None:
        from agensflow.learning.relative_judge.config import RelativeJudgeConfig
        config = RelativeJudgeConfig()

    planner: dict[str, Any] | None = None
    memory_evidence: list[str] = []
    solvers: list[SolverContribution] = []
    verifiers: list[VerifierContribution] = []
    evaluator: dict[str, Any] | None = None

    for ev in events:
        if ev.error is not None:
            # Failed attempts don't contribute content; their cost is
            # captured elsewhere (governance counters, reward retries).
            continue
        agent = ev.agent
        if agent.startswith("skip:"):
            continue
        update = ev.output_update or {}

        if agent == "planner":
            # Surface the structured planner output for the judge.
            planner = {
                "goal": update.get("goal"),
                "subproblem": update.get("subproblem"),
                "constraints": update.get("constraints", []),
            }
        elif agent == "memory":
            ev_list = update.get("evidence")
            if isinstance(ev_list, list):
                # Normalize to list[str] — evidence entries may be
                # strings or dicts depending on the memory variant.
                memory_evidence.extend(
                    e if isinstance(e, str) else json.dumps(e, default=str)
                    for e in ev_list
                )
        elif agent.startswith("solver"):
            draft = update.get("draft_answer")
            if isinstance(draft, str) and draft:
                solvers.append(SolverContribution(
                    skill_name=agent,
                    model=ev.model,
                    draft=draft,
                ))
        elif agent.startswith("verifier"):
            verification_str = update.get("verification")
            if isinstance(verification_str, str) and verification_str:
                # Verifier output is JSON-encoded VerifierOutput.
                try:
                    parsed = json.loads(verification_str)
                    verifiers.append(VerifierContribution(
                        skill_name=agent,
                        verdict=str(parsed.get("verdict", "")),
                        reasoning=str(parsed.get("reasoning", "")),
                    ))
                except (json.JSONDecodeError, TypeError):
                    # Defensive — schema variance shouldn't crash extraction.
                    continue
        elif agent == "evaluator":
            metadata = update.get("metadata") or {}
            ev_meta = metadata.get("evaluator")
            if isinstance(ev_meta, dict):
                evaluator = {
                    "done": ev_meta.get("done"),
                    "reasoning": ev_meta.get("reasoning"),
                    # Final answer is captured at the trajectory level
                    # already; don't duplicate.
                }

    return TrajectoryEvidence(
        planner=planner,
        memory_evidence=memory_evidence,
        solvers=solvers,
        verifiers=verifiers,
        evaluator=evaluator,
    )
