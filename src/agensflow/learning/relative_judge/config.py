"""
RelativeJudgeConfig — typed configuration for RelativeJudge trajectory
scoring.

RelativeJudge is the framework's peer-relative scoring method: an LLM
judge sees N trajectories produced for the same task and ranks them
relative to each other against an explicit rubric. The judge model,
max-tokens budget, and the rubric are the user-facing knobs. The
default rubric is the one shipped in `core.DEFAULT_RUBRIC`; overriding
via YAML lets users adapt the rubric to their workload's quality
criteria without forking.

The method is inspired by the external RULER framework (Brown et al.,
distinct work). This is an in-house reimplementation, not a wrapper —
the framework's cross-judge averaging and per-axis decomposition
extensions are bolted on top.

See `README.md` for per-knob explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelativeJudgeConfig:
    """Configuration for `relative_judge_score_group`.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention.
    """

    # Judge model used to score the trajectory group. Defaults to
    # claude-haiku-4.5 — cheap, fast, and (per chunk-3) less prone to
    # the same-family bias the agents under test (gpt-nano + haiku
    # mix) might trip if the judge were the same family. For
    # production-grade grading consider gpt-5.4-mini or claude-sonnet-4.5,
    # acknowledging the cost trade-off.
    judge_model: str = "anthropic/claude-haiku-4.5"

    # Max output tokens for the judge call. 1500 covers ~5-trajectory
    # groups with explanations; raise for larger groups.
    max_tokens: int = 1500

    # Sampling temperature for the judge. Kept at 0.0 so the same
    # group of trajectories produces the same scores across re-runs —
    # critical for reproducible reward training.
    temperature: float = 0.0

    # Score returned for single-trajectory groups (relative scoring is
    # not applicable). 0.5 is the neutral center; raise to be more
    # generous with degenerate groups, lower to be more punishing.
    neutral_single_trajectory_score: float = 0.5

    # Rubric text shown to the judge. Defaults to the framework's
    # built-in 4-axis rubric (goal achievement, grounding, coordination
    # quality, recovery cleanliness). Override via YAML to inject
    # workload-specific quality criteria.
    #
    # Empty string means "use the in-code DEFAULT_RUBRIC". Most users
    # will leave this empty in YAML and override only when they want
    # to customize.
    rubric: str = ""

    # ----- evidence-decompression knobs (chunk 11.A1) ----- #
    # When `TrajectoryToScore.evidence` is provided AND `evidence_mode`
    # is not "off", these knobs control how the structured per-agent
    # contributions are rendered into the judge prompt.
    #
    # "off": skip evidence rendering entirely; judge sees only
    # `path_summary` (chunk-2..10 reproduction). Harness-side
    # `build_trajectory_evidence` calls also gate on this mode so no
    # cycles are wasted building evidence the prompt won't use.
    #
    # "budgeted" (default for chunk-11+): top-K memory snippets,
    # solver drafts truncated to `solver_draft_max_chars`, hard cap of
    # `evidence_max_chars_per_agent` per field. ~2-3x judge cost vs
    # path_summary alone.
    #
    # "full": no truncation. Full memory list, full solver drafts,
    # full verifier reasoning. ~5-10x judge cost. Reserved for
    # calibration / debug / cross-eval runs where the most-informative
    # judge signal matters more than per-call cost.
    evidence_mode: str = "budgeted"

    # Number of top memory-evidence snippets shown to the judge in
    # budgeted mode. Memory typically retrieves 5-10 candidates; the
    # top-K is enough for the judge to assess grounding without seeing
    # the full retrieval set.
    evidence_topk: int = 3

    # Hard cap on character length of each solver's draft answer in
    # budgeted mode. Solver drafts can run 3-10k chars; capping at 2k
    # keeps judge prompts manageable while preserving enough content
    # for grounding + correctness assessment.
    solver_draft_max_chars: int = 2000

    # Hard cap on character length of any single agent contribution
    # field (planner output, evaluator reasoning, verifier reasoning).
    # Catches pathologically long outputs without truncating typical
    # cases.
    evidence_max_chars_per_agent: int = 4000

    # ----- Cross-judge averaging + disagreement (chunk 11.A2) ----- #
    # When non-empty, `relative_judge_score_group` runs each model in this list
    # as an independent judge and aggregates scores (mean across
    # judges) with disagreement telemetry (std, range). When empty
    # (default), a single `judge_model` runs (chunk-2..10 behavior).
    #
    # Recommended cross-family triple for chunk-11+: one model from
    # each of {anthropic, openai, qwen}. The mode + extra_body required
    # for each is documented in `learning/ruler/README.md` and configured
    # via `cross_judge_modes` + `cross_judge_extra_body` below.
    #
    # Validated 3-family triple from chunk-11 probe:
    #   - anthropic/claude-haiku-4.5      TOOLS  + extra_body
    #   - openai/gpt-5.4-mini             TOOLS  + (no extra_body)
    #   - qwen/qwen3.6-flash              JSON   + extra_body
    cross_judge_models: list[str] = field(default_factory=list)

    # Per-judge Instructor mode (chunk 11.A2 probe finding). When a
    # judge's model id appears here, that mode is used for its
    # complete_typed call instead of the client's default. Values:
    # "tools" or "json". Missing keys → use client default ("tools").
    #
    # Why per-judge: OpenRouter's routing for some models (qwen-all,
    # gemini-flash) returns 404 "no endpoints support tool_choice"
    # under TOOLS mode; JSON mode bypasses tool_choice and works.
    cross_judge_modes: dict[str, str] = field(default_factory=dict)

    # Per-judge `extra_body` override (chunk 11.A2 probe finding).
    # When a judge's model id appears here, that extra_body is passed
    # to the OpenAI SDK call. Empty dict {} explicitly disables
    # extra_body for that judge (necessary for OpenAI's primary
    # OpenRouter route — `require_parameters: True` paradoxically
    # breaks it). Missing keys → use the global `extra_body_default`.
    #
    # The standard value is `{"provider": {"require_parameters": True}}`
    # which forces OpenRouter to route only to providers that fully
    # support all parameters (Instructor+OpenRouter integration pattern).
    cross_judge_extra_body: dict[str, dict] = field(default_factory=dict)

    # Default extra_body applied to judges NOT explicitly listed in
    # `cross_judge_extra_body`. Empty dict = no extra_body. Set to
    # `{"provider": {"require_parameters": True}}` for the recommended
    # chunk-11 default. Per-judge overrides above always win.
    extra_body_default: dict = field(default_factory=dict)

    # Disagreement-to-confidence calibration. Confidence is computed as
    # `1 - clamp(disagreement_std / threshold, 0, 1)`. Default 0.2
    # means: judge std of 0.2 (relatively wide disagreement on the
    # 0..1 scale) collapses confidence to 0; std of 0.1 → confidence
    # 0.5; std of 0.0 (judges agree) → confidence 1.0.
    #
    # Tighter (lower) threshold = more sensitive to small judge
    # disagreement; looser (higher) = only blatant disagreement
    # downweights confidence. 0.2 is a defensible mid-point given
    # RelativeJudge's [0, 1] score range.
    #
    # Single-judge mode (cross_judge_models empty) always reports
    # confidence=1.0 since there's no disagreement to measure.
    disagreement_confidence_threshold: float = 0.2

    # ----- Per-axis rubric scoring (chunk 11.A3) ----- #
    # The composed scalar RelativeJudge score is `sum(weight × axis_score)`
    # over the axes the judge returned. Default mirrors the four
    # axes in DEFAULT_RUBRIC (goal_achievement, grounding,
    # coordination, recovery) at equal-pair weighting (30/30/20/20).
    #
    # Tuning per workload: deep-research domains might raise
    # `grounding`; safety-critical domains might raise `recovery`;
    # high-volume customer-support might raise `coordination` (it
    # captures path conciseness / wasted-call penalty in the rubric).
    #
    # Empty dict {} = backward-compat path: composition reduces to
    # the judge's scalar score (chunk-2..10 behavior). Custom
    # rubrics that don't ask for axis_scores fall through this path
    # automatically.
    #
    # When the judge returns axis_scores keys NOT in this dict, they
    # are ignored (weight 0). When this dict has keys the judge did
    # NOT return, those axes are skipped from composition (so the
    # weighted sum is over the intersection, normalized by the
    # intersection's total weight). This makes the framework
    # forgiving to schema mismatch without silently misweighting.
    axis_weights: dict[str, float] = field(default_factory=lambda: {
        "goal_achievement":   0.30,
        "grounding":          0.30,
        "coordination":       0.20,
        "recovery":           0.20,
    })
