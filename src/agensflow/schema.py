"""
Schema primitives for AgensFlow.

These types define the structured objects that flow through the orchestration
policy. They are deliberately small, typed, and validated, because the whole
thesis of AgensFlow rests on orchestration state being inspectable rather than
free-form text.

The Handoff object in particular is designed to carry upstream references so
that information-theoretic metrics over handoffs (e.g., handoff fidelity) can
be computed downstream without modifying the schema later.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Enums (kept as Literal types for cheap structural typing)
# --------------------------------------------------------------------------- #

RegimeLabel = Literal[
    "straightforward",
    "evidence_heavy",
    "ambiguous",
    "contradictory",
    "high_risk",
    "exploratory",
]
"""Coarse categorisation of task regimes that the activation policy conditions on."""

MergeStrategy = Literal[
    "select_best",
    "weighted_merge",
    "critic_select",
    "verifier_gate",
    "consensus",
]
"""Strategies for combining results across branches when branching is enabled."""

SkillKind = Literal["agent", "skill", "meta_skill"]
"""Coarse role of a registered specialist within the system."""


# --------------------------------------------------------------------------- #
# Task features
# --------------------------------------------------------------------------- #


class TaskFeatures(BaseModel):
    """
    Features of an incoming task that the regime detector reads.

    All numeric fields are in [0.0, 1.0]. They are intended as soft scores,
    not booleans. A learned regime classifier (Layer 2/3) would replace the
    rule-based defaults that currently consume these.
    """

    requires_factual_grounding: bool = False
    ambiguity_level: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_level: float = Field(default=0.0, ge=0.0, le=1.0)
    time_horizon_complexity: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_availability: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_sensitivity: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_need: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Regime estimate
# --------------------------------------------------------------------------- #


class RegimeEstimate(BaseModel):
    """
    Output of a regime detector.

    `confidence` is the detector's estimated confidence in `label`.
    `alternative_labels` are next-best candidates, in order. They are surfaced
    so downstream policy can hedge or branch when confidence is low.
    """

    label: RegimeLabel
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alternative_labels: list[RegimeLabel] = Field(default_factory=list)

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Skill spec
# --------------------------------------------------------------------------- #


class SkillSpec(BaseModel):
    """
    Specification for a registered specialist (agent, skill, or meta-skill).

    The `regime_affinity`, `branch_compatibility`, and `merge_preference` fields
    encode the institutional contract: under which regimes this specialist is
    appropriate, which other specialists it composes well with, and how its
    outputs prefer to be merged. These hints feed the activation planner.
    """

    name: str
    kind: SkillKind = "agent"
    preconditions: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    handoff_requirements: list[str] = Field(default_factory=list)
    preferred_successors: list[str] = Field(default_factory=list)

    confidence_effect: float = 0.0
    cost_estimate: float = 1.0

    regime_affinity: list[RegimeLabel] = Field(default_factory=list)
    branch_compatibility: list[str] = Field(default_factory=list)
    merge_preference: MergeStrategy = "select_best"

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Branch rule
# --------------------------------------------------------------------------- #


class BranchRule(BaseModel):
    """
    Conditional branching policy attached to an activation plan.

    Branching is only triggered when `enabled` is True and at least one of the
    threshold conditions is met. `branch_skill_sets` enumerates the candidate
    coalitions that may be instantiated as branches.
    """

    enabled: bool = False
    trigger_if_ambiguity_above: float = Field(default=1.0, ge=0.0, le=1.0)
    trigger_if_contradiction_above: float = Field(default=1.0, ge=0.0, le=1.0)
    max_branches: int = Field(default=1, ge=1)
    branch_skill_sets: list[list[str]] = Field(default_factory=list)

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Activation plan
# --------------------------------------------------------------------------- #


class ActivationPlan(BaseModel):
    """
    Output of the activation planner.

    Combines the detected regime, the selected coalition of specialists for
    that regime, the branching rule (if any), the merge strategy for combining
    branch results, and the evaluation criteria the evaluator should apply.
    """

    regime: RegimeEstimate
    selected_skills: list[str]
    branch_rule: BranchRule = Field(default_factory=BranchRule)
    merge_strategy: MergeStrategy = "select_best"
    evaluation_criteria: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Belief — latent estimates over the run state.
# --------------------------------------------------------------------------- #


class Belief(BaseModel):
    """
    Latent belief estimates that travel alongside the Handoff.

    The Handoff carries observable content (goal, evidence, draft_answer, ...).
    The Belief carries latent estimates inferred from the run so far:
    *how confident are we that the answer is correct?* *how thin is the
    evidence?* *how likely is contradiction?*

    These estimates feed two things:
      1. The folded policy graph's signature (Layer 1 learning).
      2. Eventually, the metric layer's reward signal (Layer 2).

    For the framework's first contact with a task, all estimates start at
    sensible defaults (low correctness, high uncertainty, no contradiction
    seen yet). After each agent call, deltas are applied based on what that
    agent contributed (planner increases handoff_quality, verifier increases
    correctness when verdict is supported, critic increases contradiction
    risk, etc.). The update rules live in `agensflow.learning.belief`.

    The estimates are deliberately heuristic at this stage — symbolic
    Bayesian-style updates. A future Layer 2/3 piece will replace the
    update rules with learned belief updaters anchored to observable
    proxies (logprobs, verifier outcomes, agreement across agents).
    """

    estimated_correctness: float = Field(default=0.1, ge=0.0, le=1.0)
    estimated_uncertainty: float = Field(default=0.9, ge=0.0, le=1.0)
    estimated_handoff_quality: float = Field(default=0.2, ge=0.0, le=1.0)
    estimated_contradiction_risk: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_evidence_sufficiency: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = {"frozen": True}


# --------------------------------------------------------------------------- #
# Handoff
# --------------------------------------------------------------------------- #


class Handoff(BaseModel):
    """
    The structured object passed between specialists.

    A Handoff is the search state of the orchestration policy. It is
    deliberately structured rather than free-form because:

      1. Inspectability: humans (and metrics) can read what was passed.
      2. Computability: information-theoretic metrics over handoffs require
         a defined source/summary boundary, which the explicit `upstream_refs`
         field provides.
      3. Trust: structured fields can be validated at boundaries; free text
         cannot.

    Each emitting specialist updates only its own portion of the Handoff and
    records what it consumed via `upstream_refs`. This makes each handoff a
    self-describing record of the information transfer that produced it.

    `upstream_refs` maps a downstream field name to the upstream field names
    (or external sources) it was derived from. This is the load-bearing field
    for handoff-fidelity metrics in Layer 2.
    """

    # Goal-and-plan layer
    goal: str | None = None
    subproblem: str | None = None
    constraints: list[str] = Field(default_factory=list)

    # Evidence layer
    evidence: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)

    # Reasoning artifacts
    draft_answer: str | None = None
    critique: str | None = None
    verification: str | None = None
    merged_answer: str | None = None

    # Belief / uncertainty
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    open_questions: list[str] = Field(default_factory=list)

    # Latent belief estimates updated after each agent call. Used by the
    # folded policy graph's signature (Layer 1) and the metric layer (Layer 2).
    belief: Belief = Field(default_factory=Belief)

    # Provenance: which downstream fields derived from which upstream sources.
    # Used by Layer 2 metrics (e.g., handoff fidelity) to compute information
    # loss across the handoff boundary.
    upstream_refs: dict[str, list[str]] = Field(default_factory=dict)

    # Free-form metadata (kept narrow on purpose).
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("upstream_refs")
    @classmethod
    def _refs_keys_are_field_names(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Soft check: keys of upstream_refs should be field names of Handoff.
        We do not enforce strictly because metadata-derived fields may also be
        referenced, but we warn if a key is clearly not a field.
        """
        # Intentionally permissive at v0.1.0; tightened once metric layer lands.
        return v

    model_config = {"validate_assignment": True}
