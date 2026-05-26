"""
Activation planner.

Takes task features, runs them through a regime detector, and returns an
ActivationPlan: the coalition of specialists to invoke, the branching rule,
the merge strategy, and the evaluation criteria.

The plan-per-regime mapping is currently hard-coded. This is the right shape
for the open framework: the *structure* of regime-conditioned activation is
the contribution, while the *specific plans* are sensible defaults that users
will customize for their domains.

Layer 2/3 will replace this with a learned policy over (regime, beliefs) ->
plan. The interface stays the same; the caller does not need to know whether
the plan came from a rule table or a trained policy.
"""

from __future__ import annotations

from agensflow.regime.rule_based import detect_regime
from agensflow.schema import (
    ActivationPlan,
    BranchRule,
    RegimeEstimate,
    TaskFeatures,
)


def make_activation_plan(
    features: TaskFeatures,
    regime: RegimeEstimate | None = None,
) -> ActivationPlan:
    """
    Build an activation plan for a task.

    If `regime` is supplied, it is used directly. Otherwise the default
    rule-based regime detector is invoked. Pass an explicit regime when you
    have used a custom or learned detector upstream.
    """
    estimate = regime if regime is not None else detect_regime(features)
    label = estimate.label

    if label == "straightforward":
        return ActivationPlan(
            regime=estimate,
            selected_skills=["planner", "solver", "evaluator"],
            branch_rule=BranchRule(enabled=False, max_branches=1),
            merge_strategy="select_best",
            evaluation_criteria=["task_completion", "coherence"],
        )

    if label == "evidence_heavy":
        return ActivationPlan(
            regime=estimate,
            selected_skills=["planner", "memory", "solver", "verifier", "evaluator"],
            branch_rule=BranchRule(enabled=False, max_branches=1),
            merge_strategy="verifier_gate",
            evaluation_criteria=[
                "evidence_coverage",
                "verification_strength",
                "coherence",
            ],
        )

    if label == "ambiguous":
        return ActivationPlan(
            regime=estimate,
            selected_skills=["planner", "memory"],
            branch_rule=BranchRule(
                enabled=True,
                trigger_if_ambiguity_above=0.6,
                trigger_if_contradiction_above=0.5,
                max_branches=2,
                branch_skill_sets=[
                    ["solver", "critic"],
                    ["solver", "verifier"],
                ],
            ),
            merge_strategy="critic_select",
            evaluation_criteria=[
                "diversity_of_hypotheses",
                "contradiction_resolution",
                "verification_strength",
            ],
        )

    if label == "contradictory":
        return ActivationPlan(
            regime=estimate,
            selected_skills=["planner", "memory", "critic"],
            branch_rule=BranchRule(
                enabled=True,
                trigger_if_ambiguity_above=0.4,
                trigger_if_contradiction_above=0.7,
                max_branches=2,
                branch_skill_sets=[
                    ["solver", "critic"],
                    ["solver", "verifier"],
                ],
            ),
            merge_strategy="verifier_gate",
            evaluation_criteria=[
                "contradiction_risk",
                "verification_strength",
                "uncertainty_reduction",
            ],
        )

    if label == "high_risk":
        return ActivationPlan(
            regime=estimate,
            selected_skills=["planner", "memory", "solver", "verifier"],
            branch_rule=BranchRule(
                enabled=True,
                trigger_if_ambiguity_above=0.5,
                trigger_if_contradiction_above=0.5,
                max_branches=2,
                branch_skill_sets=[
                    ["critic", "verifier"],
                    ["solver", "verifier"],
                ],
            ),
            merge_strategy="verifier_gate",
            evaluation_criteria=[
                "safety",
                "verification_strength",
                "uncertainty_reduction",
            ],
        )

    # exploratory (and any future label) falls through to a sensible default.
    return ActivationPlan(
        regime=estimate,
        selected_skills=["planner", "memory", "solver"],
        branch_rule=BranchRule(enabled=False, max_branches=1),
        merge_strategy="select_best",
        evaluation_criteria=["coherence"],
    )
