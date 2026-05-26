"""Tests for activation planning and branch instantiation."""

from __future__ import annotations

from agensflow.activation.branching import instantiate_branches
from agensflow.activation.planner import make_activation_plan
from agensflow.schema import RegimeEstimate, TaskFeatures


class TestStraightforwardPlan:
    def test_minimal_coalition_no_branching(self) -> None:
        plan = make_activation_plan(TaskFeatures())
        assert plan.regime.label == "straightforward"
        assert plan.selected_skills == ["planner", "solver", "evaluator"]
        assert plan.branch_rule.enabled is False
        assert plan.merge_strategy == "select_best"


class TestEvidenceHeavyPlan:
    def test_includes_memory_and_verifier(self) -> None:
        plan = make_activation_plan(
            TaskFeatures(evidence_availability=0.9, verification_need=0.8)
        )
        assert plan.regime.label == "evidence_heavy"
        assert "memory" in plan.selected_skills
        assert "verifier" in plan.selected_skills
        assert plan.merge_strategy == "verifier_gate"
        assert plan.branch_rule.enabled is False


class TestAmbiguousPlan:
    def test_branches_with_critic_and_verifier_alternatives(self) -> None:
        plan = make_activation_plan(TaskFeatures(ambiguity_level=0.9))
        assert plan.regime.label == "ambiguous"
        assert plan.branch_rule.enabled is True
        assert plan.branch_rule.max_branches == 2
        assert plan.merge_strategy == "critic_select"


class TestExplicitRegimeOverride:
    def test_uses_supplied_regime_instead_of_detecting(self) -> None:
        forced = RegimeEstimate(label="high_risk", confidence=0.95)
        plan = make_activation_plan(TaskFeatures(), regime=forced)
        assert plan.regime.label == "high_risk"
        assert "verifier" in plan.selected_skills


class TestInstantiateBranches:
    def test_no_branching_returns_single_sequence(self) -> None:
        plan = make_activation_plan(TaskFeatures())
        branches = instantiate_branches(plan)
        assert len(branches) == 1
        assert branches[0] == ["planner", "solver", "evaluator"]

    def test_branching_returns_multiple_sequences(self) -> None:
        plan = make_activation_plan(TaskFeatures(ambiguity_level=0.9))
        branches = instantiate_branches(plan)
        assert len(branches) == 2
        for branch in branches:
            assert branch[-1] == "evaluator"
            # Base coalition is preserved at the front of every branch.
            assert branch[: len(plan.selected_skills)] == plan.selected_skills

    def test_branches_respect_max_branches(self) -> None:
        plan = make_activation_plan(TaskFeatures(ambiguity_level=0.9))
        # Manually shrink to one
        plan_capped = plan.model_copy(
            update={"branch_rule": plan.branch_rule.model_copy(update={"max_branches": 1})}
        )
        branches = instantiate_branches(plan_capped)
        assert len(branches) == 1
