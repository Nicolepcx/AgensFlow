"""Tests for AgensFlow schema primitives."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agensflow.schema import (
    ActivationPlan,
    BranchRule,
    Handoff,
    RegimeEstimate,
    SkillSpec,
    TaskFeatures,
)


class TestTaskFeatures:
    def test_defaults_are_neutral(self) -> None:
        f = TaskFeatures()
        assert f.requires_factual_grounding is False
        assert f.ambiguity_level == 0.0
        assert f.evidence_availability == 0.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TaskFeatures(ambiguity_level=1.5)
        with pytest.raises(ValidationError):
            TaskFeatures(ambiguity_level=-0.1)

    def test_is_frozen(self) -> None:
        f = TaskFeatures(ambiguity_level=0.5)
        with pytest.raises(ValidationError):
            f.ambiguity_level = 0.7  # type: ignore[misc]


class TestRegimeEstimate:
    def test_defaults(self) -> None:
        e = RegimeEstimate(label="straightforward")
        assert e.confidence == 0.5
        assert e.alternative_labels == []

    def test_rejects_invalid_label(self) -> None:
        with pytest.raises(ValidationError):
            RegimeEstimate(label="not_a_real_regime")  # type: ignore[arg-type]


class TestSkillSpec:
    def test_minimal(self) -> None:
        s = SkillSpec(name="my_agent")
        assert s.kind == "agent"
        assert s.cost_estimate == 1.0
        assert s.merge_preference == "select_best"


class TestBranchRule:
    def test_default_disabled(self) -> None:
        b = BranchRule()
        assert b.enabled is False
        assert b.max_branches == 1

    def test_max_branches_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BranchRule(max_branches=0)


class TestActivationPlan:
    def test_minimal_construction(self) -> None:
        plan = ActivationPlan(
            regime=RegimeEstimate(label="straightforward"),
            selected_skills=["planner", "solver"],
        )
        assert plan.merge_strategy == "select_best"
        assert plan.branch_rule.enabled is False


class TestHandoff:
    def test_empty_construction(self) -> None:
        h = Handoff()
        assert h.goal is None
        assert h.evidence == []
        assert h.upstream_refs == {}

    def test_uncertainty_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Handoff(uncertainty=1.5)

    def test_upstream_refs_records_provenance(self) -> None:
        h = Handoff(
            draft_answer="some answer",
            upstream_refs={"draft_answer": ["subproblem", "evidence"]},
        )
        assert h.upstream_refs["draft_answer"] == ["subproblem", "evidence"]

    def test_assignment_validation_active(self) -> None:
        h = Handoff()
        with pytest.raises(ValidationError):
            h.uncertainty = 1.5  # type: ignore[assignment]
