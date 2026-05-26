"""
Tests for per-agent output schemas.

These tests do NOT call any LLM. They exercise the Pydantic output schemas
directly, demonstrating that:
  - Well-formed outputs validate cleanly.
  - Missing required fields raise ValidationError.
  - Extra fields are silently ignored (forgiving toward the model).
  - Specific value constraints (Literal, min_length) are enforced.

The actual parse-and-validate boundary now lives in Instructor, which is
exercised by the live example. Unit tests here cover the schema contracts
themselves — the part we own — so a contract change can't slip through
without a corresponding test update.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agensflow.runtime.agent_outputs import (
    EvaluatorOutput,
    MemoryOutput,
    PlannerOutput,
    SolverOutput,
    VerifierOutput,
)
from agensflow.runtime.errors import InvalidAgentOutputError


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class TestPlannerOutput:
    def test_well_formed_validates(self) -> None:
        out = PlannerOutput(
            goal="answer the question",
            subproblem="what is X",
            constraints=["be brief"],
        )
        assert out.goal == "answer the question"
        assert out.constraints == ["be brief"]

    def test_missing_subproblem_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput(goal="g")  # type: ignore[call-arg]

    def test_empty_goal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput(goal="", subproblem="s")

    def test_constraints_default_to_empty_list(self) -> None:
        out = PlannerOutput(goal="g", subproblem="s")
        assert out.constraints == []

    def test_extra_fields_ignored(self) -> None:
        out = PlannerOutput.model_validate(
            {"goal": "g", "subproblem": "s", "constraints": [], "thinking": "..."}
        )
        assert out.goal == "g"


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


class TestMemoryOutput:
    def test_well_formed_validates(self) -> None:
        out = MemoryOutput(
            evidence=["fact 1", "fact 2"],
            retrieved_context=["doc1"],
        )
        assert out.evidence == ["fact 1", "fact 2"]

    def test_empty_lists_are_valid(self) -> None:
        out = MemoryOutput(evidence=[], retrieved_context=[])
        assert out.evidence == []

    def test_both_default_to_empty(self) -> None:
        out = MemoryOutput()
        assert out.evidence == []
        assert out.retrieved_context == []


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class TestSolverOutput:
    def test_well_formed_validates(self) -> None:
        out = SolverOutput(draft_answer="the answer is 42")
        assert out.draft_answer == "the answer is 42"

    def test_empty_draft_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput(draft_answer="")

    def test_missing_draft_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SolverOutput()  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


class TestVerifierOutput:
    def test_supported_verdict(self) -> None:
        out = VerifierOutput(
            verdict="supported",
            rationale="all claims grounded",
            uncertain_claims=[],
        )
        assert out.verdict == "supported"

    def test_partially_supported_verdict(self) -> None:
        out = VerifierOutput(
            verdict="partially_supported",
            rationale="ok",
            uncertain_claims=["claim X"],
        )
        assert out.verdict == "partially_supported"

    def test_unknown_verdict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerifierOutput(verdict="maybe", rationale="unsure")  # type: ignore[arg-type]

    def test_empty_rationale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerifierOutput(verdict="supported", rationale="")


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class TestEvaluatorOutput:
    def test_done_true(self) -> None:
        out = EvaluatorOutput(
            done=True,
            final_answer="the answer",
            reasoning="verifier supported",
        )
        assert out.done is True
        assert out.final_answer == "the answer"

    def test_done_false_with_reasoning(self) -> None:
        out = EvaluatorOutput(
            done=False,
            final_answer="partial",
            reasoning="needs more evidence",
        )
        assert out.done is False

    def test_reasoning_defaults_to_empty(self) -> None:
        out = EvaluatorOutput(done=True, final_answer="the answer")
        assert out.reasoning == ""

    def test_empty_final_answer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluatorOutput(done=True, final_answer="")


# --------------------------------------------------------------------------- #
# Error type — kept on the public surface for consumers who catch validation
# failures explicitly.
# --------------------------------------------------------------------------- #


class TestInvalidAgentOutputError:
    def test_carries_agent_name_and_reason(self) -> None:
        err = InvalidAgentOutputError(
            "solver",
            '{"x": 1}',
            "missing field draft_answer",
        )
        assert err.agent_name == "solver"
        assert err.reason == "missing field draft_answer"
        assert err.raw_content == '{"x": 1}'

    def test_truncates_long_raw_content_in_message(self) -> None:
        long = "x" * 1000
        err = InvalidAgentOutputError("solver", long, "bad")
        assert len(str(err)) < 500
