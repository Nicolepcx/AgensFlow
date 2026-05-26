"""
Tests for RULER-style relative scoring.

These tests mock the LLM judge so they're fast, deterministic, and don't burn
API budget. The mock pattern is the same one used in `test_client_hooks.py`:
construct a real openai.OpenAI client with a fake key, monkey-patch its
`chat.completions.create` to return pre-canned responses, then wrap with
Instructor before letting the OpenRouterClient touch it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agensflow.learning.relative_judge import (
    DEFAULT_RUBRIC,
    RelativeJudgeScoreGroup,
    TrajectoryToScore,
    relative_judge_score_group,
)
from agensflow.runtime.client import OpenRouterClient


# --------------------------------------------------------------------------- #
# Tool-call mock helpers (same shape as test_client_hooks.py)
# --------------------------------------------------------------------------- #


def _make_tool_call_message(
    arguments_json: str, tool_name: str = "_StrictRelativeJudgement",
) -> SimpleNamespace:
    """Build a mock tool-call message. Default `tool_name` matches the
    chunk-11.A3-default Pydantic class (`_StrictRelativeJudgement`, fired
    when `RelativeJudgeConfig.axis_weights` is non-empty — the default).
    Pass `tool_name="_RelativeJudgement"` explicitly for tests that opt
    into the relaxed schema (axis_weights={})."""
    return SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_ruler",
            type="function",
            function=SimpleNamespace(
                name=tool_name,
                arguments=arguments_json,
            ),
        )],
        refusal=None,
    )


def _make_completion(
    arguments_json: str, p: int = 1500, c: int = 200,
    tool_name: str = "_StrictRelativeJudgement",
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-mock",
        model="anthropic/claude-haiku-4.5-mock",
        choices=[SimpleNamespace(
            index=0,
            message=_make_tool_call_message(arguments_json, tool_name),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c),
    )


def _traj_entry(
    trajectory_id: str, score: float, explanation: str,
    axis_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a mock judge-output entry. Auto-populates axis_scores to
    match the holistic score so the strict-schema (chunk 11.A3 default)
    accepts the mock; tests not asserting per-axis behavior get
    sensible defaults. Pass `axis_scores=` explicitly to test cases
    that DO care about per-axis values."""
    if axis_scores is None:
        axis_scores = {
            "goal_achievement": score,
            "grounding": score,
            "coordination": score,
            "recovery": score,
        }
    return {
        "trajectory_id": trajectory_id,
        "score": score,
        "explanation": explanation,
        "axis_scores": axis_scores,
    }


def _build_client_with_responses(responses: list[Any]) -> OpenRouterClient:
    import instructor
    import openai
    from instructor import Mode

    real_openai = openai.OpenAI(api_key="fake-test-key")
    queue = list(responses)

    def fake_create(**kwargs: Any) -> Any:
        if not queue:
            raise RuntimeError("Mock transport exhausted")
        return queue.pop(0)

    real_openai.chat.completions.create = fake_create  # type: ignore[method-assign]

    client = OpenRouterClient.__new__(OpenRouterClient)
    client._raw = real_openai  # type: ignore[attr-defined]
    client._default_mode = Mode.TOOLS  # type: ignore[attr-defined]
    client._instructor = instructor.from_openai(real_openai, mode=Mode.TOOLS)
    # Chunk 11.A2: per-call mode selection requires both wrappers.
    client._instructor_json = instructor.from_openai(real_openai, mode=Mode.JSON)  # type: ignore[attr-defined]
    client._register_hooks()
    return client


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestRulerScoreGroup:
    def test_empty_group_returns_empty(self) -> None:
        client = _build_client_with_responses([])
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[],
            client=client,
        )
        assert result.scores == {}

    def test_single_trajectory_gets_neutral_score_no_judge_call(self) -> None:
        # No mock responses queued — confirms no LLM call is made.
        client = _build_client_with_responses([])
        traj = TrajectoryToScore(trajectory_id="solo", final_answer="x")
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[traj],
            client=client,
        )
        assert "solo" in result.scores
        assert result.scores["solo"].score == 0.5
        assert "Single-trajectory" in result.scores["solo"].explanation

    def test_three_trajectories_scored_relatively(self) -> None:
        judge_response = json.dumps({
            "scores": [
                _traj_entry("good", 0.9, "best one"),
                _traj_entry("ok", 0.5, "middling"),
                _traj_entry("bad", 0.1, "worst"),
            ]
        })
        client = _build_client_with_responses([_make_completion(judge_response)])

        trajectories = [
            TrajectoryToScore(trajectory_id="good", final_answer="thorough answer"),
            TrajectoryToScore(trajectory_id="ok", final_answer="ok answer"),
            TrajectoryToScore(trajectory_id="bad", final_answer="wrong"),
        ]
        result = relative_judge_score_group(
            user_task="explain X",
            trajectories=trajectories,
            client=client,
        )

        assert isinstance(result, RelativeJudgeScoreGroup)
        # Composed-from-axes can introduce tiny float-arithmetic noise
        # (chunk 11.A3): 0.9*(0.3+0.3+0.2+0.2)/1.0 ≈ 0.9 + 1e-16. Use
        # approx; the underlying mechanic is "all axes 0.9 → composed 0.9".
        assert result.score_for("good") == pytest.approx(0.9)
        assert result.score_for("ok") == 0.5
        assert result.score_for("bad") == 0.1
        assert result.judge_tokens > 0  # mock returned usage

    def test_judge_omits_trajectory_filled_with_neutral(self) -> None:
        # Judge returns scores for only 2 of 3 trajectories.
        judge_response = json.dumps({
            "scores": [
                _traj_entry("a", 0.8, "good"),
                _traj_entry("b", 0.3, "weak"),
                # 'c' missing
            ]
        })
        client = _build_client_with_responses([_make_completion(judge_response)])

        trajectories = [
            TrajectoryToScore(trajectory_id="a", final_answer="A"),
            TrajectoryToScore(trajectory_id="b", final_answer="B"),
            TrajectoryToScore(trajectory_id="c", final_answer="C"),
        ]
        result = relative_judge_score_group(
            user_task="task",
            trajectories=trajectories,
            client=client,
        )

        assert result.score_for("a") == 0.8
        assert result.score_for("b") == 0.3
        # Missing trajectory got the neutral fallback.
        assert result.score_for("c") == 0.5
        assert "omitted" in result.scores["c"].explanation

    def test_score_for_missing_trajectory_returns_zero(self) -> None:
        judge_response = json.dumps({
            "scores": [
                _traj_entry("a", 0.7, "ok"),
            ]
        })
        client = _build_client_with_responses([_make_completion(judge_response)])
        trajectories = [TrajectoryToScore(trajectory_id="a", final_answer="A"),
                        TrajectoryToScore(trajectory_id="b", final_answer="B")]
        result = relative_judge_score_group(
            user_task="task",
            trajectories=trajectories,
            client=client,
        )
        # Score lookup of an id the result didn't fill returns 0.0.
        assert result.score_for("nonexistent") == 0.0

    def test_path_summary_included_in_judge_prompt(self) -> None:
        # We can't easily inspect the prompt the judge saw without intercepting,
        # but we can verify that path_summary on a trajectory doesn't crash.
        judge_response = json.dumps({
            "scores": [
                _traj_entry("x", 0.7, "ok"),
                _traj_entry("y", 0.4, "weaker"),
            ]
        })
        client = _build_client_with_responses([_make_completion(judge_response)])
        trajectories = [
            TrajectoryToScore(
                trajectory_id="x",
                final_answer="answer",
                path_summary="planner -> memory -> solver_capable -> verifier -> evaluator",
            ),
            TrajectoryToScore(
                trajectory_id="y",
                final_answer="answer",
                path_summary="planner -> solver_fast -> evaluator",
            ),
        ]
        result = relative_judge_score_group(
            user_task="task",
            trajectories=trajectories,
            client=client,
        )
        assert result.score_for("x") == 0.7
        assert result.score_for("y") == 0.4

    def test_custom_rubric_does_not_crash(self) -> None:
        custom_rubric = "Score harshly. Best gets 0.6 not 1.0."
        judge_response = json.dumps({
            "scores": [
                _traj_entry("a", 0.55, "harsh"),
                _traj_entry("b", 0.20, "harsher"),
            ]
        })
        client = _build_client_with_responses([_make_completion(judge_response)])
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[
                TrajectoryToScore(trajectory_id="a", final_answer="A"),
                TrajectoryToScore(trajectory_id="b", final_answer="B"),
            ],
            client=client,
            rubric=custom_rubric,
        )
        assert result.score_for("a") == 0.55


class TestDefaultRubric:
    def test_default_rubric_mentions_load_bearing_axes(self) -> None:
        # Sanity: the default rubric includes the four machine-friendly
        # axis names (matching `RelativeJudgeConfig.axis_weights` keys).
        # Chunk 11.A3 switched the rubric from prose names ("Goal
        # achievement") to machine-friendly identifiers
        # ("goal_achievement") so the judge populates `axis_scores`
        # under those keys.
        for axis in ["goal_achievement", "grounding",
                     "coordination", "recovery"]:
            assert axis in DEFAULT_RUBRIC


class TestRulerScoreInvalidRange:
    def test_judge_score_out_of_range_rejected_at_validation(self) -> None:
        # The Pydantic schema enforces [0, 1]. A judge returning >1 should
        # trigger the standard validation-retry behavior of the client.
        # We simulate that by returning a bad response, then a good one.
        # axis_scores included in both so we test the score-range check
        # specifically (chunk 11.A3 strict schema also requires axes).
        bad = json.dumps({"scores": [_traj_entry("a", 1.5, "out of range")]})
        good = json.dumps({"scores": [_traj_entry("a", 0.7, "valid")]})
        client = _build_client_with_responses([
            _make_completion(bad),
            _make_completion(good),
        ])
        trajectories = [TrajectoryToScore(trajectory_id="a", final_answer="A")]

        # With only one trajectory the function short-circuits before calling
        # the judge, so we need at least two for this test.
        # Adjust: send two trajectories, both with the same id... no, ids must
        # be distinct. Just verify the validation path triggers a retry.
        client = _build_client_with_responses([
            _make_completion(bad),
            _make_completion(good),
        ])
        # Use two trajectories; second response only fills 'a' so 'b' gets the
        # neutral fallback. The point is the bad first response triggers retry.
        trajectories = [
            TrajectoryToScore(trajectory_id="a", final_answer="A"),
            TrajectoryToScore(trajectory_id="b", final_answer="B"),
        ]
        # The judge's first response would fail validation (score=1.5).
        # The corrective retry should land on the good response.
        # If the validation fails twice, we'd raise InvalidAgentOutputError;
        # here the second attempt is valid.
        # But actually the second response only has one score, so 'b' gets neutral.
        result = relative_judge_score_group(
            user_task="task",
            trajectories=trajectories,
            client=client,
        )
        # The retry mechanism handled the bad first response.
        assert result.score_for("a") == 0.7
        assert result.score_for("b") == 0.5  # neutral fallback for missing
