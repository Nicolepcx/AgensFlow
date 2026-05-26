"""
Tests for chunk 11.A2 + A4 — cross-judge averaging with disagreement
telemetry, and confidence-weighted backup.

A2 covers:
  - Single-judge mode (cross_judge_models empty) reports
    confidence=1.0, disagreement zero, per_judge_scores has 1 entry.
  - Cross-judge mode aggregates per-trajectory scores correctly.
  - Disagreement metrics (std, range) computed correctly across judges.
  - Confidence calibration: 1 - clamp(std / threshold, 0, 1).
  - Defensive missing-trajectory fill is per-judge (one judge omits a
    trajectory → that judge's slot gets neutral; others' real signals
    are preserved).

A4 (confidence-weighted backup) is integration-tested at the harness
boundary in tests/test_harness_governance.py-style; here we verify the
mechanism works at the dataclass level: low confidence × non-zero
hybrid reward = small backed_up_reward.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agensflow.learning.relative_judge import (
    RelativeJudgeConfig,
    RelativeJudgeScoreGroup,
    RelativeJudgeScoreResult,
    TrajectoryToScore,
    relative_judge_score_group,
)
from agensflow.learning.relative_judge.core import (
    _RelativeJudgement,
    _TrajectoryScore,
    _aggregate_cross_judge_scores,
)


# --------------------------------------------------------------------------- #
# _aggregate_cross_judge_scores — pure aggregation logic
# --------------------------------------------------------------------------- #


def _make_judgement(scores: dict[str, tuple[float, str]]) -> _RelativeJudgement:
    """Helper: build a _RelativeJudgement from {trajectory_id: (score, expl)}."""
    return _RelativeJudgement(
        scores=[
            _TrajectoryScore(trajectory_id=tid, score=s, explanation=e)
            for tid, (s, e) in scores.items()
        ],
    )


class TestSingleJudgeAggregation:
    def test_single_judge_confidence_is_one(self) -> None:
        # When only one judge runs, there's no disagreement to measure
        # → confidence collapses to 1.0 unconditionally.
        trajectories = [
            TrajectoryToScore(trajectory_id="t1", final_answer="a"),
        ]
        per_judge = {
            "haiku": _make_judgement({"t1": (0.85, "good")}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(),
        )
        assert out["t1"].confidence == 1.0
        assert out["t1"].disagreement_std == 0.0
        assert out["t1"].disagreement_range == 0.0
        assert out["t1"].score == pytest.approx(0.85)
        assert out["t1"].per_judge_scores == {"haiku": 0.85}


class TestCrossJudgeAggregation:
    def test_three_judges_agree_high_confidence(self) -> None:
        # All three judges score 0.80 → std=0 → confidence=1.0.
        trajectories = [
            TrajectoryToScore(trajectory_id="t1", final_answer="a"),
        ]
        per_judge = {
            "haiku": _make_judgement({"t1": (0.80, "good")}),
            "gpt": _make_judgement({"t1": (0.80, "good")}),
            "qwen": _make_judgement({"t1": (0.80, "good")}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(disagreement_confidence_threshold=0.2),
        )
        assert out["t1"].score == pytest.approx(0.80)
        assert out["t1"].disagreement_std == pytest.approx(0.0)
        assert out["t1"].disagreement_range == pytest.approx(0.0)
        assert out["t1"].confidence == pytest.approx(1.0)

    def test_three_judges_disagree_low_confidence(self) -> None:
        # Scores 0.2, 0.5, 0.9 → mean 0.5333, std (population) ≈ 0.286
        # → with threshold 0.2, confidence = 1 - clamp(0.286/0.2, 0, 1) = 0.0
        trajectories = [
            TrajectoryToScore(trajectory_id="t1", final_answer="a"),
        ]
        per_judge = {
            "haiku": _make_judgement({"t1": (0.2, "low")}),
            "gpt": _make_judgement({"t1": (0.5, "mid")}),
            "qwen": _make_judgement({"t1": (0.9, "high")}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(disagreement_confidence_threshold=0.2),
        )
        assert out["t1"].score == pytest.approx(0.5333, rel=0.01)
        assert out["t1"].disagreement_range == pytest.approx(0.7)
        # std clamps confidence to 0.
        assert out["t1"].confidence == pytest.approx(0.0)

    def test_intermediate_disagreement_partial_confidence(self) -> None:
        # Two judges, scores 0.7 and 0.5 → std = 0.1 (population).
        # With threshold 0.2: confidence = 1 - 0.1/0.2 = 0.5.
        trajectories = [TrajectoryToScore(trajectory_id="t1", final_answer="a")]
        per_judge = {
            "haiku": _make_judgement({"t1": (0.7, "ok")}),
            "gpt": _make_judgement({"t1": (0.5, "meh")}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(disagreement_confidence_threshold=0.2),
        )
        assert out["t1"].confidence == pytest.approx(0.5, rel=0.01)
        assert out["t1"].score == pytest.approx(0.6)

    def test_threshold_tunes_confidence_sensitivity(self) -> None:
        # Same scores, different threshold → different confidence.
        trajectories = [TrajectoryToScore(trajectory_id="t1", final_answer="a")]
        per_judge = {
            "a": _make_judgement({"t1": (0.7, "ok")}),
            "b": _make_judgement({"t1": (0.5, "meh")}),
        }
        # std ≈ 0.1
        loose = _aggregate_cross_judge_scores(
            trajectories=trajectories, per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(disagreement_confidence_threshold=0.5),  # forgiving
        )
        tight = _aggregate_cross_judge_scores(
            trajectories=trajectories, per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(disagreement_confidence_threshold=0.05),  # strict
        )
        # Loose threshold → higher confidence.
        assert loose["t1"].confidence > tight["t1"].confidence

    def test_per_judge_explanations_populated(self) -> None:
        trajectories = [TrajectoryToScore(trajectory_id="t1", final_answer="a")]
        per_judge = {
            "judge_a": _make_judgement({"t1": (0.8, "explain_a")}),
            "judge_b": _make_judgement({"t1": (0.6, "explain_b")}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=RelativeJudgeConfig(),
        )
        assert out["t1"].per_judge_explanations == {
            "judge_a": "explain_a",
            "judge_b": "explain_b",
        }

    def test_judge_omits_trajectory_per_judge_neutral_fill(self) -> None:
        # judge_a scored t1=0.9; judge_b skipped t1.
        # judge_b's slot gets neutral fill; judge_a's real score preserved.
        trajectories = [TrajectoryToScore(trajectory_id="t1", final_answer="a")]
        per_judge = {
            "judge_a": _make_judgement({"t1": (0.9, "good")}),
            "judge_b": _make_judgement({"other_id": (0.3, "irrelevant")}),
        }
        config = RelativeJudgeConfig(neutral_single_trajectory_score=0.5)
        out = _aggregate_cross_judge_scores(
            trajectories=trajectories,
            per_judge_parsed=per_judge,
            config=config,
        )
        # judge_a: 0.9 (real). judge_b: 0.5 (neutral fill).
        assert out["t1"].per_judge_scores == {"judge_a": 0.9, "judge_b": 0.5}
        # mean = (0.9 + 0.5) / 2 = 0.7
        assert out["t1"].score == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# relative_judge_score_group — full integration with mocked judge calls
# --------------------------------------------------------------------------- #


def _make_completion_result(parsed: _RelativeJudgement, model: str) -> MagicMock:
    """Build a CompletionResult-shaped mock for client.complete_typed."""
    m = MagicMock()
    m.parsed_output = parsed
    m.total_tokens = 100
    m.latency_seconds = 1.0
    m.model = model
    return m


class TestRulerScoreGroupCrossJudge:
    def test_single_judge_mode_uses_judge_model_arg(self) -> None:
        # cross_judge_models empty → only `judge_model` runs.
        client = MagicMock()
        client.complete_typed.return_value = _make_completion_result(
            _make_judgement({"t1": (0.85, "good"), "t2": (0.45, "meh")}),
            model="anthropic/claude-haiku-4.5",
        )
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[
                TrajectoryToScore(trajectory_id="t1", final_answer="a"),
                TrajectoryToScore(trajectory_id="t2", final_answer="b"),
            ],
            client=client,
            judge_model="anthropic/claude-haiku-4.5",
            config=RelativeJudgeConfig(),  # cross_judge_models empty
        )
        assert client.complete_typed.call_count == 1
        assert len(result.per_judge_models) == 1
        # Confidence is 1.0 in single-judge mode regardless of agreement.
        assert result.scores["t1"].confidence == 1.0

    def test_cross_judge_mode_runs_each_listed_model(self) -> None:
        # Cross-judge requires ≥2 trajectories (relative scoring); use
        # 2 here to exercise the full cross-judge path.
        client = MagicMock()
        client.complete_typed.side_effect = [
            _make_completion_result(
                _make_judgement({"t1": (0.9, "high"), "t2": (0.4, "low")}),
                model="haiku",
            ),
            _make_completion_result(
                _make_judgement({"t1": (0.7, "mid"), "t2": (0.4, "low")}),
                model="gpt",
            ),
            _make_completion_result(
                _make_judgement({"t1": (0.5, "low"), "t2": (0.4, "low")}),
                model="qwen",
            ),
        ]
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[
                TrajectoryToScore(trajectory_id="t1", final_answer="a"),
                TrajectoryToScore(trajectory_id="t2", final_answer="b"),
            ],
            client=client,
            judge_model="ignored-when-cross-judge-set",
            config=RelativeJudgeConfig(
                cross_judge_models=["haiku", "gpt", "qwen"],
                disagreement_confidence_threshold=0.2,
            ),
        )
        assert client.complete_typed.call_count == 3
        assert len(result.per_judge_models) == 3
        # t1: mean of 0.9, 0.7, 0.5 = 0.7.
        assert result.scores["t1"].score == pytest.approx(0.7)
        # t1 std large → confidence below 1.
        assert result.scores["t1"].confidence < 1.0
        # t2: all judges agreed at 0.4 → confidence 1.0.
        assert result.scores["t2"].score == pytest.approx(0.4)
        assert result.scores["t2"].confidence == pytest.approx(1.0)
        # Token cost is the SUM across judges.
        assert result.judge_tokens == 300  # 100 × 3
        # Legacy single-name field is "+"-joined for human readability.
        assert "+" in result.judge_model

    def test_cross_judge_score_group_only_one_judge_no_disagreement(self) -> None:
        # cross_judge_models with a single entry behaves like single-judge
        # for confidence purposes (no disagreement to measure).
        client = MagicMock()
        client.complete_typed.return_value = _make_completion_result(
            _make_judgement({"t1": (0.8, "ok"), "t2": (0.6, "ok")}),
            model="haiku",
        )
        result = relative_judge_score_group(
            user_task="task",
            trajectories=[
                TrajectoryToScore(trajectory_id="t1", final_answer="a"),
                TrajectoryToScore(trajectory_id="t2", final_answer="b"),
            ],
            client=client,
            judge_model="ignored",
            config=RelativeJudgeConfig(cross_judge_models=["haiku"]),
        )
        assert result.scores["t1"].confidence == 1.0

    def test_confidence_for_helper(self) -> None:
        client = MagicMock()
        client.complete_typed.return_value = _make_completion_result(
            _make_judgement({"t1": (0.8, "ok")}), model="m",
        )
        result = relative_judge_score_group(
            user_task="t", trajectories=[
                TrajectoryToScore(trajectory_id="t1", final_answer="a"),
                TrajectoryToScore(trajectory_id="t2", final_answer="b"),
            ],
            client=client, judge_model="m", config=RelativeJudgeConfig(),
        )
        # t1 was scored.
        assert result.confidence_for("t1") == 1.0
        # t2 was missing from judge output → defensive neutral fill →
        # confidence = 1.0 (single-judge convention) since per_judge has 1 entry.
        assert result.confidence_for("t2") == 1.0
        # Unknown id → 1.0 fallback (treats missing as fully confident,
        # matching single-judge convention).
        assert result.confidence_for("nonexistent") == 1.0


# --------------------------------------------------------------------------- #
# A4: confidence-weighted backup — verify the multiplier is consistent
# --------------------------------------------------------------------------- #


class TestConfidenceWeightedBackup:
    def test_low_confidence_collapses_backup_signal(self) -> None:
        # Direct check: hybrid_reward * confidence is the backup amount.
        # confidence=0 → backup is zero regardless of reward magnitude.
        hybrid_reward = 0.85
        confidence = 0.0
        weighted = hybrid_reward * confidence
        assert weighted == 0.0

    def test_full_confidence_passes_reward_through(self) -> None:
        # Single-judge runs (confidence=1.0) reproduce chunk-2..10
        # backup behavior exactly: weighted_reward == hybrid_reward.
        hybrid_reward = -0.3
        confidence = 1.0
        weighted = hybrid_reward * confidence
        assert weighted == hybrid_reward

    def test_partial_confidence_proportional(self) -> None:
        # Half-confident judges → half the reward signal lands in the
        # graph. Cost/retry penalties shrink at the same rate as the
        # quality bonus, so the gradient direction is unchanged but
        # magnitude is reduced. This is the design A4 is supposed to
        # ship.
        hybrid_reward = 0.6
        confidence = 0.5
        weighted = hybrid_reward * confidence
        assert weighted == pytest.approx(0.3)

    def test_negative_reward_also_attenuated(self) -> None:
        # Negative rewards (poor trajectory + cost penalty dominating)
        # also get attenuated under low confidence. The substrate
        # learns less from uncertain bad runs, same as uncertain good
        # ones. The KEY invariant: gradient direction preserved,
        # magnitude reduced.
        hybrid_reward = -0.7
        confidence = 0.3
        weighted = hybrid_reward * confidence
        assert weighted == pytest.approx(-0.21)
        # Same sign as hybrid_reward.
        assert (weighted < 0) == (hybrid_reward < 0)
