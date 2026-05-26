"""
Tests for chunk 11.A3 — per-axis rubric scoring.

Covers:
  - `_compose_score_from_axes` math (weighted sum, normalization,
    intersection handling, edge cases).
  - `_aggregate_cross_judge_scores` per-axis fields populated.
  - Per-axis × per-judge disagreement std.
  - Composition asymmetry: some judges return axes, others scalar.
  - Backward compat: judge with no axes → falls back to holistic scalar.
  - Empty axis_weights config → composition disabled, holistic used.
"""

from __future__ import annotations

import pytest

from agensflow.learning.relative_judge import RelativeJudgeConfig, TrajectoryToScore
from agensflow.learning.relative_judge.core import (
    _RelativeJudgement,
    _TrajectoryScore,
    _aggregate_cross_judge_scores,
    _compose_score_from_axes,
)


# --------------------------------------------------------------------------- #
# _compose_score_from_axes — pure math
# --------------------------------------------------------------------------- #


class TestComposeScoreFromAxes:
    def test_weighted_sum_normalized(self) -> None:
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0, "b": 0.0},
            axis_weights={"a": 0.5, "b": 0.5},
        )
        assert out == pytest.approx(0.5)

    def test_unequal_weights(self) -> None:
        # axis a=0.8 (weight 0.7), axis b=0.2 (weight 0.3).
        # weighted_sum = 0.8*0.7 + 0.2*0.3 = 0.62; / 1.0 = 0.62.
        out = _compose_score_from_axes(
            axis_scores={"a": 0.8, "b": 0.2},
            axis_weights={"a": 0.7, "b": 0.3},
        )
        assert out == pytest.approx(0.62)

    def test_normalization_when_weights_dont_sum_to_1(self) -> None:
        # weights are 2 and 1 (not pre-normalized).
        # axes: a=1.0, b=0.0 → weighted = 2.0; / 3.0 = 0.667.
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0, "b": 0.0},
            axis_weights={"a": 2.0, "b": 1.0},
        )
        assert out == pytest.approx(0.6667, rel=0.01)

    def test_intersection_excludes_extra_axes(self) -> None:
        # Judge returned axes {a, b, EXTRA}; config weights {a, b}.
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0, "b": 0.0, "EXTRA": 0.5},
            axis_weights={"a": 0.5, "b": 0.5},
        )
        assert out == pytest.approx(0.5)

    def test_intersection_excludes_missing_axes(self) -> None:
        # Config has weights {a, b, c}; judge returned {a, b}. The
        # composition uses {a, b}, normalized by their weights only —
        # NOT 0 for the missing c (which would yield 0.8).
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0, "b": 1.0},
            axis_weights={"a": 0.4, "b": 0.4, "c": 0.2},
        )
        assert out == pytest.approx(1.0), (
            "Missing axes should not be treated as 0"
        )

    def test_empty_axis_scores_returns_none(self) -> None:
        out = _compose_score_from_axes(
            axis_scores={},
            axis_weights={"a": 1.0},
        )
        assert out is None

    def test_empty_axis_weights_returns_none(self) -> None:
        # Chunk-2..10 reproduction: empty weights → no composition.
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0},
            axis_weights={},
        )
        assert out is None

    def test_no_overlap_returns_none(self) -> None:
        out = _compose_score_from_axes(
            axis_scores={"foo": 1.0, "bar": 0.5},
            axis_weights={"a": 0.5, "b": 0.5},
        )
        assert out is None

    def test_zero_weight_total_returns_none(self) -> None:
        out = _compose_score_from_axes(
            axis_scores={"a": 1.0, "b": 0.5},
            axis_weights={"a": 0.0, "b": 0.0},
        )
        assert out is None


# --------------------------------------------------------------------------- #
# _aggregate_cross_judge_scores — per-axis aggregation
# --------------------------------------------------------------------------- #


def _judgement_with_axes(
    trajectory_id: str,
    holistic: float,
    axes: dict[str, float],
    explanation: str = "ok",
) -> _RelativeJudgement:
    return _RelativeJudgement(scores=[
        _TrajectoryScore(
            trajectory_id=trajectory_id,
            score=holistic,
            explanation=explanation,
            axis_scores=axes,
        ),
    ])


class TestPerAxisAggregation:
    def test_single_judge_with_axes_uses_composed_scalar(self) -> None:
        # Judge returns holistic=0.6 but axes that compose to 0.8.
        # Effective scalar is the composed one — axes are authoritative
        # when present.
        config = RelativeJudgeConfig(
            axis_weights={"goal_achievement": 0.5, "grounding": 0.5},
        )
        per_judge = {"haiku": _judgement_with_axes(
            "t1", holistic=0.6,
            axes={"goal_achievement": 0.9, "grounding": 0.7},
        )}
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        # Composed: 0.9*0.5 + 0.7*0.5 = 0.8.
        assert out["t1"].score == pytest.approx(0.8)
        # Per-judge effective scalar (not holistic).
        assert out["t1"].per_judge_scores["haiku"] == pytest.approx(0.8)
        assert out["t1"].axis_scores == {"goal_achievement": 0.9, "grounding": 0.7}
        assert out["t1"].per_judge_axis_scores == {
            "haiku": {"goal_achievement": 0.9, "grounding": 0.7},
        }

    def test_judge_without_axes_falls_back_to_holistic(self) -> None:
        config = RelativeJudgeConfig(axis_weights={"goal_achievement": 1.0})
        per_judge = {"haiku": _judgement_with_axes(
            "t1", holistic=0.7, axes={},
        )}
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        assert out["t1"].score == pytest.approx(0.7)
        assert out["t1"].axis_scores == {}
        assert out["t1"].per_judge_axis_scores == {"haiku": {}}

    def test_empty_axis_weights_disables_composition(self) -> None:
        # axis_weights={} reproduces chunk-2..10 behavior — even
        # axis-returning judges get the holistic-fallback path.
        config = RelativeJudgeConfig(axis_weights={})
        per_judge = {"haiku": _judgement_with_axes(
            "t1", holistic=0.7,
            axes={"goal_achievement": 0.9, "grounding": 0.5},
        )}
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        assert out["t1"].score == pytest.approx(0.7)
        # Per-axis scores still surfaced for telemetry.
        assert out["t1"].axis_scores == {"goal_achievement": 0.9, "grounding": 0.5}

    def test_per_axis_disagreement_std_across_judges(self) -> None:
        # Three judges:
        #   goal_achievement: 0.9, 0.5, 0.1 → mean 0.5, std large
        #   grounding:        0.8, 0.7, 0.6 → mean 0.7, std small
        # Per-axis std lets us see WHICH axis judges disagreed on.
        config = RelativeJudgeConfig(
            axis_weights={"goal_achievement": 0.5, "grounding": 0.5},
        )
        per_judge = {
            "judge_a": _judgement_with_axes(
                "t1", holistic=0.85,
                axes={"goal_achievement": 0.9, "grounding": 0.8},
            ),
            "judge_b": _judgement_with_axes(
                "t1", holistic=0.6,
                axes={"goal_achievement": 0.5, "grounding": 0.7},
            ),
            "judge_c": _judgement_with_axes(
                "t1", holistic=0.35,
                axes={"goal_achievement": 0.1, "grounding": 0.6},
            ),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        assert out["t1"].axis_scores["goal_achievement"] == pytest.approx(0.5)
        assert out["t1"].axis_scores["grounding"] == pytest.approx(0.7)
        assert out["t1"].per_axis_disagreement_std["goal_achievement"] > \
               out["t1"].per_axis_disagreement_std["grounding"]
        # Per-judge composed: 0.85, 0.60, 0.35 → mean 0.60.
        assert out["t1"].score == pytest.approx(0.60)

    def test_mixed_judges_some_return_axes_others_dont(self) -> None:
        # judge_a returns axes; judge_b returns only holistic.
        config = RelativeJudgeConfig(axis_weights={"goal_achievement": 1.0})
        per_judge = {
            "judge_a": _judgement_with_axes(
                "t1", holistic=0.5, axes={"goal_achievement": 0.9},
            ),  # composed = 0.9
            "judge_b": _judgement_with_axes(
                "t1", holistic=0.7, axes={},
            ),  # falls back to holistic = 0.7
        }
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        # Effective: judge_a=0.9, judge_b=0.7 → mean 0.8.
        assert out["t1"].score == pytest.approx(0.8)
        # Per-axis mean over judges that scored the axis (only judge_a).
        assert out["t1"].axis_scores["goal_achievement"] == pytest.approx(0.9)
        assert out["t1"].per_axis_disagreement_std["goal_achievement"] == 0.0

    def test_judges_score_different_axes_union(self) -> None:
        # judge_a scores goal_achievement; judge_b scores grounding.
        # axis_scores reports BOTH (union); std is 0 for each.
        config = RelativeJudgeConfig(
            axis_weights={"goal_achievement": 0.5, "grounding": 0.5},
        )
        per_judge = {
            "judge_a": _judgement_with_axes(
                "t1", holistic=0.5, axes={"goal_achievement": 0.9},
            ),
            "judge_b": _judgement_with_axes(
                "t1", holistic=0.5, axes={"grounding": 0.7},
            ),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        assert out["t1"].axis_scores == {
            "goal_achievement": 0.9,
            "grounding": 0.7,
        }
        assert out["t1"].per_judge_axis_scores == {
            "judge_a": {"goal_achievement": 0.9},
            "judge_b": {"grounding": 0.7},
        }

    def test_chunk_11_a2_behavior_unchanged_when_no_axes(self) -> None:
        # Regression: when axis_scores are absent (legacy judge),
        # the aggregation produces identical results to chunk 11.A2's
        # mean-of-scalars behavior.
        config = RelativeJudgeConfig(disagreement_confidence_threshold=0.2)
        per_judge = {
            "a": _judgement_with_axes("t1", holistic=0.9, axes={}),
            "b": _judgement_with_axes("t1", holistic=0.7, axes={}),
            "c": _judgement_with_axes("t1", holistic=0.5, axes={}),
        }
        out = _aggregate_cross_judge_scores(
            trajectories=[TrajectoryToScore(trajectory_id="t1", final_answer="a")],
            per_judge_parsed=per_judge,
            config=config,
        )
        assert out["t1"].score == pytest.approx(0.7)
        # std of {0.9, 0.7, 0.5} = sqrt(0.0267) ≈ 0.163.
        assert out["t1"].disagreement_std == pytest.approx(0.163, rel=0.01)
        assert out["t1"].axis_scores == {}
        assert out["t1"].per_axis_disagreement_std == {}


# --------------------------------------------------------------------------- #
# Schema accepts axis_scores (Pydantic boundary)
# --------------------------------------------------------------------------- #


class TestSchemaAcceptsAxisScores:
    def test_pydantic_accepts_axes(self) -> None:
        score = _TrajectoryScore(
            trajectory_id="t1",
            score=0.7,
            explanation="ok",
            axis_scores={"goal_achievement": 0.9, "grounding": 0.5},
        )
        assert score.axis_scores == {"goal_achievement": 0.9, "grounding": 0.5}

    def test_pydantic_axes_default_empty(self) -> None:
        # Backward compat: omitting axis_scores yields {} (no error).
        score = _TrajectoryScore(
            trajectory_id="t1", score=0.7, explanation="ok",
        )
        assert score.axis_scores == {}
