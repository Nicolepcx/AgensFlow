"""
Tests for the hybrid reward function and RewardConfig.

These exercise the reward arithmetic, the stability warnings, and the
behavior under edge cases (extreme weights, out-of-range RULER scores,
zero-cost trajectories). No LLM calls.
"""

from __future__ import annotations

import warnings

import pytest

from agensflow.learning.reward import (
    RewardConfig,
    RewardInputs,
    compute_hybrid_reward,
)


# --------------------------------------------------------------------------- #
# RewardConfig
# --------------------------------------------------------------------------- #


class TestRewardConfigDefaults:
    def test_default_weights(self) -> None:
        cfg = RewardConfig()
        assert cfg.ruler_weight == 1.0
        assert cfg.cost_weight == 0.3
        assert cfg.retry_weight == 0.15
        assert cfg.cost_normalizer == 8000

    def test_default_satisfies_stability_constraint(self) -> None:
        # The defaults should not trigger the stability warning.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardConfig()
        # Filter for the specific stability warning (other warnings might leak).
        stability_warnings = [
            x for x in w if "stability concern" in str(x.message)
        ]
        assert len(stability_warnings) == 0


class TestStabilityWarning:
    def test_warns_when_cost_exceeds_ruler(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardConfig(ruler_weight=0.2, cost_weight=0.5)
        stability_warnings = [
            x for x in w if "stability concern" in str(x.message)
        ]
        assert len(stability_warnings) == 1

    def test_warns_when_retry_exceeds_ruler(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardConfig(ruler_weight=0.1, cost_weight=0.05, retry_weight=0.5)
        stability_warnings = [
            x for x in w if "stability concern" in str(x.message)
        ]
        assert len(stability_warnings) == 1

    def test_warning_can_be_suppressed(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RewardConfig(
                ruler_weight=0.1,
                cost_weight=0.5,
                enable_stability_warning=False,
            )
        stability_warnings = [
            x for x in w if "stability concern" in str(x.message)
        ]
        assert len(stability_warnings) == 0


# --------------------------------------------------------------------------- #
# Hybrid reward arithmetic
# --------------------------------------------------------------------------- #


def _inputs(tokens: int = 4000, retries: int = 0) -> RewardInputs:
    """Convenience: create RewardInputs with the hackable v1 fields zeroed."""
    return RewardInputs(
        done=True,                # ignored by hybrid
        verification_str=None,    # ignored by hybrid
        total_tokens=tokens,
        n_validation_retries=retries,
    )


class TestHybridRewardArithmetic:
    def test_perfect_trajectory_zero_cost(self) -> None:
        # Perfect ruler, 0 tokens, 0 retries → reward = ruler_weight * 1.0
        r = compute_hybrid_reward(
            ruler_score=1.0,
            inputs=_inputs(tokens=0, retries=0),
        )
        assert r == pytest.approx(1.0)

    def test_zero_quality_zero_cost(self) -> None:
        # ruler=0, no cost, no retries → reward = 0
        r = compute_hybrid_reward(
            ruler_score=0.0,
            inputs=_inputs(tokens=0, retries=0),
        )
        assert r == pytest.approx(0.0)

    def test_cost_normalizer_caps_penalty(self) -> None:
        # Tokens beyond cost_normalizer don't add more penalty.
        cfg = RewardConfig()
        r1 = compute_hybrid_reward(
            ruler_score=0.5,
            inputs=_inputs(tokens=cfg.cost_normalizer),
            config=cfg,
        )
        r2 = compute_hybrid_reward(
            ruler_score=0.5,
            inputs=_inputs(tokens=cfg.cost_normalizer * 5),
            config=cfg,
        )
        # Both saturate at cost_normalizer → identical reward.
        assert r1 == pytest.approx(r2)

    def test_higher_ruler_higher_reward(self) -> None:
        ins = _inputs(tokens=4000)
        r_low = compute_hybrid_reward(ruler_score=0.2, inputs=ins)
        r_high = compute_hybrid_reward(ruler_score=0.8, inputs=ins)
        assert r_high > r_low

    def test_higher_cost_lower_reward(self) -> None:
        r_cheap = compute_hybrid_reward(
            ruler_score=0.7, inputs=_inputs(tokens=1000)
        )
        r_expensive = compute_hybrid_reward(
            ruler_score=0.7, inputs=_inputs(tokens=8000)
        )
        assert r_cheap > r_expensive

    def test_retries_lower_reward(self) -> None:
        r_clean = compute_hybrid_reward(
            ruler_score=0.7, inputs=_inputs(retries=0)
        )
        r_retried = compute_hybrid_reward(
            ruler_score=0.7, inputs=_inputs(retries=2)
        )
        assert r_clean > r_retried

    def test_out_of_range_ruler_clamped(self) -> None:
        # Defensive against a judge returning slightly out-of-range values.
        r_clamped_high = compute_hybrid_reward(
            ruler_score=1.5, inputs=_inputs(tokens=0)
        )
        r_max = compute_hybrid_reward(
            ruler_score=1.0, inputs=_inputs(tokens=0)
        )
        assert r_clamped_high == pytest.approx(r_max)

        r_clamped_low = compute_hybrid_reward(
            ruler_score=-0.5, inputs=_inputs(tokens=0)
        )
        r_min = compute_hybrid_reward(
            ruler_score=0.0, inputs=_inputs(tokens=0)
        )
        assert r_clamped_low == pytest.approx(r_min)


class TestHybridRewardConfigSensitivity:
    def test_larger_cost_weight_amplifies_cost_penalty(self) -> None:
        ins = _inputs(tokens=4000)
        cheap = RewardConfig(cost_weight=0.1)
        expensive = RewardConfig(cost_weight=0.6)
        r_cheap = compute_hybrid_reward(ruler_score=0.7, inputs=ins, config=cheap)
        r_expensive = compute_hybrid_reward(ruler_score=0.7, inputs=ins, config=expensive)
        # Same ruler score and tokens, but the higher cost_weight produces
        # a smaller (lower) reward.
        assert r_cheap > r_expensive

    def test_ruler_dominant_when_weights_default(self) -> None:
        # With the defaults (ruler=1.0, cost=0.3, retry=0.15), a high RULER
        # score should lift the reward above 0 even with maximal cost penalty.
        cfg = RewardConfig()
        r = compute_hybrid_reward(
            ruler_score=1.0,
            inputs=_inputs(tokens=cfg.cost_normalizer * 5, retries=0),
            config=cfg,
        )
        # 1.0 - 0.3 = 0.7
        assert r == pytest.approx(0.7)

    def test_no_internal_state_dependence(self) -> None:
        # Hybrid reward does NOT depend on done/verification_str.
        # Two RewardInputs differing only on those should produce identical
        # rewards — which is the whole point: those were the hackable fields
        # in v1.
        a = RewardInputs(
            done=True, verification_str='{"verdict":"supported"}',
            total_tokens=2000, n_validation_retries=0,
        )
        b = RewardInputs(
            done=False, verification_str=None,
            total_tokens=2000, n_validation_retries=0,
        )
        ra = compute_hybrid_reward(ruler_score=0.6, inputs=a)
        rb = compute_hybrid_reward(ruler_score=0.6, inputs=b)
        assert ra == pytest.approx(rb)
