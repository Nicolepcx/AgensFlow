"""
Reward computation for the policy graph.

The reward is the signal that gets backpropagated through the policy graph.
Higher reward → the (signature, action) edges on this run's path get more
credit; future selections at those signatures will favor those actions.

Two reward functions live here:

  - `compute_reward` (chunk 4 v1): a hand-tuned linear combination of the
    evaluator's done flag, the verifier's verdict, token cost, and
    validation retry count. Useful as a baseline and as a fast-path for
    tests that don't want LLM judge cost. **Hackable** — chunk 5 showed the
    policy can game the internal evaluator/verifier flags. Kept for ablation
    studies but not the recommended reward for production training.

  - `compute_hybrid_reward` (chunk 6 v2): the recommended reward. Combines
    a RelativeJudge-anchored relative-quality score with operational penalties
    (cost, retries) under configurable hyperparameter weights. Resists
    the chunk-5 hacking pattern because the primary signal comes from
    external rubric-based ranking rather than internal flags.

Both are bounded roughly in [-1.5, 2.0] so backed-up values stay
interpretable when divided by visit counts.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

# We can't import RunResult here without a circular import (runtime imports
# learning, learning imports runtime). The reward function takes the
# specific fields it needs; the runner extracts them.


@dataclass(frozen=True)
class RewardInputs:
    """
    Minimal subset of a RunResult that the reward function needs.

    Decoupled from RunResult to avoid a runtime->learning circular import
    and to make the reward function easy to test with synthetic inputs.
    """

    done: bool
    verification_str: str | None
    total_tokens: int
    n_validation_retries: int


def compute_reward(
    inputs: RewardInputs,
    *,
    cost_normalizer: int = 8000,
    cost_weight: float = 0.4,
    retry_weight: float = 0.15,
    success_reward: float = 1.0,
    failure_penalty: float = 0.3,
    verifier_supported_bonus: float = 0.5,
    verifier_partial_bonus: float = 0.1,
    verifier_unsupported_penalty: float = 0.4,
) -> float:
    """
    Composite reward in roughly [-1.5, 2.0].

    cost_normalizer: tokens that map to the maximum cost penalty. Larger
    cost_normalizer → cost matters less per token. Default tuned for the
    chunk-3 benchmark scale.
    """
    reward = 0.0

    # Outcome
    if inputs.done:
        reward += success_reward
    else:
        reward -= failure_penalty

    # Verifier verdict
    verdict = _extract_verifier_verdict(inputs.verification_str)
    if verdict == "supported":
        reward += verifier_supported_bonus
    elif verdict == "partially_supported":
        reward += verifier_partial_bonus
    elif verdict == "unsupported":
        reward -= verifier_unsupported_penalty

    # Cost penalty (normalized, weighted)
    cost_fraction = min(inputs.total_tokens / max(cost_normalizer, 1), 1.0)
    reward -= cost_fraction * cost_weight

    # Retry penalty (each retry costs)
    reward -= inputs.n_validation_retries * retry_weight

    return reward


def _extract_verifier_verdict(verification_str: str | None) -> str | None:
    if not verification_str:
        return None
    try:
        parsed = json.loads(verification_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    return verdict if isinstance(verdict, str) else None


# --------------------------------------------------------------------------- #
# Hybrid reward (chunk 6 v2): RelativeJudge + operational penalties.
# --------------------------------------------------------------------------- #


@dataclass
class RewardConfig:
    """
    Hyperparameter weights for the hybrid reward function.

    The hybrid reward combines:
      - **ruler_weight × RelativeJudge score**: external rubric-based ranking, [0, 1].
        This is the primary anchor — it's what makes the reward resistant to
        the internal-state hacking pattern observed in chunk 5.
      - **−cost_weight × cost_normalized**: operational efficiency penalty.
        Drives the policy toward fewer-token routing decisions.
      - **−retry_weight × n_validation_retries**: clean-execution penalty.
        Drives the policy toward orchestration paths that don't require
        corrective retries to land.

    Defaults are sensible starting points for production-shape MAS workloads.
    Customers should expect to tune these per workload — different teams care
    about different axes (deep-research workloads weight RelativeJudge more, high-
    volume customer-support weights cost more, safety-critical weights
    retries more).

    Stability constraint (informal):
        keep ruler_weight >= max(cost_weight, retry_weight).
    If operational penalties exceed the primary anchor, the policy may
    converge to cheap-but-wrong trajectories. The framework logs a warning
    on construction if this constraint is violated; pass
    `enable_stability_warning=False` to suppress.

    Note: not `frozen=True` because OmegaConf's structured-config merge
    requires mutable nested dataclasses. Treat as immutable by
    convention: construct once at startup via `agensflow.config.load_config`
    and pass into `compute_hybrid_reward(config=...)`. See
    `agensflow/learning/reward/README.md` design notes for the full
    rationale (frozen-by-convention applies to every YAML-merged
    config in the framework).
    """

    ruler_weight: float = 1.0
    cost_weight: float = 0.3
    retry_weight: float = 0.15
    cost_normalizer: int = 8000
    enable_stability_warning: bool = True

    def __post_init__(self) -> None:
        if not self.enable_stability_warning:
            return
        max_op = max(self.cost_weight, self.retry_weight)
        if self.ruler_weight < max_op:
            warnings.warn(
                f"RewardConfig stability concern: ruler_weight="
                f"{self.ruler_weight} is less than max(cost_weight={self.cost_weight}, "
                f"retry_weight={self.retry_weight})={max_op}. Operational penalties "
                f"may dominate the rubric-anchored signal, which can converge the "
                f"policy to low-quality trajectories. Set "
                f"enable_stability_warning=False to suppress this warning if "
                f"this is intentional.",
                stacklevel=3,
            )


def compute_hybrid_reward(
    *,
    ruler_score: float,
    inputs: RewardInputs,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute the hybrid reward for a single run.

    Arguments:
      ruler_score: the trajectory's RelativeJudge score, [0, 1]. Typically obtained
        by passing this run's trajectory to `ruler.relative_judge_score_group`
        alongside other trajectories from the same task.
      inputs: existing RewardInputs (cost, retry count). The `done` and
        `verification_str` fields on RewardInputs are *ignored* by the
        hybrid reward — those were the hackable pieces in v1. The hybrid
        reward gets its quality signal from the RelativeJudge score, not from
        internal flags.
      config: hyperparameter weights. Defaults to RewardConfig() if None.

    Returns: a scalar reward, roughly in [-1.5, 1.0]:
      - +1.0 corresponds to a top-rubric trajectory at zero cost and zero
        retries.
      - 0.0 corresponds to a middling trajectory at moderate cost.
      - Negative values are possible when cost + retry penalties exceed
        the RelativeJudge bonus (e.g. a poor trajectory that also burned tokens
        and triggered retries).
    """
    cfg = config if config is not None else RewardConfig()

    # Clamp the RelativeJudge score defensively in case the judge returned a value
    # slightly outside [0, 1].
    ruler_clamped = max(0.0, min(1.0, ruler_score))

    cost_term = min(inputs.total_tokens / max(cfg.cost_normalizer, 1), 1.0)

    return (
        cfg.ruler_weight * ruler_clamped
        - cfg.cost_weight * cost_term
        - cfg.retry_weight * inputs.n_validation_retries
    )
