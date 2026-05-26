"""
PolicyGraphConfig — typed configuration for the learning substrate's
core hyperparameters.

The constants `UCB_C`, `UCB_C_FLOOR`, `UCB_ANNEAL_HALF_LIFE`,
`DEFAULT_CONFIDENCE_THRESHOLD`, `DEFAULT_RELIABILITY_WEIGHT` remain
importable from `core.py` for backward compat. This config dataclass
mirrors them as YAML-overridable knobs so users can tune the
exploration / confidence / reliability balance without forking.

These knobs ARE the learning policy. Tuning them is how to
adapt AgensFlow's substrate to your workload's signal-to-noise ratio.

See `README.md` for per-knob explanations and tuning guidance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyGraphConfig:
    """Hyperparameters for the policy graph's UCB-based action selection.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, never mutate.

    Defaults tuned for chunk-7/8/9 production-traffic experiments.
    Workloads with very different signal/noise should expect to tune.
    """

    # ----- UCB exploration ----- #
    # Initial exploration constant (`c` in UCB1's `mean + c·√(ln(N)/n)`).
    # Higher c → more exploration. 1.4 is the canonical default;
    # workloads with high reward variance may want to lower for less
    # exploration noise.
    ucb_c: float = 1.4

    # Lower bound on the *annealed* exploration constant. Prevents
    # collapse to pure exploitation forever — the policy retains the
    # ability to recover if a previously-good action stops working.
    ucb_c_floor: float = 0.5

    # Visits at which the annealed exploration constant has decayed to
    # half of its initial value. Larger → slower decay → more sustained
    # exploration. 50 visits matches chunk-8 experiment scale; raise
    # for higher-traffic settings where 50 visits accumulates fast.
    ucb_anneal_half_life: int = 50

    # ----- Confidence ----- #
    # Minimum visits to a signature node before the graph's
    # recommendation is trusted over the rule-based prior. Raised from
    # 3 (chunk-4 default) after chunk-5 showed 3 was too eager — UCB
    # started overriding the prior before there was enough data to
    # trust the alternative routes.
    confidence_threshold: int = 5

    # ----- Reliability term in UCB ----- #
    # Weight on the per-(signature, action) failure-rate term in the
    # UCB score: `score = mean + c·exploration − λ·failure_rate`.
    # 0.5 means an action that has failed in 50% of attempts gets a
    # -0.25 penalty against its UCB score — meaningful but not
    # dominant. Raise to make the policy more reliability-sensitive
    # (preferring lower-failure actions even at some cost to mean
    # reward); lower to treat failures as just noise.
    reliability_weight: float = 0.5

    # ----- Backup discount (chunk 11.C1) ----- #
    # Discount factor applied to backed-up reward as a function of an
    # edge's distance from the path's terminal action. The action
    # immediately preceding the outcome (last edge on the path) gets
    # full credit (`reward * gamma^0 = reward`); the first edge gets
    # `reward * gamma^(path_len - 1)`. Earlier decisions thus accumulate
    # less credit per run.
    #
    # Default 1.0 = undiscounted (preserves chunk-2..10 backup
    # semantics; safe default, no behavioral change for existing graphs).
    # Typical discounted setting: 0.9-0.95 for mild trajectory-credit
    # weighting; <0.8 makes the policy aggressively myopic.
    #
    # When tuning: lower gamma helps when early decisions in a long
    # path are noise relative to the later, more-causal ones (e.g. the
    # planner's framing rarely determines whether the verifier passed).
    # Keep gamma=1.0 when paths are short (≤3 edges) or every edge is
    # genuinely co-causal.
    #
    # Token Welford updates are NOT discounted — token counts are
    # facts about an action's invocation, independent of its position
    # on the path. Reward Welford updates ARE discounted (use the
    # edge's discounted reward consistently across mean + variance).
    gamma: float = 1.0
