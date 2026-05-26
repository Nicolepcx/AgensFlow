# `agensflow.learning.reward`

Reward computation for the policy graph. Two functions:

- **`compute_reward` (v1)** — hand-tuned linear combination of
  evaluator/verifier flags + cost + retries. Hackable
  (chunk-5 finding); kept for ablation studies.
- **`compute_hybrid_reward` (v2, recommended)** — RULER-anchored
  relative quality + operational penalties. Resists the
  internal-flag hacking pattern because the primary signal comes from
  external rubric-based ranking.

Both bounded roughly in `[-1.5, 2.0]` so backed-up values stay
interpretable when divided by visit counts.

## Purpose

Reward is the signal that gets backpropagated through the policy
graph. Higher reward → the `(signature, action)` edges on this run's
path get more credit; future selections at those signatures favor those
actions. The reward function IS the optimization target.

The chunk-5 finding pinned why `compute_hybrid_reward` is the production
recommendation: when the policy sees that `evaluator.done=True` is
worth `+1.0`, and the *internal* evaluator is itself an LLM that the
policy can route to, the policy learns to game the evaluator rather
than improve the answer. RULER's external rubric anchoring breaks that
loop — the judge sees the trajectory's outputs, not the trajectory's
internal state, so the optimization target maps onto actual quality.

## Architecture

```
compute_reward(inputs)                       # v1 baseline
  └─ +success_reward / -failure_penalty   based on inputs.done
  └─ +/- verifier_bonuses                 based on parsed verdict
  └─ - cost_weight * (tokens / cost_normalizer)
  └─ - retry_weight * inputs.n_validation_retries

compute_hybrid_reward(ruler_score, inputs, config)   # v2 production
  └─ + config.ruler_weight * clamp(ruler_score, 0, 1)
  └─ - config.cost_weight * (tokens / config.cost_normalizer)
  └─ - config.retry_weight * inputs.n_validation_retries
```

## Configuration knobs

`RewardConfig` (see `config.py`) — applies to v2 only. `compute_reward`
v1 takes its weights as kwargs (kept as-is for ablation
reproducibility). Defaults ship in
`agensflow/configs/defaults/reward.yaml`.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `ruler_weight` | 1.0 | quality anchor magnitude | research-style workloads where quality matters most — raise to 1.5; cost-sensitive workloads — lower (carefully — see stability rule) |
| `cost_weight` | 0.3 | token-cost penalty magnitude | high-volume customer-support — raise to 0.5; experiments where cost is irrelevant — lower to 0.1 |
| `retry_weight` | 0.15 | validation-retry penalty magnitude | safety-critical (failed attempts are a signal of fragility) — raise to 0.4 |
| `cost_normalizer` | 8000 | tokens that map to max cost penalty | larger cost_normalizer → cost matters less per token; tune to your typical run's token count |
| `enable_stability_warning` | true | whether `ruler_weight < max(cost,retry)` logs a warning | off only when you've intentionally chosen operational-dominant weights |

**Stability rule of thumb:** `ruler_weight >= max(cost_weight,
retry_weight)`. If operational penalties exceed the primary anchor,
the policy may converge to cheap-but-wrong trajectories. The constructor
warns when violated; suppress only deliberately.

## Usage

### Default config:

```python
from agensflow.learning.reward import (
    RewardConfig, RewardInputs, compute_hybrid_reward,
)
inputs = RewardInputs(
    done=True, verification_str=None,
    total_tokens=2400, n_validation_retries=0,
)
reward = compute_hybrid_reward(
    ruler_score=0.85, inputs=inputs, config=RewardConfig(),
)
```

### Override via YAML:

```yaml
# my-config.yaml
reward:
  cost_weight: 0.5      # high-volume workload, cost matters more
  cost_normalizer: 4000 # typical run uses ~4k tokens
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
reward = compute_hybrid_reward(
    ruler_score=ruler_score, inputs=inputs, config=cfg.reward,
)
```

## Design notes

- **`RewardConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module config — see
  `web_search/README.md` for the rationale. Per Nicole's directive,
  this brings RewardConfig into the OmegaConf flow **without changing
  the public API** (`compute_hybrid_reward(*, ruler_score, inputs,
  config=None)` keeps the same signature, with `config=None` defaulting
  to `RewardConfig()` for backward compat).

- **v1 `compute_reward` is intentionally NOT YAML-driven.** It exists
  for ablation reproducibility — a fixed reward function that callers
  can compare against. Burying its kwargs in a YAML would invite
  drift between "the v1 reward I tested last month" and "the v1 reward
  I get today."

- **Reward is bounded but not normalized to `[0, 1]`.** Negative reward
  is meaningful — it tells the policy that this `(signature, action)`
  edge is *worse than nothing*. A clipped-to-zero reward would lose
  that signal.

## Caveats

- **`enable_stability_warning` fires at construction time.** If you
  build many `RewardConfig` instances per run (test code), the warning
  fires for each. Suppress at construction or at the warnings filter
  level if it's noisy.

- **Hackability of v1.** `compute_reward` reads `inputs.done` (set by
  the internal evaluator). The chunk-5 finding showed the policy can
  learn to route in ways that make `evaluator.done=True` even when the
  answer is wrong. v2 fixes this by ignoring `done` and
  `verification_str` entirely — quality comes only from RULER. If you
  use v1, be aware of the failure mode.

- **`ruler_score` is clamped on input.** Defensive: a judge that
  returns 1.05 (out of bounds) gets clamped to 1.0 inside
  `compute_hybrid_reward`. Won't hide the bug from observability
  (the raw judge output is in the trace) but won't propagate
  out-of-range rewards into the substrate.

## Tests

`tests/test_reward.py` covers the v1 reward across all evaluator/
verifier outcomes, the v2 hybrid reward at various RULER scores +
costs + retries, the stability warning, the clamp behavior.
