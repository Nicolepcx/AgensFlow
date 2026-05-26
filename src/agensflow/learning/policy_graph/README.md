# `agensflow.learning.policy_graph`

The framework's learning substrate. **Online folded-signature
contextual bandit with annealed UCB action selection and per-edge
reliability tracking.** The policy graph stores
per-`(belief_signature, action)` value estimates which accumulate
across runs; every run contributes its outcome via `backup`, future
runs consult the graph via `best_action`.

The mechanism's lineage is POMCGS-shaped (signatures fold partial
observability; the graph is the value table) but the implementation
runs UCB1 over visited cells rather than full Monte Carlo tree search
with rollouts. We use "POMCGS-inspired" in design notes for that
lineage; the public claim is what's actually implemented:
folded-signature UCB.

## Purpose

Without this module, AgensFlow is a fancy LangGraph wrapper. With it,
coordination becomes **learnable**:

- The graph fold (signature-keyed nodes) is what compresses the
  combinatorial state space into something the substrate can actually
  accumulate evidence over.
- UCB1-based action selection balances exploring under-tried actions
  with exploiting the current best. The selection rule is one-step UCB
  (no rollouts, no learned value estimates, no full tree search) — a
  practical contextual-bandit-shaped policy over the folded signature
  space. The lineage is the same trade-off MCTS makes, applied to
  *which agent to invoke next* rather than which board position to
  evaluate.
- Per-edge failure tallies + reliability weighting (Mechanism A+C) let
  the policy learn that some `(signature, action)` edges are
  unreliable EVEN when their reward looks fine, because a recovered
  validation retry still produces a normal reward.
- Welford running variance per edge (chunk-9) tracks reward stability
  AND token stability — the substrate for "skill-as-constraint" tests
  that need to observe whether a skill spec genuinely bounds output
  length, not just shifts its mean.

The graph is the place where the framework's central claim becomes
real: that coordination over a folded policy graph can be learned
online from trajectory rewards, and the resulting policy generalizes
via signature folding. Stronger search machinery (rollouts, learned
transition/value estimates, discounted TD backup) is a research
extension that this module's clean API leaves room for; chunks 11.C1
ships the discount knob (`gamma`) as a first step.

## Architecture

```
PolicyGraph
  └─ nodes: dict[Signature, GraphNode]
       └─ visits, value_sum
       └─ action_visits, action_value_sums (per (sig, action))
       └─ action_failure_count                (Mechanism A+C)
       └─ action_reward_m2                    (Welford reward variance)
       └─ action_token_sums, action_token_m2  (Welford token variance)
       └─ outgoing: dict[action, Signature]   (multi-graph edges)

  Action selection:
    best_action(sig, legal, c, confidence_threshold, reliability_weight)
      └─ if visits < confidence_threshold: return None  (defer to prior)
      └─ for each legal: ucb_score(action) =
            mean + c_effective·√(ln(N+1)/n) − λ·failure_rate
      └─ return argmax

  Backup:
    backup(path=[(sig, action), ...], reward, action_tokens=None)
      └─ for each (sig, action) on the path:
           visits++; value_sum += reward
           action_visits++; action_value_sums += reward
           Welford: update reward_m2 (always) and token_m2 (when tokens supplied)

  Failure recording:
    record_failure(sig, action)
      └─ action_failure_count[action] += 1
      (called by client hooks when validation retries fire)
```

## Configuration knobs

All knobs live in `PolicyGraphConfig` (see `config.py`); defaults ship
in `agensflow/configs/defaults/policy_graph.yaml`. Override any subset
in your own YAML.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `ucb_c` | 1.4 | UCB1 exploration constant | high-reward-variance workloads — lower to 1.0 for less exploration noise; very low traffic — raise to 2.0 for more aggressive exploration |
| `ucb_c_floor` | 0.5 | min annealed exploration | rarely tune — keeping >0 is what allows recovery from previously-good-now-bad actions |
| `ucb_anneal_half_life` | 50 | visits before `c` halves | high-traffic (1000s of visits) — raise to 200; fast-converging (low-noise rewards) — lower to 20 |
| `confidence_threshold` | 5 | min visits before graph overrides prior | conservative learning — raise to 10; aggressive learning on a known-good prior — lower to 3 |
| `reliability_weight` | 0.5 | λ on failure-rate penalty in UCB | safety-critical (any failure is bad) — raise to 1.0; pure-quality workloads (failures don't matter much) — lower to 0.2 |
| `gamma` | 1.0 | per-edge discount during backup; last edge gets full reward, earlier edges get `reward * gamma^distance_from_terminal` | long paths where early decisions are noise relative to later ones — set 0.9-0.95; keep 1.0 for short paths or genuinely co-causal edges |

## Usage

### Default (no setup):

```python
from agensflow.learning.policy_graph import PolicyGraph
graph = PolicyGraph()
# ... runs accumulate via graph.backup() / graph.record_failure()
action = graph.best_action(signature, legal_actions)  # uses module defaults
```

### Override via YAML:

```yaml
# my-config.yaml
policy_graph:
  confidence_threshold: 10  # be more conservative
  reliability_weight: 1.0   # safety-critical workload
```

```python
from agensflow.config import load_config
from agensflow.learning.policy_graph import PolicyGraph
cfg = load_config("my-config.yaml")
graph = PolicyGraph()
action = graph.best_action(
    signature, legal_actions,
    c=cfg.policy_graph.ucb_c,
    confidence_threshold=cfg.policy_graph.confidence_threshold,
    reliability_weight=cfg.policy_graph.reliability_weight,
)
```

(The router takes the config knobs as arguments — `select_next_action`
threads them through. See `learning/router/README.md`.)

## Design notes

- **`PolicyGraphConfig` is mutable by mechanism, immutable by
  convention.** Same OmegaConf trade-off as every other module
  config — see `web_search/README.md` for the rationale. `GraphNode`
  itself stays a normal dataclass (mutable runtime state, not a
  config schema).

- **Module-level constants kept for backward compat.** Existing code
  that imports `UCB_C`, `DEFAULT_CONFIDENCE_THRESHOLD`, etc. directly
  keeps working — these constants stay populated with the in-code
  defaults. Code that wants YAML-driven hyperparameters goes through
  `agensflow.config.load_config(...).policy_graph` instead.

- **Confidence threshold defers to the prior, not to "do nothing".**
  When `visits < confidence_threshold`, `best_action` returns `None`,
  and the router falls back to the rule-based prior (the activation
  plan's order). This is what makes the graph **strictly improve over
  the prior**: it overrides only when confident; before that, the
  prior runs unchanged.

- **Welford for both reward AND tokens.** The chunk-9 systems claim
  needs variance to be observable at the EDGE level for both signals.
  Welford's online update gives numerically-stable running variance
  with the same data we already pass through `backup` — no extra
  observation cost.

## Caveats

- **Discounted backup is opt-in via `gamma` (chunk 11.C1).** The
  default `gamma=1.0` preserves the chunk-2..10 undiscounted behavior:
  every `(sig, action)` on the path gets the full final reward. Set
  `gamma < 1.0` (typically 0.9-0.95) when you want earlier decisions
  weighted less than later ones — useful on long paths where the
  planner's framing rarely determines whether the verifier passed.
  Token Welford updates are not affected by `gamma` (token counts are
  facts about an action's invocation, not its position on the path).
  Full TD-style backup with learned value estimates is a separate
  research extension.

- **Last-write-wins for outgoing edges.** When the same `(sig, action)`
  leads to different next-signatures across runs, only the most recent
  destination is remembered in `outgoing`. Sufficient for v1; future
  iterations may want a distribution.

- **No pruning.** Low-value or rarely-visited subgraphs accumulate
  forever. Graph size grows with the number of distinct signatures
  observed; `persistence` snapshots get bigger over time. For
  multi-month deployments, an offline prune step would pay for
  itself.

## Tests

`tests/test_policy_graph.py` covers GraphNode invariants (UCB math,
Welford variance, failure-rate calculation), PolicyGraph backup
semantics, best_action confidence-threshold gating, annealed
exploration math.
