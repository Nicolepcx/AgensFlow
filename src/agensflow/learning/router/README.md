# `agensflow.learning.router`

The routing decision function — Layer 1's "real feature." Where the
policy-learning substrate becomes actionable.

## Purpose

`select_next_action` is a **pure function**: state in →
`(action_or_None, reason)` out. No LLM calls, no I/O. This is what
makes routing testable with synthetic graphs + states, and what makes
the routing logic re-usable in any context where "what should the next
action be?" needs answering (LangGraph runtime today, programmatic
planning tomorrow, MCTS rollouts later).

The decision flow:

1. **Budget check** → `("budget_exhausted")` if too many actions taken.
2. **Evaluator-done check** → `("evaluator_done")` if metadata says done.
3. **Legal-action computation** → `("no_legal_actions")` if none.
4. **Policy graph consultation** → `("graph_recommendation")` if the
   node at the current signature has ≥`confidence_threshold` visits.
5. **Rule-based prior fallback** → `("rule_based_prior")` — the first
   legal action in the activation plan's order.

The router never hides decisions: every output carries a reason string
that explains why this action (or termination) was picked.

## Architecture

```
select_next_action(current_state, plan, policy_graph, actions_taken,
                   max_steps, exploration_c, confidence_threshold,
                   reliability_weight, enable_skip)
  └─ if budget exhausted: return (None, "budget_exhausted")
  └─ if evaluator_done(state): return (None, "evaluator_done")
  └─ legal = _legal_actions(plan, state, actions_taken)
  └─ if not legal: return (None, "no_legal_actions")
  └─ signature = belief_signature(state, regime_label)
  └─ candidates = legal + (skip:X for X in legal if enable_skip)
  └─ recommended = policy_graph.best_action(signature, candidates, ...)
       └─ defers if visits < confidence_threshold
  └─ if recommended: return (recommended, "graph_recommendation")
  └─ return (legal[0], "rule_based_prior")  # the prior never picks skip
```

## Configuration knobs

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `max_steps` | 12 | budget for agent invocations per run | research workloads with many sub-questions — raise to 20+ |
| `enable_skip` | false | whether `skip:X` is a candidate action | chunk-8/9-style experiments learning coordination *choice* — set true |
| `enable_router_logging` | false | per-iteration forensic logging to `trace.router_log` | debugging a runaway loop — set true |

The UCB/confidence/reliability knobs (`exploration_c`,
`confidence_threshold`, `reliability_weight`) come from
`PolicyGraphConfig` so they stay aligned with the substrate's defaults.
You can override them per-call as `select_next_action` kwargs if a
specific routing decision needs different settings, but the library
default is to read from `cfg.policy_graph`.

Defaults ship in `agensflow/configs/defaults/router.yaml`.

## Usage

### Default (matches chunk-2/3 contract):

```python
from agensflow.learning.router import select_next_action
decision = select_next_action(
    current_state=handoff,
    plan=plan,
    policy_graph=graph,
    actions_taken=actions_taken,
)
print(decision.action, decision.reason)
```

### YAML-driven router setup (chunks 8/9):

```yaml
# my-config.yaml
router:
  enable_skip: true
  enable_router_logging: true   # for debugging
policy_graph:
  confidence_threshold: 10
  reliability_weight: 1.0
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
decision = select_next_action(
    current_state=handoff, plan=plan, policy_graph=graph,
    actions_taken=actions_taken,
    max_steps=cfg.router.max_steps,
    confidence_threshold=cfg.policy_graph.confidence_threshold,
    reliability_weight=cfg.policy_graph.reliability_weight,
    enable_skip=cfg.router.enable_skip,
)
```

(In practice, the graph builder
`agensflow.runtime.graph.build_learning_graph` threads the same
config-driven kwargs into the inner `select_next_action` calls — see
`runtime/graph/README.md`.)

## Design notes

- **`RouterConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module config — see
  `web_search/README.md` for the rationale.

- **Two configs split by concern.** UCB/confidence/reliability are
  *substrate* knobs (PolicyGraphConfig); max_steps/enable_skip/logging
  are *routing* knobs (RouterConfig). Both are accepted as per-call
  kwargs of `select_next_action` so callers can override either
  independently.

- **The skip-offer is gated by alternatives.** `skip:X` candidates are
  only added when `len(legal) > 1` — a single-legal-action state
  can't be skipped or we'd terminate productive runs. Termination of
  otherwise-progressing runs is the job of `evaluator_done` /
  `budget_exhausted` / `no_legal_actions`, NOT skip.

- **The rule-based prior never picks `skip:X`.** When the graph isn't
  confident, fallback returns `legal[0]` (a real action), never a
  skip. Skipping is a learned-policy decision; the prior is the
  conservative default.

## Caveats

- **`actions_taken` semantics matter.** The graph builder dedupes
  by AGENT NAME (regardless of error status) — that's what fixes the
  chunk-9 epoch-8 recursion bug. The router itself just consumes
  whatever list it's given. Keep the deduping at the call site so the
  router stays a pure function.

- **`max_steps` overlaps with LangGraph's recursion_limit.** The graph
  builder sets `recursion_limit = max(200, 12 * max_steps + 32)` to
  cover the worst case. If you raise `max_steps` past ~15 in custom
  builds, double-check the recursion limit math.

## Tests

`tests/test_router.py` covers each termination reason, the legal-
action computation under all precondition combinations, the
graph-confidence-threshold gating, the skip-offer gating, and the
rule-based prior fallback.
