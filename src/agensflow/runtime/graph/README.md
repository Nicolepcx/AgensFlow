# `agensflow.runtime.graph`

LangGraph builders. Two flavors:

- **`build_graph`** — linear, rule-based execution
  (chunk-2/3 backward-compat path).
- **`build_learning_graph`** — dynamic, policy-driven execution; a
  single router node consults the policy graph at each step. The
  framework's intended runtime path.

## Purpose

This module is where AgensFlow's distinguishing claim — *learnable
orchestration policy* — lands as actual code. The dynamic builder
turns the pure-function `select_next_action` into a LangGraph topology
that the runtime can execute end-to-end.

The split:

- The *substrate* (`policy_graph`) holds the value estimates.
- The *routing* (`router`) is the pure function that decides.
- The *graph* (this module) wires routing into LangGraph's
  agent-dispatch topology.

Each agent node returns control to a single router node; the router
emits `Command(goto=<next_agent>)` or `Command(goto=END)`. This keeps
the LangGraph topology compact (no agent-to-agent edges) and puts all
the routing logic in one place.

## Architecture

```
build_learning_graph(plan, nodes, *, policy_graph, trace,
                     max_steps, confidence_threshold,
                     reliability_weight, enable_skip,
                     enable_router_logging)
  └─ assert plan is non-branching (branching not yet supported)
  └─ assert every selected_skill has a node function
  └─ build StateGraph(Handoff)
       ├─ add router_node (entry point)
       │    └─ on each entry:
       │         ├─ actions_taken = dedupe-by-agent(trace.events)
       │         ├─ select_next_action(...)
       │         ├─ if action is None: Command(goto=END)
       │         ├─ if action.startswith("skip:"):
       │         │    └─ record synthetic trace event, loop
       │         └─ Command(goto=<action>)
       └─ for each skill: add agent node + edge skill→router
  └─ compile().with_config(recursion_limit=<from GraphConfig math>)
```

## Configuration knobs

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `recursion_limit_floor` | 200 | minimum LangGraph recursion budget | CI tightness — lower to 100 for fast-fail on routing bugs |
| `recursion_limit_per_step_multiplier` | 12 | per-step budget multiplier in the limit math | empirical worst-case; raise if you see legitimate runs hit the ceiling |
| `recursion_limit_buffer` | 32 | added budget on top of `multiplier * max_steps` | rarely tune |

The substrate-level routing knobs (`max_steps`, `enable_skip`,
`enable_router_logging`) come from `RouterConfig`; the
UCB/confidence/reliability knobs come from `PolicyGraphConfig`. The
builder threads them as kwargs into the inner `select_next_action`
calls — they're not duplicated on `GraphConfig`.

Defaults ship in `agensflow/configs/defaults/graph.yaml`.

## Usage

### Linear builder (no learning):

```python
from agensflow.runtime.graph import build_graph
graph = build_graph(plan, nodes)
result = graph.invoke(handoff)
```

### Dynamic builder (the framework's intended path):

```python
from agensflow.runtime.graph import build_learning_graph
graph = build_learning_graph(
    plan, nodes,
    policy_graph=policy_graph,
    trace=trace,
    max_steps=12,
    enable_skip=False,
)
result = graph.invoke(handoff)
```

### YAML-driven (chunks 8/9):

```yaml
# my-config.yaml
router:
  max_steps: 12
  enable_skip: true
  enable_router_logging: false
policy_graph:
  confidence_threshold: 5
  reliability_weight: 0.5
graph:
  recursion_limit_floor: 200
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
graph = build_learning_graph(
    plan, nodes, policy_graph=policy_graph, trace=trace,
    max_steps=cfg.router.max_steps,
    confidence_threshold=cfg.policy_graph.confidence_threshold,
    reliability_weight=cfg.policy_graph.reliability_weight,
    enable_skip=cfg.router.enable_skip,
    enable_router_logging=cfg.router.enable_router_logging,
)
```

## Design notes

- **`GraphConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module config.

- **Branching is unimplemented.** `plan.branch_rule.enabled=True`
  raises `NotImplementedError` in both builders. Branching runtime
  (parallel coalitions + merge strategies) is a separate later
  workstream.

- **`actions_taken` is deduped by agent name.** This is the chunk-9
  fix: tool-failure events used to be filtered out (`error is None`),
  causing the router to re-pick the failing tool indefinitely until
  LangGraph's recursion ceiling fired. Now any event for agent X
  counts as "X was attempted." Per-edge failure data is preserved
  separately for the substrate's reliability learning (Mechanism A+C).

- **Recursion limit math is conservative.** Each substrate step
  triggers multiple LangGraph accounting events; the worst-case
  multiplier of 12 was empirically derived. Tightening it risks
  legit-but-slow routes hitting the ceiling.

## Caveats

- **Inline skip loop blocks the event loop.** When `enable_skip=True`
  and the policy chooses many consecutive skips, the router_node
  runs them all in a single LangGraph dispatch (no agent-node
  invocation between them). Cheap (synthetic events only) but
  CPU-bound.

- **The graph builder does NOT bind governance to the trace.** That's
  the runner's job — see `runtime/runner.py`. If you build a graph
  manually and want governance enforcement, call
  `bind_governance_to_trace(trace, state)` yourself.

## Tests

`tests/test_graph.py` covers linear-builder edge wiring, dynamic-
builder router-node behavior across all decision branches, skip-loop
mechanics, recursion-limit math.
