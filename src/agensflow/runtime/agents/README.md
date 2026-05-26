# `agensflow.runtime.agents`

Agent factories for the five base skills (planner, memory, solver,
verifier, evaluator). Each factory returns a node function suitable
for use as a LangGraph node.

## Purpose

Agents are the units the policy graph orchestrates. This module
provides the *factories* — closures that capture `(client, trace,
model_override, skill_name)` and return a `NodeFn(state) -> dict`
matching LangGraph's contract.

The pattern is deliberate:

- **Factories not classes.** A LangGraph node is a callable; closures
  give us callable + captured context with no inheritance ceremony.
- **`skill_name` is a parameter.** That's how the same factory
  (e.g. `make_solver`) backs every solver variant
  (`solver_fast`, `solver_qwen_max`, `solver_concise_haiku`, etc.).
  Each variant resolves to a different model + optionally a different
  skill card; the node-level mechanism is identical.
- **Trace recording lives in the agent, not the framework.** Each
  factory calls `_record_success(trace, ...)` after a successful
  completion. Failed attempts are recorded automatically by the
  client's hooks. This split keeps "successful agent invocation"
  visible at the agent boundary (where outputs are known) while
  cost-of-failed-attempts is captured at the transport boundary.

## Architecture

```
make_planner(client, user_task, trace, model_override=None)
  └─ resolve model via models.get_model_for_skill("planner", ...)
  └─ return planner_node:
       └─ snapshot state
       └─ client.complete_typed(model=..., output_model=PlannerOutput, ...)
       └─ build update dict from parsed output
       └─ _with_belief_update(state, update, "planner")
       └─ _record_success(trace, ...)
       └─ return update

(same shape for make_memory, make_solver, make_verifier, make_evaluator)
```

`make_solver` and `make_verifier` additionally take `skill_name=` to
back variants. `make_solver` also resolves a skill-card system prompt
when the variant has a `variant_cards` binding.

## Configuration knobs

**None at present.** Per-call tuning (max_retries, temperature,
max_tokens) lives on `ClientConfig`; per-skill model bindings live on
`ModelsConfig`; per-skill behavioral spec lives on skill cards
(registered separately). This module is where those decisions
*compose*, not where they're declared.

If we add agent-level knobs later (e.g. per-agent system-prompt
overrides unattached to skill cards, or agent-specific token budgets
that override the client default), they would land in a new `config.py`
here following the canonical pattern.

## Usage

### Default (matches the chunk-2/3 contract):

```python
from agensflow.runtime.agents import make_planner, make_memory, make_solver
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.trace import TraceCollector

client = OpenRouterClient()
trace = TraceCollector()
nodes = {
    "planner": make_planner(client, user_task, trace),
    "memory": make_memory(client, documents, trace),
    "solver": make_solver(client, trace),
    "verifier": make_verifier(client, trace),
    "evaluator": make_evaluator(client, trace),
}
```

### Variant binding (chunks 6–9):

```python
nodes["solver_qwen_flash"] = make_solver(
    client, trace, skill_name="solver_qwen_flash",
)
nodes["solver_concise_haiku"] = make_solver(
    client, trace, skill_name="solver_concise_haiku",
)  # uses solver_concise skill card automatically
```

### Per-call model override (rare; usually go through variants):

```python
nodes["solver"] = make_solver(
    client, trace,
    model_override={"solver": "openai/gpt-5.4-mini"},
)
```

## Design notes

- **`skill_name` decoupled from `model`.** The solver factory takes
  `skill_name` (e.g. `solver_concise_haiku`); model resolution flows
  through `models.get_model_for_skill(skill_name)`. This means the
  policy graph records `skill_name` as the action — and the
  `(signature, action)` value backup is correctly attributed at
  per-variant granularity. If the factory took `model` directly, two
  variants on the same model would collide in the value table.

- **`_with_belief_update` is here, not in the agent function bodies.**
  Each agent's update dict gets a `belief` field appended via the
  shared helper. The actual update rules live in
  `agensflow.learning.belief` (rule-based per-agent contributions),
  imported lazily inside the helper to avoid the runtime↔learning
  circular import.

- **Zero retry stack at this layer.** Agents call
  `client.complete_typed(...)` once. Bounded validation retry happens
  inside Instructor (the client passes `max_retries=2`); transport
  retry happens inside the OpenAI SDK. There's no third retry layer
  at the agent boundary, by design — this IS the framework's "no
  retry stacks" thesis applied to its own implementation.

## Caveats

- **`AGENT_FACTORIES` is documentation, not a registry.** It's a
  `dict[skill, factory_function_name]` for human reference. The actual
  factory dispatch happens in the experiment code (or a higher-level
  builder like `agensflow.runtime.runner`), which knows how to call
  each factory's distinct signature.

- **Critic and synthesizer factories are intentionally unimplemented.**
  The chunks where they're needed (ambiguous, contradictory, high_risk
  regimes) haven't shipped. Adding them is straightforward by analogy
  with the existing factories.

## Tests

`tests/test_agents.py` covers each factory's happy path against
mocked completions, the variant `skill_name` flow for solver/verifier,
the skill-card resolution path for solver, and trace-event recording.
