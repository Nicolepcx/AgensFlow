# `agensflow.runtime.trace`

In-memory trace collection. Every agent invocation produces one
`TraceEvent`; the collector accumulates them across a run. The
substrate that everything else (governance, reward, report, viz)
reads from.

## Purpose

Tracing is the framework's universal observation channel. Three
distinct consumers depend on it:

- **Governance** binds `state.check_event` to `TraceCollector.on_event`
  so policy violations fire at the same boundary the trace records
  events. The trace stays generic — it doesn't import governance.
- **Reward + RULER** read the event stream to compute cost (token sum)
  and quality signals.
- **Report + viz** roll up per-agent activity, error reasons, models
  used, and skip-counts from the same events.

Failed validation attempts (Instructor's parse-error path) ARE recorded
as trace events with `error="..."` populated — so cost accounting sees
them and the framework's "fewer tokens per successful task than retry
stacks" claim has honest underlying data.

## Architecture

```
agent invocation (in OpenRouterClient)
  └─ on success:  TraceCollector.record(TraceEvent(...))
  └─ on parse_error hook: TraceCollector.record(TraceEvent(error="..."))
       └─ trace.events.append(event)
       └─ trace.on_event(event)   # callback (governance binds here)

router (graph.build_learning_graph)
  └─ on each iteration: TraceCollector.record_router_iteration({...})
       (only when enable_router_logging=True)
```

## Configuration knobs

**None at present.** Trace collection has no user-tunable library-wide
knobs. The runtime mechanisms are configured at their call sites:

- `enable_router_logging`: passed to `build_learning_graph(...)` —
  whether the router writes per-iteration forensic entries to
  `trace.router_log`. Off by default; flipped on for debugging
  recursion-loop runs.
- `on_event`: assigned by `bind_governance_to_trace(trace, state)` —
  the callback that fires after every recorded event. Generic
  mechanism; governance is one consumer.

If we add tunable knobs later (event-retention cap, JSON-line file
sink, sampling rate for high-volume runs), they would land in a new
`config.py` here following the canonical pattern.

## Usage

```python
from agensflow.runtime.trace import TraceCollector

trace = TraceCollector()
# ... pass to agent factories, runner, ruler, etc.
print(trace.summary())
print(f"Total tokens: {trace.total_tokens}")
print(f"Agent sequence: {trace.agent_call_sequence}")
```

## Design notes

- **The collector is generic about consumers.** It exposes an
  `on_event` callback slot but doesn't know what governance or
  observability is. This keeps trace usable in unit tests + ad-hoc
  scripts without dragging in the policy machinery.

- **`record_router_iteration` is a separate channel.** Router-loop
  forensics live on `trace.router_log` (a list of dicts), not in
  `trace.events`. Reason: events are AGENT invocations; router
  iterations are routing DECISIONS — distinct things. Conflating them
  would break per-agent reward backup and per-agent report rollups.

- **`record()` calls `on_event` AFTER appending.** This means the
  callback sees the event in its post-append position — useful for
  governance, which counts the latest event when checking the
  consecutive-failure threshold.

## Caveats

- **Unbounded list.** No retention cap. A pathological run that hits
  LangGraph's recursion ceiling can produce hundreds of events; the
  trace holds them all. Acceptable for current AgensFlow scale.

- **Synchronous callback.** `on_event` runs in the same call as
  `record()`. A heavy callback (e.g. one that writes to disk per event)
  blocks agent execution. Governance's check is cheap (~µs); custom
  callbacks should match.

## Tests

`tests/test_trace.py` covers basic collector semantics; broader
coverage lives in the modules that consume the trace
(`test_governance.py`, `test_report.py`, `test_runner.py`).
