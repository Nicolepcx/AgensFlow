# `agensflow.runtime.governance`

Error taxonomy + per-run policy enforcement. The substrate that
distinguishes **learnable signal** ("agent X is flaky-but-recoverable;
route around it") from **infrastructure failure** ("the API key is
broken; halt the run before more tokens are spent and do NOT pollute
the value estimates").

## Purpose

AgensFlow's whole pitch is *learnable coordination over a policy graph*.
That only works if the graph's value estimates aren't corrupted by data
that has nothing to do with coordination quality:

- A bad API key shows up as 100% failure on every variant of the
  affected agent. Without governance, the router would learn "never
  invoke this agent" — a wasted update that doesn't generalize.
- A persistent rate-limit looks identical to a slow agent in the trace.
  Without a typed reason, the cost penalty in the hybrid reward charges
  the agent for backoff time it doesn't deserve.
- A pathologically-cycling router (re-invoking the same agent with no
  state advancement) burns tokens indefinitely with no learning.

This module catches all three at the trace boundary, before they reach
the policy graph. The harness handles `BrokenAgentError` *separately*
from generic exceptions — skipping the policy-graph backup so
infrastructure problems never become "learned" router behavior.

## Architecture

```
trace.TraceCollector.record(event)
  └─ trace.on_event(event)              # callback bound by bind_governance_to_trace
       └─ classify_error(event.error)   # → AgentErrorReason
       └─ trace_logger.log(...)         # structured log with reason
       └─ GovernanceState.check_event(...)
            └─ [terminal-error halt]    # AUTH/QUOTA → BrokenAgentError immediately
            └─ [consecutive-failure]    # ≥ max_consecutive_failures_per_agent → halt
            └─ [max-calls cap]          # > max_calls_per_agent → halt (pathological cycle)
       (BrokenAgentError propagates to harness)

harness.run_one(...)
  └─ except BrokenAgentError as exc:
       └─ skip policy-graph backup
       └─ build RunReport with violation
       └─ surface to user
```

## Configuration knobs

All knobs live in `GovernancePolicy` (re-exported from `config.py`);
defaults ship in `agensflow/configs/defaults/governance.yaml`. Override
any subset in your own YAML and pass via `load_config("your.yaml")`.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `max_consecutive_failures_per_agent` | 5 | halt threshold for a single agent failing N times in a row with no success between | aggressive cost control — lower to 3; very flaky network — raise to 8-10 |
| `max_calls_per_agent` | 12 | halt threshold for total invocations of a single agent in one run | topology genuinely re-invokes agents (rare) — raise to 20+; stricter cycle-detection — lower to 6 |
| `halt_on_terminal_errors` | true | whether AUTH/QUOTA errors halt on first occurrence (skipping the consecutive-failure threshold) | testing substrate behavior under persistent provider failure — set false |

## Usage

### Default config (no setup):

```python
from agensflow.runtime.governance import GovernancePolicy, GovernanceState, bind_governance_to_trace
from agensflow.runtime.trace import TraceCollector

trace = TraceCollector()
state = GovernanceState(policy=GovernancePolicy())
bind_governance_to_trace(trace, state)
# now every trace.record() runs through governance
```

### Override via YAML:

```yaml
# my-config.yaml
governance:
  max_consecutive_failures_per_agent: 3
  max_calls_per_agent: 8
```

```python
from agensflow.config import load_config
from agensflow.runtime.governance import GovernanceState, bind_governance_to_trace

cfg = load_config("my-config.yaml")
state = GovernanceState(policy=cfg.governance)
bind_governance_to_trace(trace, state)
```

### Override programmatically (e.g., for tests):

```python
from agensflow.runtime.governance import GovernancePolicy, GovernanceState

policy = GovernancePolicy(
    max_consecutive_failures_per_agent=2,
    halt_on_terminal_errors=False,  # for testing recovery behavior
)
state = GovernanceState(policy=policy)
```

### Catching halts in the harness:

```python
from agensflow.runtime.governance import BrokenAgentError

try:
    result = run_one(...)
except BrokenAgentError as exc:
    # exc.violation is a structured GovernanceViolation
    print(f"Halted: {exc.violation.reason} on {exc.violation.agent}")
    print(f"Suggested fix: {exc.violation.suggested_fix}")
    # IMPORTANT: skip the policy-graph backup — don't pollute value estimates
```

## Error taxonomy

`AgentErrorReason` (StrEnum, public re-export) — typed reasons behind
any error string:

| reason | trigger | terminal? |
|---|---|---|
| `AUTH` | 401, 403, "invalid api key" | yes |
| `QUOTA` | 402 (without rate keyword), "credits exhausted" | yes |
| `RATE_LIMITED` | 429, 432, 402-with-credits/rate keyword, "rate-limited after" | no |
| `TIMEOUT` | "timeout", "timed out", "deadline exceeded" | no |
| `SCHEMA` | "ValidationError", Instructor `InvalidAgentOutputError` | no |
| `NETWORK` | "connection refused", "DNS", "SSL" | no |
| `SERVER` | 5xx | no |
| `UPSTREAM` | "unparseable", "malformed" | no |
| `PRECONDITION` | "no subproblem", "missing precondition" | no |
| `UNKNOWN` | didn't match any classifier | no |

`TERMINAL_REASONS = {AUTH, QUOTA}` — these won't recover within a single
run, so they trigger immediate halt when `halt_on_terminal_errors=True`.

Extension principle: only add a reason when there's a *behavioral*
distinction the framework should make based on it. Reasons that exist
purely for human readability go in the error string detail, not the enum.

## Required environment / wiring

- No environment variables specific to governance.
- Requires the runtime to call `bind_governance_to_trace(trace, state)`
  once at run construction. The runner (`agensflow.runtime.runner`)
  does this when a policy is supplied.
- Requires the harness to catch `BrokenAgentError` separately and skip
  the policy-graph backup. See `experiments/e03_production_traffic/harness.py`
  for the canonical pattern.

## Design notes

- **`GovernancePolicy` is mutable by mechanism, immutable by convention.**
  We deliberately did NOT mark the dataclass `frozen=True`, even though
  immutability would be the safer Python default. The reason: OmegaConf's
  structured-config merge requires mutable nested dataclasses — `frozen=True`
  causes `ReadonlyConfigError` during YAML overlay merging. Since the
  YAML-driven config flow is the entire point of bringing this module
  into `agensflow.config`, we accept the trade-off. **Treat policy
  instances as immutable in application code:** construct once at
  startup via `load_config(...)`, pass the instance into runtime code,
  never mutate after that. If you need per-run variation, build
  different `GovernancePolicy` instances rather than mutating one.

  `GovernanceViolation` IS still `frozen=True` — it's a runtime artifact,
  not a config schema, so OmegaConf doesn't touch it and the safer
  default applies.

- **Counters update before checks fire.** `check_event()` increments
  call/failure counters first, then evaluates policy bounds. This means
  the violation object reports the *post-event* state (e.g., "5 consecutive
  failures") which matches the user's mental model better than "would-be
  6th failure".

- **The trace stays generic.** `agensflow.runtime.trace` doesn't import
  governance. Wiring is one-way: governance binds itself to trace via
  `bind_governance_to_trace(trace, state)`. This keeps trace usable
  without governance (e.g., in unit tests) and lets governance evolve
  without forcing a trace-package change.

## Caveats

- **Classifier is conservative.** When an error string doesn't match a
  known pattern, `classify_error` returns `UNKNOWN` rather than
  guessing. This means new provider error formats may not trigger
  terminal-halt until added to the classifier — keep an eye on
  `agensflow.trace` logs for `error_reason=unknown` and extend
  the patterns when you see them.

- **Per-agent counters use the agent name as the key.** If two
  invocations of the same logical tool happen under different display
  names (e.g. `web_search_exa` vs `web_search_exa_v2`), they're tracked
  separately. Usually correct; occasionally surprising.

- **No cross-run state.** Each run gets a fresh `GovernanceState`. A
  provider that fails the first run but recovers by the second won't be
  remembered — the second run starts from zero failure counts. If you
  want cross-run circuit-breaking, build it as a wrapper around
  `GovernanceState` rather than modifying this module.

## Tests

`tests/test_governance.py` covers:

- `AgentErrorReason` enum semantics + uniqueness
- `classify_error` parametric matrix across all reasons
- `is_terminal` predicate + `TERMINAL_REASONS` membership
- `GovernancePolicy` construction + validation (rejects 0/negative limits)
- `GovernanceState` terminal-error halt path
- `GovernanceState` consecutive-failure threshold + reset on success
- `GovernanceState` max-calls cap
- `bind_governance_to_trace` wiring against a real `TraceCollector`

`tests/test_harness_governance.py` covers the harness-integration path:
`BrokenAgentError` is caught, policy-graph backup is skipped, RunReport
includes the structured violation.
