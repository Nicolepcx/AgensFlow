# `agensflow.runtime.report`

`RunReport` (per-run artifact) and `SessionReport` (sweep-level
aggregate). Pure-data structures built from a trace + governance state
+ outcome flag, with two consumers:

1. **User-facing pretty-print** (`format_human()`): a 10-second-readable
   stdout summary the user sees after every run / sweep. On governance
   halt, includes the specific violation + suggested fix.
2. **Programmatic / dashboards** (`to_dict()`): JSON-serializable
   form so structured log handlers, observability platforms, or
   downstream analyses consume runs uniformly.

## Purpose

Reports answer the questions a user has 10 seconds after a run finishes:

- **Did it work?** Status glyph + RULER score + reward delta.
- **What did the framework do?** Per-agent activity table — which agents
  fired, how many times, with what model bindings, total tokens.
- **What broke?** Per-agent error breakdown (rate_limited=3, schema=1)
  + governance violations with halt reason + suggested fix.
- **Did the substrate learn from this run?** Closing line on whether the
  policy graph was backed up — explicit "skipped because halted" message
  on infrastructure failures, so users know their value estimates aren't
  silently corrupted.

The reports never import from harness or runner — they're built from
trace events + governance state, both of which live elsewhere. This
keeps the report layer reusable across runners, smoke tests, and ad-hoc
diagnostics.

## Architecture

```
RunReport.from_run_artifacts(task_id, trace, governance_state, status, ...)
  └─ _summarize_agents(trace.events) → list[AgentActivitySummary]
       (excludes "skip:X" events — counted separately under skip_count)
  └─ pulls violations from governance_state (empty list if no governance)
  └─ derives halt_reason / suggested_fix from most-recent violation

RunReport
  ├─ .format_human(config?)  → multi-line stdout summary
  └─ .to_dict()              → JSON-serializable form

SessionReport(runs=[RunReport, ...])
  ├─ .per_agent_aggregate    → roll-up across all runs
  ├─ .status_counts          → {"completed": N, "halted_by_policy": M, ...}
  ├─ .format_human(config?)  → sweep-level pretty print
  └─ .to_dict()              → JSON-serializable, runs nested
```

## Configuration knobs

All knobs live in `ReportConfig` (see `config.py`); defaults ship in
`agensflow/configs/defaults/report.yaml`. Override any subset in your
own YAML and pass via `load_config("your.yaml")`.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `run_agent_col_width` | 28 | width of agent-name column in per-run table | long agent names wrap — raise; narrow terminal — lower |
| `run_model_col_width` | 32 | width of model-name column in per-run table | as above |
| `run_max_agents_in_table` | 0 | cap on rows in per-run agent table; 0 = no cap | run with huge variant pool — set to 10-20 to focus on top agents |
| `include_agent_error_detail` | true | include the "↳ errors: ..." sub-line | one-line dashboards / CI logs — disable; humans reading reports — keep on |
| `violation_detail_max_chars` | 400 | truncate violation.detail in human view (0 = no trunc) | violation messages get verbose with stack traces — keep at 400; debugging — set to 0 |
| `session_agent_col_width` | 32 | width of agent column in session rollup | as run version |
| `session_max_agents_in_table` | 0 | cap on rows in session rollup; 0 = no cap | hundreds of variants — set to 20-30 |

## Usage

### Default config (no setup):

```python
from agensflow.runtime.report import RunReport
report = RunReport.from_run_artifacts(
    task_id="t-001", trace=trace, governance_state=state,
    status="completed", policy_graph_backed_up=True,
)
print(report.format_human())  # uses ReportConfig defaults
```

### Override via YAML:

```yaml
# my-config.yaml
report:
  run_max_agents_in_table: 10        # only top 10 agents per run
  include_agent_error_detail: false  # one-line-per-agent CI mode
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
print(report.format_human(config=cfg.report))
```

### Override programmatically (e.g., for tests / dashboards):

```python
from agensflow.runtime.report import ReportConfig, RunReport

compact = ReportConfig(
    run_max_agents_in_table=5,
    include_agent_error_detail=False,
    violation_detail_max_chars=80,
)
print(report.format_human(config=compact))
```

### JSON serialization:

```python
import json
data = report.to_dict()
# Datetimes / enums in violations need default=str:
print(json.dumps(data, default=str, indent=2))
```

## Required wiring

- A `TraceCollector` instance to read events from. Skip events
  (`agent.startswith("skip:")`) are excluded from per-agent rollups by
  convention — they're routing decisions, not invocations, and are
  surfaced separately via `skip_count`.
- Optionally a `GovernanceState` (`None` is fine; reports just show
  zero violations). Pass the same `state` you bound via
  `bind_governance_to_trace(trace, state)`.
- Status string: `"completed" | "halted_by_policy" | "errored"`. The
  harness assigns this based on whether the run finished, was halted by
  `BrokenAgentError`, or crashed otherwise.

## Design notes

- **`ReportConfig` is mutable by mechanism, immutable by convention.**
  We deliberately did NOT mark the dataclass `frozen=True`, even though
  immutability would be the safer Python default. The reason: OmegaConf's
  structured-config merge requires mutable nested dataclasses —
  `frozen=True` causes `ReadonlyConfigError` during YAML overlay merging.
  Since the YAML-driven config flow is the entire point of bringing this
  module into `agensflow.config`, we accept the trade-off. **Treat config
  instances as immutable in application code:** construct once at
  startup via `load_config(...)`, pass into `format_human(config=...)`,
  never mutate after that.

  `RunReport`, `SessionReport`, and `AgentActivitySummary` ARE all
  `frozen=True` — they're runtime artifacts, not config schemas, so
  OmegaConf doesn't touch them and the safer default applies.

- **Reports are pure-data and don't import from the runner.** A
  `RunReport` can be built from any `(trace, state, status)` triple, so
  the same construction path works for the harness, ad-hoc smoke tests,
  and post-hoc replay tooling.

- **`to_dict()` is `dataclasses.asdict` — datetimes/enums need
  `default=str` for JSON.** We don't transform inside `to_dict` because
  observability platforms that ingest the dict directly (Datadog,
  honeycomb) handle these types natively; forcing string conversion
  would lose information. The README example shows the JSON-serialization
  recipe.

- **`format_human` uses Unicode glyphs.** ✓ ⛔ ✗ ⚠ ↳ … — these render
  fine on modern terminals + CI logs but may fail on legacy ASCII-only
  outputs. If you need pure-ASCII reports, use `to_dict()` and render
  via your own template instead of `format_human`.

## Caveats

- **Skip events are excluded from per-agent counts.** A
  `skip:web_search_exa` event indicates the router DECIDED not to invoke
  that agent — counting it as activity for the agent would conflate two
  distinct things (decision vs. invocation). Skips show up as
  `skip_count` in the run header instead.

- **`per_agent_aggregate` recomputes per-call.** Cheap for tens of runs,
  noticeable for thousands. If you build large session reports
  repeatedly, cache the result rather than calling the property in
  every render.

- **No automatic file persistence.** Writing reports to disk is the
  caller's job (the harness does this in `_write_run_report()`). This
  module just produces the structured object + the formatted strings.

## Tests

`tests/test_report.py` covers:

- `_summarize_agents` rollup math (counts, tokens, mean latency,
  error_reasons map, models list)
- Skip events excluded from per-agent rollups, surfaced in skip_count
- `RunReport.from_run_artifacts` derives halt_reason / suggested_fix
  from most-recent violation
- `format_human` content — status glyphs, governance section presence,
  policy-graph closing line variants
- `SessionReport.per_agent_aggregate` — multi-run roll-up math
- `to_dict` round-trip correctness
