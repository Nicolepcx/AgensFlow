# `agensflow.runtime.preflight`

Pre-flight checks — validate external dependencies with cheap probes
before any LLM token is spent. Born from the chunk-9 disaster: ~$5 of
LLM cost burned running against a throttled EXA endpoint that a single
30-second probe at sweep start would have caught.

## Purpose

A pre-flight check earns its keep when:

- **One real LLM call costs more than the probe.** Even at OpenRouter
  pennies, a few hundred coordination iterations per epoch dwarfs the
  ~$0.01 per dependency the probe spends.
- **The failure mode is binary** ("API key invalid" → 100% failure rate
  across every variant). Catching this once at startup beats classifying
  it 56 times mid-run.
- **The fix is structural** (env var, top up account, network) — no
  amount of router cleverness will route around a bad credential.

This module probes each registered dependency with the cheapest valid
request, classifies failures via `governance.classify_error`, and
returns a `PreflightResult` with a human-readable report. The runner
calls `result.all_passed` and aborts before the policy graph spins up.

## Architecture

```
run_preflight_checks(checks=None, *, registry=DEFAULT_CHECKS, config=None)
  └─ for each check name in (checks ?? config.default_checks ?? registry.keys()):
       └─ DEFAULT_CHECKS[name](timeout_s=config.<name>_timeout_s)
            ├─ check_openrouter:  GET  https://openrouter.ai/api/v1/models
            ├─ check_exa:         POST https://api.exa.ai/search        (numResults=1)
            └─ check_tavily:      POST https://api.tavily.com/search    (max_results=1)
       └─ classify HTTP status / exception → AgentErrorReason
       └─ build CheckResult(passed, detail, suggested_fix, ...)
  └─ PreflightResult(checks=[...])
       ├─ .all_passed  — True if every CONFIGURED check passed (skipped don't fail)
       └─ .format_report()  — multi-line human-readable summary for stdout
```

## Configuration knobs

All knobs live in `PreflightConfig` (see `config.py`); defaults ship in
`agensflow/configs/defaults/preflight.yaml`. Override any subset in
your own YAML and pass via `load_config("your.yaml")`.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `openrouter_timeout_s` | 10.0 | HTTP timeout for the `/models` probe | network is flaky — raise to 20; fast-fail CI — lower to 5 |
| `exa_timeout_s` | 15.0 | HTTP timeout for the EXA probe | EXA's neural search is slow today — raise to 25; same fast-fail rationale to lower |
| `tavily_timeout_s` | 15.0 | HTTP timeout for the Tavily probe | as Exa |
| `default_checks` | `[openrouter, exa, tavily]` | which checks `run_preflight_checks()` runs when no explicit list given | LLM-only sweep — drop to `[openrouter]` (saves 30s); MCP-heavy run — extend with custom check names registered via the `registry` arg |

## Usage

### Default config (no setup):

```python
from agensflow.runtime.preflight import run_preflight_checks

result = run_preflight_checks()  # all defaults
if not result.all_passed:
    print(result.format_report())
    sys.exit(1)
```

### Override via YAML:

```yaml
# my-config.yaml
preflight:
  openrouter_timeout_s: 5.0
  default_checks:
    - openrouter   # LLM-only sweep, skip web search probes
```

```python
from agensflow.config import load_config
from agensflow.runtime.preflight import run_preflight_checks

cfg = load_config("my-config.yaml")
result = run_preflight_checks(config=cfg.preflight)
```

### Override programmatically (e.g., for tests):

```python
from agensflow.runtime.preflight import PreflightConfig, run_preflight_checks

cfg = PreflightConfig(default_checks=["openrouter"], openrouter_timeout_s=2.0)
result = run_preflight_checks(config=cfg)
```

### Custom checks (MCP tools, internal services):

```python
from agensflow.runtime.preflight import (
    DEFAULT_CHECKS, CheckResult, run_preflight_checks,
)

def check_my_mcp(*, timeout_s: float = 10.0) -> CheckResult:
    # ... probe my MCP server ...
    return CheckResult(name="my_mcp", passed=True, detail="ok")

registry = {**DEFAULT_CHECKS, "my_mcp": check_my_mcp}
result = run_preflight_checks(registry=registry)
```

## Required environment variables

Each built-in check is gated on its API key:

- `OPENROUTER_API_KEY` — for `check_openrouter`
- `EXA_API_KEY` — for `check_exa`
- `TAVILY_API_KEY` — for `check_tavily`

When a key isn't set, the check returns with `not_configured=True` —
treated as "this dependency isn't used in this experiment" rather than
as failure. `result.all_passed` ignores skipped checks. Lets users opt
out of dependencies they don't need without false-fail.

## Design notes

- **`PreflightConfig` is mutable by mechanism, immutable by convention.**
  We deliberately did NOT mark the dataclass `frozen=True`. The reason:
  OmegaConf's structured-config merge requires mutable nested dataclasses
  — `frozen=True` causes `ReadonlyConfigError` during YAML overlay
  merging. Since the YAML-driven config flow is the entire point of
  bringing this module into `agensflow.config`, we accept the trade-off.
  **Treat config instances as immutable in application code:** construct
  once at startup via `load_config(...)`, pass into
  `run_preflight_checks(config=...)`, never mutate after that. If you
  need per-run variation, build different `PreflightConfig` instances.

  `CheckResult` and `PreflightResult` ARE still `frozen=True` — they're
  runtime artifacts, not config schemas, so OmegaConf doesn't touch
  them and the safer default applies.

- **Classifier reuse.** Pre-flight uses the same `classify_error` +
  `_suggest_fix_for` from the governance module that the runtime uses
  mid-run. So a 402-with-credits-keyword from EXA gets classified as
  `RATE_LIMITED` (Exa quirk) by the probe in exactly the same way the
  retry wrapper would mid-run — single source of truth for what
  "broken" means.

- **Optional `timeout_s` kwarg.** User-registered checks may take a
  `timeout_s` kwarg or no kwargs. The runner tries `check_fn(timeout_s=...)`
  first and falls back to a bare call on `TypeError`. Lets you write
  custom checks without buying into the timeout-is-a-config-knob model.

## Caveats

- **The probe is itself a network call.** A pre-flight in a fully
  airgapped CI environment will fail (rightly) with NETWORK errors. If
  you have an offline test mode, gate `run_preflight_checks` behind a
  flag at the call site, not inside this module.

- **No quota inspection.** We probe with a single minimal request — if
  the user has $0.01 left and the probe consumes it, the experiment
  itself will then fail. Pre-flight catches "broken NOW", not "will
  break in 30 minutes". A budget tracker (Layer 4 territory) is the
  complementary mechanism.

- **No retries.** Pre-flight is fail-fast by design; if an EXA probe
  fails with a 429, we want to know *now* before launching, not after
  4 retries of backoff. The runtime's Layer 0 retry wrapper exists
  precisely so transient throttles mid-run don't kill the experiment —
  pre-flight covers the orthogonal "is this dependency working at all"
  question.

## Tests

`tests/test_preflight.py` covers:

- `CheckResult` and `PreflightResult` semantics (`all_passed`,
  `total_elapsed`, skipped-checks behavior)
- `format_report` content (status icons, suggested fixes, summary)
- `check_openrouter` / `check_exa` / `check_tavily` against mocked HTTP
  responses for: success, missing-key skip, 401/403 auth, 402 quota,
  429 rate-limit, 5xx server, timeout, network error
- `run_preflight_checks` aggregate path: all-pass, partial-fail,
  subset selection, unknown-check error
