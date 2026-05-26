# `agensflow.runtime.web_search`

First-class web-search skills for AgensFlow's variant pool. Wraps two
providers (Exa and Tavily) with a retry/backoff/clamp layer that
prevents transient throttle errors from cascading into experiment-wide
failures.

## Purpose

The framework's policy graph treats `web_search_exa` and
`web_search_tavily` as **alternative retrieval actions** at
post-planner signatures (alongside corpus `memory`). The router learns
per signature class:

- whether external search is worth invoking at all
- which provider to use when both are available (Exa is stronger on
  technical/academic content; Tavily is more general)
- whether to combine corpus retrieval with external search

This module is the substrate that makes those decisions executable.
Each call goes through a wrapper that:

1. **Clamps expensive parameters** (`numResults`, `contextMaxCharacters`)
   to bounded defaults — prevents the framework from accidentally
   maxing out provider cost during exploration.
2. **Retries on rate-limit responses** (HTTP 429, Tavily's 432, Exa's
   peculiar 402-with-credits-limit-language) with exponential backoff —
   so a brief provider throttle doesn't kill the experiment.
3. **Records cost into the trace** as synthetic token-equivalents — so
   the hybrid reward's cost penalty sees web-search calls on the same
   axis as LLM tokens.

## Architecture

```
make_web_search_exa(trace, config=WebSearchConfig())
  └─ exa_node(state)
       └─ _exa_request_with_retry(...)   # retry/backoff loop
            └─ _clamp_exa_args(...)      # bound expensive params
            └─ httpx.post → exa.ai       # actual API call
       └─ _record_tool_event(...)        # emit trace.TraceEvent

make_web_search_tavily(trace, config=WebSearchConfig())
  └─ tavily_node(state)
       └─ _tavily_request_with_retry(...)
            └─ httpx.post → tavily.com
       └─ _record_tool_event(...)
```

## Configuration knobs

All knobs live in `WebSearchConfig` (see `config.py`); defaults ship in
`agensflow/configs/defaults/web_search.yaml`. Override any subset in
your own YAML and pass it via `load_config("your.yaml")`.

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `exa_max_retries` | 4 | attempts before giving up on Exa rate-limit | provider is consistently throttling — raise to 6-8 to push through |
| `exa_backoff_base_s` | 1.0 | initial sleep between retries (seconds) | latency-sensitive runs — lower to 0.5; hot-pool runs — raise to 2 |
| `exa_backoff_cap_s` | 30.0 | max per-retry sleep | rarely needs tuning; keep ≤ provider rate-limit window |
| `exa_max_results` | 3 | hard cap on `numResults` | production user with cheap plan — raise to 5-10; chunk-9-style sweep — keep at 3 |
| `exa_context_max_chars` | 6000 | cap on `contextMaxCharacters` | solvers needing more context — raise to 10000-12000 |
| `exa_synthetic_token_cost` | 1500 | trace-level cost equivalent (~$0.005/call) | calibrate against your actual Exa pricing tier |
| `tavily_max_retries` | 4 | attempts before giving up on Tavily rate-limit | as Exa |
| `tavily_backoff_base_s` | 1.0 | initial retry sleep | as Exa |
| `tavily_backoff_cap_s` | 30.0 | max retry sleep | as Exa |
| `tavily_max_results` | 5 | cap on Tavily `max_results` | similar to Exa knob |
| `tavily_synthetic_token_cost` | 500 | trace-level cost equivalent (~$0.001-0.002/call) | calibrate against actual pricing |
| `http_timeout_s` | 20.0 | per-request timeout (both providers) | flaky network — raise to 30; fast-fail experiments — lower to 10 |

## Usage

### Default config (no setup):

```python
from agensflow.runtime.web_search import make_web_search_exa
from agensflow.runtime.trace import TraceCollector

trace = TraceCollector()
exa_node = make_web_search_exa(trace)  # defaults
```

### Override via YAML:

```yaml
# my-config.yaml
web_search:
  exa_max_results: 5
  exa_max_retries: 6
  http_timeout_s: 30.0
```

```python
from agensflow.config import load_config
from agensflow.runtime.web_search import make_web_search_exa

cfg = load_config("my-config.yaml")
exa_node = make_web_search_exa(trace, config=cfg.web_search)
```

### Override programmatically (e.g., for tests):

```python
from agensflow.runtime.web_search import WebSearchConfig, make_web_search_exa

cfg = WebSearchConfig(exa_max_retries=2, exa_backoff_cap_s=5.0)
exa_node = make_web_search_exa(trace, config=cfg)
```

## Required environment variables

- `EXA_API_KEY` — required for `make_web_search_exa`. Without it, the
  factory raises at construction time (fail-fast).
- `TAVILY_API_KEY` — required for `make_web_search_tavily`.

The pre-flight module (`agensflow.runtime.preflight`) probes these
keys + headroom before any LLM token is spent. Recommended at the top
of any experiment runner:

```python
from agensflow.runtime.preflight import run_preflight_checks
result = run_preflight_checks(checks=["exa", "tavily"])
if not result.all_passed:
    print(result.format_report())
    sys.exit(1)
```

## Design notes

- **`WebSearchConfig` is mutable by mechanism, immutable by convention.**
  We deliberately did NOT mark the dataclass `frozen=True`, even though
  immutability would be the safer Python default. The reason: OmegaConf's
  structured-config merge requires mutable nested dataclasses — `frozen=True`
  causes `ReadonlyConfigError` during YAML overlay merging. Since the YAML-driven
  config flow is the entire point of this module's configurability, we
  accept the trade-off. **Treat config instances as immutable in
  application code:** construct once at startup via `load_config(...)`,
  pass the instance to factories, never mutate after that. If you need
  per-run variation, build different `WebSearchConfig` instances rather
  than mutating one.

## Caveats

- **Provider error semantics differ.** Exa returns HTTP 402 with
  "credits limit" verbiage on burst rate-limit *and* on real credit
  exhaustion. The wrapper distinguishes them by looking for the "rate"
  keyword, but this is fragile — keep an eye on Exa's API changelog.
- **The retry wrapper is synchronous** (uses `time.sleep`). Inside an
  async LangGraph runtime, this blocks the event loop for the backoff
  duration. Acceptable for current AgensFlow but worth noting if you
  later move to a fully-async router.
- **No request deduplication.** The same (signature, query) pair issued
  twice in a session hits the provider twice. Adding a session-level
  LRU cache here would be a small + cheap addition; not in scope for
  v0.1.

## Tests

`tests/test_web_search_retry.py` (30 tests) covers:

- `_is_rate_limited` discrimination across status codes + keywords
- `_backoff_seconds` exponential math + cap
- `_clamp_exa_args` defaults + bounds + non-mutation
- `_exa_request_with_retry` and `_tavily_request_with_retry` for
  success / retry-then-succeed / exhaust-retries / terminal-fail paths
