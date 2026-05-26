"""
Web-search skill — Exa + Tavily wrappers with retry/backoff/clamp.

This package wraps the two web-search providers AgensFlow supports as
first-class skills. Configuration lives in `WebSearchConfig` and ships
with sensible defaults; see `README.md` in this directory for the full
explanation of every knob.

Public API:

  - `make_web_search_exa(...)`, `make_web_search_tavily(...)` —
    factories that produce LangGraph node functions for the policy
    graph to invoke
  - `WebSearchConfig` — the typed config dataclass; instances are
    constructed by the central `agensflow.config` loader and passed
    through to the factories

Implementation lives in `core.py` (HTTP wrappers, retry logic, clamp
helpers); the configuration schema lives in `config.py` so users can
tune behavior via YAML without editing source.
"""

from agensflow.runtime.web_search.config import WebSearchConfig
from agensflow.runtime.web_search.core import (
    EXA_BACKOFF_BASE_S,
    EXA_BACKOFF_CAP_S,
    EXA_MAX_RETRIES,
    EXA_SYNTHETIC_TOKEN_COST,
    TAVILY_SYNTHETIC_TOKEN_COST,
    _backoff_seconds,
    _clamp_exa_args,
    _exa_request_with_retry,
    _is_rate_limited,
    _tavily_request_with_retry,
    make_web_search_exa,
    make_web_search_tavily,
)

__all__ = [
    "WebSearchConfig",
    "make_web_search_exa",
    "make_web_search_tavily",
    # Internals exposed for the test suite — not part of the stable
    # public API. Will move to `_internal` in a future cleanup.
    "EXA_MAX_RETRIES",
    "EXA_BACKOFF_BASE_S",
    "EXA_BACKOFF_CAP_S",
    "EXA_SYNTHETIC_TOKEN_COST",
    "TAVILY_SYNTHETIC_TOKEN_COST",
    "_is_rate_limited",
    "_backoff_seconds",
    "_clamp_exa_args",
    "_exa_request_with_retry",
    "_tavily_request_with_retry",
]
