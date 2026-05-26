"""
WebSearchConfig — typed configuration schema for the web-search wrapper layer.

Every knob users can tune lives here. The accompanying YAML defaults are
in `agensflow/configs/defaults/web_search.yaml`. The `agensflow.config`
loader merges the defaults with any user YAML and validates the result
against this dataclass.

Typing notes:
  - All fields have defaults so the dataclass can be constructed with no
    args (used by the loader as the schema baseline).
  - The class is `frozen=True` — once constructed by the loader, runtime
    code shouldn't mutate it. If a user wants different behavior per-run,
    they pass a different config instance.
  - Field names match the YAML keys exactly (the loader doesn't do any
    name munging). Stick to lowercase + underscores.

See `README.md` in this directory for the human explanation of each knob
and when to tune it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WebSearchConfig:
    """Configuration for the Exa + Tavily web-search wrappers.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once
    at startup via `agensflow.config.load_config(...)`, then pass the
    instance into module factories — runtime code shouldn't mutate it.
    """

    # ----- Exa: retry + backoff ----- #
    # Number of attempts the wrapper makes against the Exa API before
    # giving up on a rate-limit response. Higher values give more
    # tolerance for bursty traffic but increase the worst-case wall-time
    # per failed call.
    exa_max_retries: int = 4

    # Initial sleep between rate-limit retries, in seconds. Doubles each
    # retry (exponential backoff) up to exa_backoff_cap_s.
    exa_backoff_base_s: float = 1.0

    # Hard cap on per-retry sleep time. Prevents pathological
    # exponential-blowup waits on persistent throttling.
    exa_backoff_cap_s: float = 30.0

    # ----- Exa: argument clamping ----- #
    # Hard cap on the `numResults` parameter sent to Exa. Default of 3
    # keeps cost low during exploration; production users with cheaper
    # plans can raise this.
    exa_max_results: int = 3

    # Hard cap on `contextMaxCharacters` sent to Exa. Solvers rarely
    # benefit from >6KB per result; capping bounds cost.
    exa_context_max_chars: int = 6000

    # ----- Exa: cost accounting ----- #
    # Synthetic token-equivalent cost reported in trace events for each
    # Exa call. Lets the hybrid reward's cost penalty see web-search
    # cost on the same axis as LLM token cost. ~$0.005/call ≈ 1500
    # tokens at typical model pricing.
    exa_synthetic_token_cost: int = 1500

    # ----- Tavily: retry + backoff ----- #
    tavily_max_retries: int = 4
    tavily_backoff_base_s: float = 1.0
    tavily_backoff_cap_s: float = 30.0

    # Hard cap on Tavily's `max_results`. Default 5 (Tavily's calls
    # are cheaper so we tolerate a bit more).
    tavily_max_results: int = 5

    # Tavily synthetic token cost. ~$0.001-0.002/call ≈ 500 tokens.
    tavily_synthetic_token_cost: int = 500

    # ----- HTTP timeouts ----- #
    # Per-request timeout (in seconds) sent to httpx. Both providers
    # share this knob; if you need provider-specific values, fork the
    # config and use two policies.
    http_timeout_s: float = 20.0
