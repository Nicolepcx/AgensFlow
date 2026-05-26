"""
Web search skills — Exa and Tavily as alternative retrieval providers.

These are first-class skills in the activation plan, not buried inside a
solver variant. The router treats them as alternative actions at post-planner
signatures (alongside corpus `memory`), and the policy graph learns per
signature class:

  - whether external search is worth invoking at all (some queries the corpus
    answers cleanly; others have no answer there),
  - which provider to use when both are available (Exa is stronger on
    technical/academic content, Tavily is more general),
  - whether to combine corpus retrieval and external search (some queries
    benefit from both).

Tool-call cost is real but doesn't show up in token accounting, so each
tool invocation reports a synthetic-token estimate that the hybrid reward's
cost penalty can see — calibrated so the cost gap between Exa (~$0.005/call)
and Tavily (~$0.001-0.002/call) is reflected proportionally.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from agensflow.learning.belief import update_belief
from agensflow.runtime.trace import TraceCollector, TraceEvent
from agensflow.schema import Handoff

NodeFn = Callable[[Handoff], dict[str, Any]]


# Synthetic-token cost estimates per call. Calibrated so the cost penalty
# in compute_hybrid_reward sees a meaningful gap between providers.
EXA_SYNTHETIC_TOKEN_COST = 1500
TAVILY_SYNTHETIC_TOKEN_COST = 500


# --------------------------------------------------------------------------- #
# Retry / backoff configuration (Layer 0 — tool wrapper governance)
# --------------------------------------------------------------------------- #
#
# Web-search providers like Exa and Tavily commonly throttle aggressive
# burst traffic with rate-limit responses (429, sometimes returned as
# 402 with credits-limit verbiage on pay-as-you-go plans, sometimes 432
# for plan-overage on Tavily). These are *transient* — a brief backoff
# typically clears them. Without this wrapper, every burst from a chunk-
# style sweep would hammer the provider, hit the throttle, and cause
# every web-search invocation in the experiment to fail.
#
# The retry wrapper:
#   1. Detects rate-limit / throttle errors via _is_rate_limited()
#   2. Sleeps with exponential backoff (capped) before retry
#   3. Distinguishes terminal errors (auth, schema, etc.) which raise
#      immediately without retry
#
# Cost-clamping prevents the framework from issuing maximally expensive
# requests during exploration. Each provider has its own clamp logic
# matched to its API surface.

EXA_MAX_RETRIES = 4
EXA_BACKOFF_BASE_S = 1.0
EXA_BACKOFF_CAP_S = 30.0

TAVILY_MAX_RETRIES = 4
TAVILY_BACKOFF_BASE_S = 1.0
TAVILY_BACKOFF_CAP_S = 30.0


def _is_rate_limited(exc_or_response: object) -> bool:
    """Return True if the exception/response signals rate-limit / throttle.

    Patterns covered:
      - HTTP 429 ("Too Many Requests") — standard rate-limit signal
      - HTTP 432 — Tavily's plan-overage signal
      - HTTP 402 with "credits limit" or "rate" in the body — Exa's
        peculiar pay-as-you-go throttle dressed up as a credits error
      - Any string containing "rate limit" / "rate-limit" / "throttle"

    Conservative on the *retry* side: we'd rather retry a non-throttle
    error once and waste a second than miss a throttle and burn the
    whole experiment on cascading failures.
    """
    s = str(exc_or_response).lower()
    if "429" in s or "too many requests" in s:
        return True
    if "432" in s:
        return True
    if "402" in s and ("credits limit" in s or "rate" in s or "exceeded" in s):
        return True
    if ("rate limit" in s) or ("rate-limit" in s) or ("throttl" in s):
        return True
    return False


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff: base * 2^(attempt-1), capped at `cap`."""
    return min(base * (2 ** (attempt - 1)), cap)


def _clamp_exa_args(args: dict[str, Any]) -> dict[str, Any]:
    """Bound Exa search-API parameters to prevent expensive calls.

    During training/exploration the framework can issue many web-search
    requests; without clamping, default expansive params (large
    numResults, deep crawl, large content windows) burn through the
    user's Exa balance fast. Clamping holds the line at sensible
    defaults that still produce useful evidence for downstream solvers.
    """
    clamped = dict(args or {})
    # Hard cap on result count.
    clamped["numResults"] = int(min(max(clamped.get("numResults", 3), 1), 3))
    # Conservative defaults — "auto" is cheaper than "neural"; "fallback"
    # for livecrawl prefers cached results.
    clamped.setdefault("type", "auto")
    clamped.setdefault("livecrawl", "fallback")
    # Bound character window — solvers rarely benefit from >6KB per result.
    clamped["contextMaxCharacters"] = int(
        min(clamped.get("contextMaxCharacters", 6000), 6000)
    )
    return clamped


def _exa_request_with_retry(
    *,
    api_key: str,
    query: str,
    max_results: int,
    max_retries: int = EXA_MAX_RETRIES,
    backoff_base: float = EXA_BACKOFF_BASE_S,
    backoff_cap: float = EXA_BACKOFF_CAP_S,
    timeout_s: float = 20.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Call Exa's search API with retry-on-rate-limit + arg clamping.

    Raises on terminal failure (auth, schema, network) or when
    `max_retries` rate-limit retries are exhausted. Returns the parsed
    JSON response on success.

    `sleep_fn` is injectable for testing (so tests don't actually sleep).
    """
    body = _clamp_exa_args({
        "query": query,
        "numResults": max_results,
        "useAutoprompt": True,
        "contents": {"text": {"maxCharacters": 800}},
    })
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout_s,
            )
            # Detect rate-limit *before* raise_for_status so we can retry.
            if _is_rate_limited(f"{response.status_code} {response.text[:200]}"):
                if attempt >= max_retries:
                    raise httpx.HTTPStatusError(
                        f"exa rate-limited after {max_retries} retries: "
                        f"HTTP {response.status_code} — {response.text[:200]}",
                        request=response.request, response=response,
                    )
                sleep_fn(_backoff_seconds(attempt, backoff_base, backoff_cap))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if _is_rate_limited(exc) and attempt < max_retries:
                sleep_fn(_backoff_seconds(attempt, backoff_base, backoff_cap))
                continue
            raise
    # Defensive — loop should always exit via raise or return.
    raise last_exc if last_exc else RuntimeError("exa retry loop exited unexpectedly")


def _tavily_request_with_retry(
    *,
    api_key: str,
    query: str,
    max_results: int,
    max_retries: int = TAVILY_MAX_RETRIES,
    backoff_base: float = TAVILY_BACKOFF_BASE_S,
    backoff_cap: float = TAVILY_BACKOFF_CAP_S,
    timeout_s: float = 20.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Call Tavily's search API with retry-on-rate-limit.

    Tavily returns 432 for plan-overage (which we treat as throttle —
    it sometimes clears between requests on burst traffic) and standard
    429 for rate-limit. Same exponential-backoff strategy as Exa.
    """
    body = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",  # cheaper than "advanced"
        "max_results": int(min(max(max_results, 1), 5)),  # mild clamp
        "include_raw_content": False,
    }
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=timeout_s,
            )
            if _is_rate_limited(f"{response.status_code} {response.text[:200]}"):
                if attempt >= max_retries:
                    raise httpx.HTTPStatusError(
                        f"tavily rate-limited after {max_retries} retries: "
                        f"HTTP {response.status_code} — {response.text[:200]}",
                        request=response.request, response=response,
                    )
                sleep_fn(_backoff_seconds(attempt, backoff_base, backoff_cap))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if _is_rate_limited(exc) and attempt < max_retries:
                sleep_fn(_backoff_seconds(attempt, backoff_base, backoff_cap))
                continue
            raise
    raise last_exc if last_exc else RuntimeError("tavily retry loop exited unexpectedly")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _accumulate_refs(
    existing: dict[str, list[str]],
    additions: dict[str, list[str]],
) -> dict[str, list[str]]:
    out = dict(existing)
    for k, v in additions.items():
        out[k] = list(v)
    return out


def _with_belief_update(
    state: Handoff, update: dict[str, Any], agent: str,
) -> dict[str, Any]:
    projected = state.model_copy(update=update)
    update["belief"] = update_belief(state.belief, agent=agent, handoff=projected)
    return update


def _record_tool_event(
    *,
    trace: TraceCollector,
    skill_name: str,
    provider: str,
    state_snapshot: dict[str, Any],
    output_update: dict[str, Any],
    synthetic_tokens: int,
    latency_seconds: float,
    error: str | None = None,
) -> None:
    trace.record(TraceEvent(
        agent=skill_name,
        model=provider,
        input_state=state_snapshot,
        output_update=output_update,
        prompt_tokens=0,
        completion_tokens=synthetic_tokens,
        total_tokens=synthetic_tokens,
        latency_seconds=latency_seconds,
        error=error,
    ))


# --------------------------------------------------------------------------- #
# Exa
# --------------------------------------------------------------------------- #


def make_web_search_exa(
    trace: TraceCollector,
    *,
    max_results: int | None = None,
    skill_name: str = "web_search_exa",
    config: "Any | None" = None,
) -> NodeFn:
    """
    Build a web-search node backed by Exa's semantic-search API.

    The query sent to Exa is the current Handoff's `subproblem`. Results are
    merged into the existing `evidence` and `retrieved_context` fields so
    downstream agents see them identically to corpus retrievals.

    EXA_API_KEY must be set in the environment.

    Args:
        trace: trace collector to record the tool event into
        max_results: per-call max results override; falls through to
            `config.exa_max_results` if not provided. Both are clamped
            inside the request wrapper to bound cost.
        skill_name: name recorded on the trace event; defaults to
            "web_search_exa". Override when registering as a variant.
        config: WebSearchConfig instance — controls retry/backoff/clamp
            and synthetic token cost. When None, defaults are used.
    """
    from agensflow.runtime.web_search.config import WebSearchConfig
    cfg = config if config is not None else WebSearchConfig()

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EXA_API_KEY not set; required for web_search_exa. Add it to .env."
        )

    # Resolve effective max_results: caller > config > default sentinel.
    effective_max_results = (
        max_results if max_results is not None else cfg.exa_max_results
    )

    def exa_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        query = state.subproblem or state.goal or ""
        if not query:
            update = _with_belief_update(state, {}, "memory")
            _record_tool_event(
                trace=trace, skill_name=skill_name, provider="exa",
                state_snapshot=snapshot, output_update=update,
                synthetic_tokens=0, latency_seconds=0.0,
                error="no subproblem or goal to search on",
            )
            return update

        start = time.monotonic()
        try:
            data = _exa_request_with_retry(
                api_key=api_key, query=query,
                max_results=effective_max_results,
                max_retries=cfg.exa_max_retries,
                backoff_base=cfg.exa_backoff_base_s,
                backoff_cap=cfg.exa_backoff_cap_s,
                timeout_s=cfg.http_timeout_s,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            update = _with_belief_update(state, {}, "memory")
            _record_tool_event(
                trace=trace, skill_name=skill_name, provider="exa",
                state_snapshot=snapshot, output_update=update,
                synthetic_tokens=cfg.exa_synthetic_token_cost,
                latency_seconds=elapsed, error=f"exa request failed: {exc}",
            )
            return update

        elapsed = time.monotonic() - start

        new_evidence: list[str] = list(state.evidence)
        new_context: list[str] = list(state.retrieved_context)
        for r in data.get("results", []) or []:
            title = r.get("title", "(untitled)")
            url = r.get("url", "")
            text = (r.get("text") or "").strip()
            if text:
                new_evidence.append(f"[exa] {title}: {text[:600]}")
                new_context.append(f"exa:{url}")

        update: dict[str, Any] = {
            "evidence": new_evidence,
            "retrieved_context": new_context,
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {
                    "evidence": ["subproblem", "exa_search"],
                    "retrieved_context": ["exa_search"],
                },
            ),
        }
        update = _with_belief_update(state, update, "memory")
        _record_tool_event(
            trace=trace, skill_name=skill_name, provider="exa",
            state_snapshot=snapshot, output_update=update,
            synthetic_tokens=cfg.exa_synthetic_token_cost,
            latency_seconds=elapsed,
        )
        return update

    return exa_node


# --------------------------------------------------------------------------- #
# Tavily
# --------------------------------------------------------------------------- #


def make_web_search_tavily(
    trace: TraceCollector,
    *,
    max_results: int | None = None,
    skill_name: str = "web_search_tavily",
    config: "Any | None" = None,
) -> NodeFn:
    """
    Build a web-search node backed by Tavily's general web search API.

    Cheaper per call than Exa; broader/general results. The framework's policy
    learns per signature when general web search is sufficient vs. when Exa's
    semantic-academic strength is worth the higher cost.

    TAVILY_API_KEY must be set in the environment.

    Args:
        trace: trace collector to record the tool event into
        max_results: per-call override; falls through to
            `config.tavily_max_results`. Clamped inside the wrapper.
        skill_name: trace event agent name; defaults to
            "web_search_tavily".
        config: WebSearchConfig instance — controls retry/backoff +
            synthetic token cost. When None, defaults are used.
    """
    from agensflow.runtime.web_search.config import WebSearchConfig
    cfg = config if config is not None else WebSearchConfig()

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set; required for web_search_tavily."
        )

    effective_max_results = (
        max_results if max_results is not None else cfg.tavily_max_results
    )

    def tavily_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        query = state.subproblem or state.goal or ""
        if not query:
            update = _with_belief_update(state, {}, "memory")
            _record_tool_event(
                trace=trace, skill_name=skill_name, provider="tavily",
                state_snapshot=snapshot, output_update=update,
                synthetic_tokens=0, latency_seconds=0.0,
                error="no subproblem or goal to search on",
            )
            return update

        start = time.monotonic()
        try:
            data = _tavily_request_with_retry(
                api_key=api_key, query=query,
                max_results=effective_max_results,
                max_retries=cfg.tavily_max_retries,
                backoff_base=cfg.tavily_backoff_base_s,
                backoff_cap=cfg.tavily_backoff_cap_s,
                timeout_s=cfg.http_timeout_s,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            update = _with_belief_update(state, {}, "memory")
            _record_tool_event(
                trace=trace, skill_name=skill_name, provider="tavily",
                state_snapshot=snapshot, output_update=update,
                synthetic_tokens=cfg.tavily_synthetic_token_cost,
                latency_seconds=elapsed, error=f"tavily request failed: {exc}",
            )
            return update

        elapsed = time.monotonic() - start

        new_evidence: list[str] = list(state.evidence)
        new_context: list[str] = list(state.retrieved_context)
        for r in data.get("results", []) or []:
            title = r.get("title", "(untitled)")
            url = r.get("url", "")
            content = (r.get("content") or "").strip()
            if content:
                new_evidence.append(f"[tavily] {title}: {content[:600]}")
                new_context.append(f"tavily:{url}")

        update: dict[str, Any] = {
            "evidence": new_evidence,
            "retrieved_context": new_context,
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {
                    "evidence": ["subproblem", "tavily_search"],
                    "retrieved_context": ["tavily_search"],
                },
            ),
        }
        update = _with_belief_update(state, update, "memory")
        _record_tool_event(
            trace=trace, skill_name=skill_name, provider="tavily",
            state_snapshot=snapshot, output_update=update,
            synthetic_tokens=cfg.tavily_synthetic_token_cost,
            latency_seconds=elapsed,
        )
        return update

    return tavily_node
