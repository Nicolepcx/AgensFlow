"""
Pre-flight checks — validate external dependencies before LLM cost.

The chunk-9 disaster (~$5+ wasted LLM tokens running against a throttled
EXA endpoint) could have been prevented by a single 30-second pre-flight
probe at sweep start. This module is that probe.

Each check function:
  - Hits the dependency with the cheapest valid request (~$0.01 or less)
  - Catches all exceptions and classifies them via `governance.classify_error`
  - Returns a structured `CheckResult` carrying pass/fail, diagnosis,
    error reason, and a suggested fix

Aggregate API:
  - `run_preflight_checks(checks=...)` runs the requested checks (or all
    by default) and returns a `PreflightResult`
  - `result.all_passed` — boolean for the calling runner
  - `result.format_report()` — pretty-printed multi-line summary the
    runner can print to stdout before deciding whether to abort

Default check set: openrouter, exa, tavily. Only checks the dependency
if the corresponding env var is set — `EXA_API_KEY` not set just skips
the EXA check (treats it as "not used in this experiment") rather than
failing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from agensflow.runtime.governance import (
    AgentErrorReason,
    _suggest_fix_for,
    classify_error,
)
from agensflow.runtime.preflight.config import PreflightConfig


# --------------------------------------------------------------------------- #
# Check result + aggregate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one pre-flight check.

    `passed=True` means the dependency is reachable and minimally
    functional (auth works, basic request succeeds).

    `passed=False` carries a diagnosis: what failed, why (typed via
    AgentErrorReason), and what the user can do about it.

    `not_configured=True` means the env var for this dependency wasn't
    set — treat as "not used in this experiment" rather than as failure.
    Lets users opt out of dependencies they don't need without making
    the pre-flight false-fail.
    """

    name: str  # "openrouter", "exa", "tavily", ...
    passed: bool
    detail: str
    error_reason: AgentErrorReason | None = None
    suggested_fix: str | None = None
    elapsed_seconds: float = 0.0
    not_configured: bool = False


@dataclass(frozen=True)
class PreflightResult:
    """Aggregate result from all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if every *configured* dependency passed. Skipped checks
        (not_configured=True) don't fail the aggregate."""
        return all(c.passed or c.not_configured for c in self.checks)

    @property
    def total_elapsed(self) -> float:
        return sum(c.elapsed_seconds for c in self.checks)

    def format_report(self) -> str:
        """Multi-line human-readable summary suitable for stdout."""
        lines: list[str] = []
        lines.append("═══ Pre-flight check report ═══")
        for c in self.checks:
            if c.not_configured:
                status = "○ skipped"
            elif c.passed:
                status = "✓ ok     "
            else:
                status = "✗ FAILED "
            lines.append(
                f"  {status} {c.name:<14} ({c.elapsed_seconds:.1f}s)  "
                f"{c.detail}"
            )
            if not c.passed and not c.not_configured and c.suggested_fix:
                lines.append(f"            → {c.suggested_fix}")
        lines.append(f"Total elapsed: {self.total_elapsed:.1f}s")
        if not self.all_passed:
            lines.append(
                "\nOne or more pre-flight checks FAILED. Aborting before "
                "any LLM tokens are spent. Fix the failures above and rerun."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-dependency checks
# --------------------------------------------------------------------------- #


def check_openrouter(*, timeout_s: float = 10.0) -> CheckResult:
    """Verify OpenRouter auth + connectivity via the /models endpoint.

    Hits the lightweight `/models` GET (returns the model catalog),
    which doesn't consume any LLM tokens but does require valid auth.
    Cheaper and faster than spinning up a real completion.
    """
    name = "openrouter"
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return CheckResult(
            name=name, passed=False, detail="OPENROUTER_API_KEY not set",
            error_reason=AgentErrorReason.AUTH,
            suggested_fix="Export OPENROUTER_API_KEY=... or add to .env",
            not_configured=True,
        )
    start = time.monotonic()
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - start
        if resp.status_code == 200:
            return CheckResult(
                name=name, passed=True,
                detail=f"auth ok ({len(resp.json().get('data', []))} models available)",
                elapsed_seconds=elapsed,
            )
        reason = classify_error(f"HTTP {resp.status_code} {resp.text[:200]}", agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"HTTP {resp.status_code}: {resp.text[:120]}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name, resp.text[:200]),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        reason = classify_error(str(exc), agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name),
            elapsed_seconds=elapsed,
        )


def check_exa(*, timeout_s: float = 15.0) -> CheckResult:
    """Verify EXA auth + quota with a single minimal search.

    The minimal search (numResults=1, generic query) costs ~$0.007 — well
    under the threshold where a pre-flight is worth it. Detects 402
    (quota), 429 (rate limit), 401/403 (auth), and connectivity issues.
    """
    name = "exa"
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return CheckResult(
            name=name, passed=False, detail="EXA_API_KEY not set",
            not_configured=True,
        )
    start = time.monotonic()
    try:
        resp = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"query": "agensflow preflight test", "numResults": 1, "type": "auto"},
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - start
        if resp.status_code == 200:
            n_results = len(resp.json().get("results", []) or [])
            return CheckResult(
                name=name, passed=True,
                detail=f"search ok ({n_results} results)",
                elapsed_seconds=elapsed,
            )
        reason = classify_error(f"HTTP {resp.status_code} {resp.text[:200]}", agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"HTTP {resp.status_code}: {resp.text[:120]}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name, resp.text[:200]),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        reason = classify_error(str(exc), agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name),
            elapsed_seconds=elapsed,
        )


def check_tavily(*, timeout_s: float = 15.0) -> CheckResult:
    """Verify Tavily auth + plan headroom with a single minimal search."""
    name = "tavily"
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return CheckResult(
            name=name, passed=False, detail="TAVILY_API_KEY not set",
            not_configured=True,
        )
    start = time.monotonic()
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": api_key,
                "query": "agensflow preflight test",
                "search_depth": "basic",
                "max_results": 1,
            },
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - start
        if resp.status_code == 200:
            n_results = len(resp.json().get("results", []) or [])
            return CheckResult(
                name=name, passed=True,
                detail=f"search ok ({n_results} results)",
                elapsed_seconds=elapsed,
            )
        reason = classify_error(f"HTTP {resp.status_code} {resp.text[:200]}", agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"HTTP {resp.status_code}: {resp.text[:120]}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name, resp.text[:200]),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        reason = classify_error(str(exc), agent=name)
        return CheckResult(
            name=name, passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            error_reason=reason,
            suggested_fix=_suggest_fix_for(reason, name),
            elapsed_seconds=elapsed,
        )


# --------------------------------------------------------------------------- #
# Aggregate runner
# --------------------------------------------------------------------------- #


# Registry of available checks. Users can extend by registering their own
# `Callable[[], CheckResult]`.
DEFAULT_CHECKS: dict[str, Callable[[], CheckResult]] = {
    "openrouter": check_openrouter,
    "exa": check_exa,
    "tavily": check_tavily,
}


def run_preflight_checks(
    checks: list[str] | None = None,
    *,
    registry: dict[str, Callable[..., CheckResult]] | None = None,
    config: PreflightConfig | None = None,
) -> PreflightResult:
    """Run the requested pre-flight checks and return aggregate result.

    `checks=None` (default) runs every check named in
    `config.default_checks` (or the registry's full key set if no config
    is supplied or `default_checks` is empty). Pass a list to subset
    explicitly (e.g., `["openrouter"]` for an LLM-only experiment).

    `registry` defaults to `DEFAULT_CHECKS` but can be overridden to
    inject custom checks (e.g. for user-registered MCP tools). User-
    registered checks may take a `timeout_s` kwarg or no kwargs — both
    work (we fall back if the call raises TypeError).

    `config` supplies per-provider timeouts and the default check set.
    Defaults applied when None.
    """
    reg = registry if registry is not None else DEFAULT_CHECKS
    cfg = config if config is not None else PreflightConfig()
    if checks is not None:
        names = checks
    elif registry is not None:
        # Caller supplied a custom registry without an explicit check list.
        # Honor "run everything in the registry" — the YAML default-check
        # list is meant to subset the BUILT-IN registry, not to filter
        # user-supplied ones (which the user already curated).
        names = list(reg.keys())
    else:
        # Default registry: honor cfg.default_checks if non-empty, else fall
        # back to the registry's full key set (preserves old behavior).
        names = list(cfg.default_checks) if cfg.default_checks else list(reg.keys())
    timeouts = {
        "openrouter": cfg.openrouter_timeout_s,
        "exa": cfg.exa_timeout_s,
        "tavily": cfg.tavily_timeout_s,
    }
    results: list[CheckResult] = []
    for name in names:
        if name not in reg:
            raise KeyError(
                f"Unknown pre-flight check: {name!r}. "
                f"Available: {sorted(reg.keys())}."
            )
        check_fn = reg[name]
        try:
            results.append(check_fn(timeout_s=timeouts.get(name, 15.0)))
        except TypeError:
            # User-registered check that doesn't accept timeout_s — call bare.
            results.append(check_fn())
    return PreflightResult(checks=results)
