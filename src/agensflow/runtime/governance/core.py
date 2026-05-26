"""
Governance layer — error taxonomy, policy, state, violations, and report.

The substrate's whole pitch is *learnable coordination over policy decisions*,
which only makes sense if there's a principled governance layer underneath
that distinguishes:

  - learnable signal (an agent that's flaky-but-recoverable; the substrate
    should track per-edge reliability and route around)
  - infrastructure failure (broken API key, exhausted credits, persistent
    schema mismatch; halt with diagnosis, do NOT pollute the policy graph)

This module is the foundation of that distinction. Layers built on top:

  - error taxonomy (`AgentErrorReason` + `classify_error`): typed reasons
    behind any error string the framework sees
  - governance policy (`GovernancePolicy`): user-configurable per-run
    constraints (consecutive-failure cap, max-calls, terminal-error
    handling)
  - governance state (`GovernanceState`): tracks consumption against the
    policy across a single run; raises `BrokenAgentError` on violation
  - run report (`RunReport`): structured artifact summarizing the run for
    user-facing diagnostics, generated at run-end whether successful or
    halted

The harness catches `BrokenAgentError` separately from generic exceptions
and skips the policy-graph backup for that run, so infrastructure-level
failures never corrupt the substrate's value estimates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agensflow.runtime.trace import TraceCollector, TraceEvent


# Module-level loggers. The agensflow.governance namespace is what users
# attach handlers to when they want to route governance events to files,
# observability platforms, or stderr. We deliberately do NOT call
# logging.basicConfig() — leave configuration to the application.
governance_logger = logging.getLogger("agensflow.governance")
trace_logger = logging.getLogger("agensflow.trace")


# --------------------------------------------------------------------------- #
# AgentErrorReason — typed taxonomy for any error the framework sees
# --------------------------------------------------------------------------- #


class AgentErrorReason(StrEnum):
    """Typed classification of an agent's failure mode.

    Used by:
      - GovernancePolicy: to decide which reasons are terminal (halt now)
        vs. recoverable (count toward consecutive-failure threshold)
      - RunReport: to surface a clean summary of what went wrong
      - Future Layer-2 metrics: to attribute failures to root causes

    Extension principle: only add a reason when there's a *behavioral*
    distinction the framework should make based on it. Don't add reasons
    that exist solely for human readability — those go in the error
    string detail.
    """

    AUTH = "auth"                 # 401, 403, missing/invalid API key
    QUOTA = "quota"               # actual credit/quota exhaustion
    RATE_LIMITED = "rate_limited" # 429, 432 — transient throttle
    TIMEOUT = "timeout"           # network or wall-clock timeout
    SCHEMA = "schema"             # Instructor validation failure
    NETWORK = "network"           # connection / DNS / SSL error
    SERVER = "server"             # 5xx — provider-side
    UPSTREAM = "upstream"         # tool returned unparseable response
    PRECONDITION = "precondition" # agent invoked without required state
    UNKNOWN = "unknown"           # didn't match any classifier


def classify_error(error_str: str | None, agent: str | None = None) -> AgentErrorReason:
    """Map an error string to a typed `AgentErrorReason`.

    Conservative: when in doubt returns UNKNOWN rather than mis-classifying.
    The agent name is accepted but currently unused — reserved for cases
    where the same error message means different things for different
    agent kinds (e.g., "timeout" from an LLM agent vs. a tool agent).

    The classifier ordering matters: more-specific patterns are tested
    first. e.g. "rate-limited after 4 retries" must classify as
    RATE_LIMITED, not as SCHEMA (despite the word "after" appearing).
    """
    if not error_str:
        return AgentErrorReason.UNKNOWN
    s = str(error_str).lower()

    # Rate-limit / throttle. Tested first because Layer 0's wrapper raises
    # a specifically-formatted message after retry exhaustion that we want
    # to recognize rather than fall through to QUOTA.
    if "rate-limited after" in s:
        return AgentErrorReason.RATE_LIMITED
    if " 429 " in s or "429:" in s or "http 429" in s or "too many requests" in s:
        return AgentErrorReason.RATE_LIMITED
    if " 432 " in s or "432:" in s or "http 432" in s:
        return AgentErrorReason.RATE_LIMITED
    if "rate limit" in s or "rate-limit" in s or "throttl" in s:
        return AgentErrorReason.RATE_LIMITED
    # Exa quirk: 402 with "credits limit" or "rate" in body is throttle,
    # not real quota exhaustion. Distinguish here.
    if "402" in s and ("credits limit" in s or "rate" in s):
        return AgentErrorReason.RATE_LIMITED

    # Auth — missing/invalid credential
    if "401" in s or "403" in s or "unauthorized" in s or "forbidden" in s:
        return AgentErrorReason.AUTH
    if "invalid api key" in s or "api key" in s and "missing" in s:
        return AgentErrorReason.AUTH

    # Quota — actual exhaustion (after rate-limit retries failed to recover)
    if "402" in s and "rate" not in s:
        # Bare 402 without rate-limit indicators
        return AgentErrorReason.QUOTA
    if "quota" in s or "billing" in s or "payment required" in s:
        return AgentErrorReason.QUOTA
    if "credits" in s and ("exhausted" in s or "depleted" in s or "limit" in s):
        return AgentErrorReason.QUOTA

    # Timeout
    if "timeout" in s or "timed out" in s or "deadline exceeded" in s:
        return AgentErrorReason.TIMEOUT

    # Network
    if "connection" in s and ("refused" in s or "reset" in s or "aborted" in s):
        return AgentErrorReason.NETWORK
    if "dns" in s or "ssl" in s or "name resolution" in s:
        return AgentErrorReason.NETWORK

    # Schema (Instructor validation, our InvalidAgentOutputError)
    if "validation" in s or "invalidagentoutput" in s or "pydantic" in s:
        return AgentErrorReason.SCHEMA

    # Server — 5xx
    if any(f" {code} " in s or f"{code}:" in s or f"http {code}" in s
           for code in ("500", "501", "502", "503", "504", "505")):
        return AgentErrorReason.SERVER

    # Upstream — tool returned malformed response
    if "unparseable" in s or "malformed" in s or "unexpected response" in s:
        return AgentErrorReason.UPSTREAM

    # Precondition — agent invoked without required handoff state
    if "no subproblem" in s or "missing precondition" in s or "no goal" in s:
        return AgentErrorReason.PRECONDITION

    return AgentErrorReason.UNKNOWN


# Reasons that indicate the failure WON'T recover within a single run.
# When seen, the GovernancePolicy can choose to halt immediately rather
# than waiting for the consecutive-failure threshold — there's no point
# burning more LLM tokens on something that's structurally broken.
TERMINAL_REASONS: frozenset[AgentErrorReason] = frozenset({
    AgentErrorReason.AUTH,    # wrong API key won't fix itself mid-run
    AgentErrorReason.QUOTA,   # credits won't appear mid-run
})


def is_terminal(reason: AgentErrorReason) -> bool:
    """True if `reason` is unrecoverable within a single session."""
    return reason in TERMINAL_REASONS


# --------------------------------------------------------------------------- #
# GovernanceViolation — structured record of a policy breach
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GovernanceViolation:
    """One policy violation, captured at the moment it fires.

    These accumulate on `GovernanceState` and ship in the `RunReport`
    so the user sees not just *that* something went wrong but exactly
    which constraint was breached, the policy bound, and the actual
    value that breached it.

    Designed for both human readability (the harness pretty-prints it)
    and machine consumption (JSON-serializable via dataclasses.asdict).
    """

    timestamp: datetime
    agent: str
    reason: str  # short ID like "consecutive_failures" or "max_calls"
    detail: str  # human-readable explanation
    policy_value: Any  # the policy's threshold
    actual_value: Any  # the observed value that breached it
    error_reason: AgentErrorReason | None = None  # if triggered by an error
    suggested_fix: str | None = None  # actionable hint for the user


# --------------------------------------------------------------------------- #
# BrokenAgentError — raised when policy is breached
# --------------------------------------------------------------------------- #


class BrokenAgentError(Exception):
    """Raised when an agent's behavior breaches the GovernancePolicy.

    Distinct from generic exceptions because the harness handles it
    *specially*: skip the policy-graph backup (don't pollute the
    substrate with infrastructure-failure data) and surface the
    structured violation to the user via the RunReport.

    Carries the violation as a structured object — handlers should read
    `.violation` instead of parsing the exception message.
    """

    def __init__(self, violation: GovernanceViolation) -> None:
        self.violation = violation
        msg = (
            f"Run halted by GovernancePolicy at agent {violation.agent!r}: "
            f"{violation.reason} ({violation.detail}). "
            f"Policy value: {violation.policy_value}, observed: {violation.actual_value}."
        )
        if violation.suggested_fix:
            msg += f" Suggested: {violation.suggested_fix}"
        super().__init__(msg)


# --------------------------------------------------------------------------- #
# GovernancePolicy — user-configurable per-run constraints
# --------------------------------------------------------------------------- #


@dataclass
class GovernancePolicy:
    """Per-run governance constraints.

    Defaults are conservative: catch broken infrastructure quickly without
    false-positives on flaky-but-recoverable agents. Production users can
    tighten or relax via construction args or YAML override.

    Note: not `frozen=True` because OmegaConf's structured-config merge
    requires mutable nested dataclasses. Treat as immutable by
    convention: construct once at startup (typically via the
    `agensflow.config` loader, which builds it from YAML), then pass
    the instance into runtime code — never mutate after construction.
    If the user wants different policies for different runs, they
    construct different policies and pass them in.
    """

    # Halt the run if any agent fails this many times consecutively
    # (with no successful invocations in between). Catches broken tools
    # cheaply — the threshold is low enough to limit wasted LLM tokens
    # but high enough to avoid false positives on transiently-flaky agents.
    max_consecutive_failures_per_agent: int = 5

    # Halt if any single agent is invoked more than this many times in
    # one run. With chunk-8/9's variant pools, an individual agent should
    # typically be invoked ≤2x per run; >12 indicates the router is
    # cycling pathologically. Cheap circuit-break independent of error
    # pattern.
    max_calls_per_agent: int = 12

    # Reasons that should halt the run *immediately* on first occurrence,
    # without waiting for the consecutive-failure threshold. Default:
    # AUTH and QUOTA (real credential / billing problems that won't
    # recover within the run, so spending more tokens is wasted).
    halt_on_terminal_errors: bool = True

    def __post_init__(self) -> None:
        # Light validation — catches misconfigurations at construction.
        if self.max_consecutive_failures_per_agent < 1:
            raise ValueError(
                "max_consecutive_failures_per_agent must be >= 1; "
                "policies that never halt are pointless."
            )
        if self.max_calls_per_agent < 1:
            raise ValueError("max_calls_per_agent must be >= 1.")


# --------------------------------------------------------------------------- #
# GovernanceState — tracks consumption against the policy during a run
# --------------------------------------------------------------------------- #


@dataclass
class GovernanceState:
    """Per-run accumulator that tracks agent activity against a policy.

    Mutated by the harness/runtime as agents fire. `check_event()` is
    called after every agent event (success or error) and raises
    `BrokenAgentError` if any policy bound is breached.

    The state object is also the source-of-truth for the `RunReport`'s
    governance section: violations, per-agent activity counts, etc.
    """

    policy: GovernancePolicy
    start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Per-agent invocation counts. Includes both successful and errored
    # events — counts every attempt, since the policy bounds attempts.
    agent_call_counts: dict[str, int] = field(default_factory=dict)
    # Per-agent consecutive-failure counter. Reset to 0 on a successful
    # event for that agent.
    agent_consecutive_failures: dict[str, int] = field(default_factory=dict)
    # Violations accumulated so far. The first one usually causes a halt
    # via BrokenAgentError, but we record it here too so the RunReport
    # can show it.
    violations: list[GovernanceViolation] = field(default_factory=list)

    def check_event(
        self,
        *,
        agent: str,
        error_reason: AgentErrorReason | None,
        error_detail: str = "",
    ) -> None:
        """Update state for one agent event and raise if policy breached.

        Call this after every TraceEvent the runtime records. `agent` is
        the agent name; `error_reason` is None for success events,
        otherwise the classified reason from `classify_error()`.

        Raises `BrokenAgentError` (with a structured violation) when:
          - `policy.halt_on_terminal_errors` is True AND error_reason is
            in TERMINAL_REASONS (auth/quota — won't recover this run)
          - consecutive failures for this agent reach
            `policy.max_consecutive_failures_per_agent`
          - total calls to this agent reach `policy.max_calls_per_agent`
        """
        # Update counters first so the violation reflects the post-event state.
        self.agent_call_counts[agent] = self.agent_call_counts.get(agent, 0) + 1
        if error_reason is None:
            # Successful event resets the consecutive-failure counter.
            self.agent_consecutive_failures[agent] = 0
        else:
            self.agent_consecutive_failures[agent] = (
                self.agent_consecutive_failures.get(agent, 0) + 1
            )

        # Check (1): terminal-error halt.
        if (
            self.policy.halt_on_terminal_errors
            and error_reason is not None
            and is_terminal(error_reason)
        ):
            v = GovernanceViolation(
                timestamp=datetime.now(timezone.utc),
                agent=agent,
                reason=f"terminal_error:{error_reason.value}",
                detail=(
                    f"Agent {agent!r} failed with terminal error reason "
                    f"{error_reason.value!r} (won't recover within this run)."
                ),
                policy_value=True,
                actual_value=error_reason.value,
                error_reason=error_reason,
                suggested_fix=_suggest_fix_for(error_reason, agent, error_detail),
            )
            self.violations.append(v)
            raise BrokenAgentError(v)

        # Check (2): consecutive-failure threshold.
        consec = self.agent_consecutive_failures.get(agent, 0)
        if consec >= self.policy.max_consecutive_failures_per_agent:
            v = GovernanceViolation(
                timestamp=datetime.now(timezone.utc),
                agent=agent,
                reason="consecutive_failures",
                detail=(
                    f"Agent {agent!r} failed {consec} times consecutively "
                    f"with no successful invocations between them. "
                    f"This usually indicates broken infrastructure (API key, "
                    f"network, persistent schema mismatch) rather than a "
                    f"learnable reliability signal."
                ),
                policy_value=self.policy.max_consecutive_failures_per_agent,
                actual_value=consec,
                error_reason=error_reason,
                suggested_fix=_suggest_fix_for(error_reason, agent, error_detail),
            )
            self.violations.append(v)
            raise BrokenAgentError(v)

        # Check (3): total-calls cap.
        total = self.agent_call_counts.get(agent, 0)
        if total > self.policy.max_calls_per_agent:
            v = GovernanceViolation(
                timestamp=datetime.now(timezone.utc),
                agent=agent,
                reason="max_calls_per_agent",
                detail=(
                    f"Agent {agent!r} was invoked {total} times in one run, "
                    f"exceeding the per-agent call cap. The router is "
                    f"likely cycling pathologically (re-invoking the same "
                    f"agent without state advancement)."
                ),
                policy_value=self.policy.max_calls_per_agent,
                actual_value=total,
                error_reason=error_reason,
                suggested_fix=(
                    "Inspect router log for the cycle pattern, or raise the "
                    "policy's max_calls_per_agent if your topology genuinely "
                    "requires more invocations."
                ),
            )
            self.violations.append(v)
            raise BrokenAgentError(v)


def _suggest_fix_for(
    reason: AgentErrorReason | None, agent: str, detail: str = "",
) -> str | None:
    """Return a short actionable hint for an error reason. Used by
    GovernanceViolation.suggested_fix to give the user a clear next
    step in the RunReport rather than a raw error string."""
    if reason is None:
        return None
    detail_l = detail.lower()
    if reason == AgentErrorReason.AUTH:
        return (
            f"Verify the API key for {agent!r} is set correctly in your "
            f"environment (.env or shell export). For OpenRouter agents, "
            f"check OPENROUTER_API_KEY; for tools, the provider's specific key."
        )
    if reason == AgentErrorReason.QUOTA:
        provider_hint = ""
        if "exa" in agent.lower() or "exa" in detail_l:
            provider_hint = " Top up at dashboard.exa.ai."
        elif "tavily" in agent.lower() or "tavily" in detail_l:
            provider_hint = " Check plan limits at app.tavily.com."
        return f"Provider for {agent!r} is out of credits/quota.{provider_hint}"
    if reason == AgentErrorReason.RATE_LIMITED:
        return (
            f"Provider for {agent!r} is throttling. Layer-0 wrapper retried "
            f"with backoff; if you still see this, lower your concurrency "
            f"or stagger requests, or upgrade the provider plan."
        )
    if reason == AgentErrorReason.SCHEMA:
        return (
            f"Agent {agent!r} consistently produces output that doesn't match "
            f"its declared schema. Check the prompt + schema, or try a "
            f"different model binding via the variant pool."
        )
    if reason == AgentErrorReason.NETWORK or reason == AgentErrorReason.TIMEOUT:
        return f"Network/timeout problem reaching {agent!r}. Verify connectivity."
    if reason == AgentErrorReason.SERVER:
        return f"Provider for {agent!r} returned a 5xx. Likely transient — try again later."
    if reason == AgentErrorReason.PRECONDITION:
        return (
            f"Agent {agent!r} was invoked without its required handoff state "
            f"(e.g. missing subproblem/goal). Check the activation plan or "
            f"upstream agent outputs."
        )
    return None


# --------------------------------------------------------------------------- #
# Trace integration — wire governance into the event stream
# --------------------------------------------------------------------------- #


def bind_governance_to_trace(
    trace: "TraceCollector",
    state: "GovernanceState",
    *,
    log_events: bool = True,
) -> None:
    """Wire a GovernanceState into a TraceCollector so policy is enforced
    after every recorded event.

    Sets `trace.on_event` to a callback that:
      1. Emits a structured log record for the event (if log_events=True)
      2. Calls `state.check_event(...)` with the classified error reason
      3. Re-raises BrokenAgentError if state's check fires

    The trace stays generic — it doesn't import governance. This helper
    lives in governance.py and lets the runtime opt into governance by
    one line of wiring at the top of `run()`.

    `log_events=True` (default) emits one INFO record per agent event at
    the `agensflow.trace` logger, with structured fields (agent, model,
    tokens, error_reason). Users can route or filter these via standard
    logging configuration. Disable for hot loops where logging overhead
    matters more than observability (rarely).
    """

    def _on_event(event: "TraceEvent") -> None:
        # Classify the event's error (None for successful events).
        reason: AgentErrorReason | None
        if event.error is None:
            reason = None
        else:
            reason = classify_error(event.error, agent=event.agent)

        # Structured log emission. The `extra` dict carries fields that
        # structured log handlers (json/observability) can use directly;
        # the message string is for humans tailing stdout.
        if log_events:
            level = logging.INFO if reason is None else logging.WARNING
            trace_logger.log(
                level,
                "agent_event agent=%s status=%s",
                event.agent,
                "ok" if reason is None else f"error:{reason.value}",
                extra={
                    "agent": event.agent,
                    "model": event.model,
                    "tokens": event.total_tokens,
                    "latency_seconds": event.latency_seconds,
                    "error_reason": reason.value if reason else None,
                    "error_detail": event.error[:200] if event.error else None,
                },
            )

        # Governance check — may raise BrokenAgentError. We log the
        # impending violation BEFORE re-raising so the WARNING is visible
        # even when the exception terminates the run abruptly.
        try:
            state.check_event(
                agent=event.agent,
                error_reason=reason,
                error_detail=event.error or "",
            )
        except BrokenAgentError as exc:
            v = exc.violation
            governance_logger.warning(
                "governance_violation agent=%s reason=%s policy=%s actual=%s",
                v.agent, v.reason, v.policy_value, v.actual_value,
                extra={
                    "agent": v.agent,
                    "violation_reason": v.reason,
                    "policy_value": v.policy_value,
                    "actual_value": v.actual_value,
                    "error_reason": v.error_reason.value if v.error_reason else None,
                    "suggested_fix": v.suggested_fix,
                },
            )
            raise

    trace.on_event = _on_event
