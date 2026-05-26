"""
Tests for the governance layer.

This file grows as the layer is built out. Current scope:
  - AgentErrorReason taxonomy + classify_error
  - is_terminal predicate

Future additions (tracked in chunk-9 todo):
  - GovernancePolicy + GovernanceState + violation accumulation
  - BrokenAgentError raising on threshold
  - RunReport dataclass + pretty-printer
"""

from __future__ import annotations

import pytest

from agensflow.runtime.governance import (
    TERMINAL_REASONS,
    AgentErrorReason,
    BrokenAgentError,
    GovernancePolicy,
    GovernanceState,
    GovernanceViolation,
    classify_error,
    is_terminal,
)


# --------------------------------------------------------------------------- #
# AgentErrorReason — basic enum behavior
# --------------------------------------------------------------------------- #


class TestAgentErrorReasonEnum:
    def test_str_value(self) -> None:
        # StrEnum: members are strings, comparisons against strings work.
        assert AgentErrorReason.AUTH == "auth"
        assert str(AgentErrorReason.RATE_LIMITED) == "rate_limited"

    def test_all_reasons_have_distinct_values(self) -> None:
        values = [r.value for r in AgentErrorReason]
        assert len(values) == len(set(values))


# --------------------------------------------------------------------------- #
# classify_error — taxonomy correctness
# --------------------------------------------------------------------------- #


class TestClassifyError:
    @pytest.mark.parametrize("err,expected", [
        # Rate-limit / throttle (tested first in classifier)
        ("HTTP 429 Too Many Requests", AgentErrorReason.RATE_LIMITED),
        ("Rate limit exceeded", AgentErrorReason.RATE_LIMITED),
        ("upstream throttling requests", AgentErrorReason.RATE_LIMITED),
        ("HTTP 432 plan exceeded", AgentErrorReason.RATE_LIMITED),
        # Layer 0 wrapper's specific message after exhausting retries
        ("exa rate-limited after 4 retries: HTTP 402", AgentErrorReason.RATE_LIMITED),
        # Exa quirk: 402 with credits-limit verbiage is throttle, not quota
        ("HTTP 402 You have exceeded your credits limit",
         AgentErrorReason.RATE_LIMITED),
        # Auth
        ("HTTP 401 Unauthorized", AgentErrorReason.AUTH),
        ("HTTP 403 Forbidden access", AgentErrorReason.AUTH),
        ("invalid api key", AgentErrorReason.AUTH),
        # Quota — actual exhaustion (NOT 402-with-credits-limit-which-is-rate-limit)
        ("HTTP 402 unsupported payment method", AgentErrorReason.QUOTA),
        ("Your quota has been exceeded for the month", AgentErrorReason.QUOTA),
        ("Payment Required", AgentErrorReason.QUOTA),
        ("Credits exhausted, top up to continue", AgentErrorReason.QUOTA),
        # Timeout
        ("Request timeout after 20s", AgentErrorReason.TIMEOUT),
        ("Operation timed out", AgentErrorReason.TIMEOUT),
        ("deadline exceeded", AgentErrorReason.TIMEOUT),
        # Network
        ("Connection refused", AgentErrorReason.NETWORK),
        ("DNS resolution failed", AgentErrorReason.NETWORK),
        ("SSL handshake error", AgentErrorReason.NETWORK),
        # Schema
        ("ValidationError: missing field", AgentErrorReason.SCHEMA),
        ("InvalidAgentOutputError on solver", AgentErrorReason.SCHEMA),
        ("pydantic.ValidationError", AgentErrorReason.SCHEMA),
        # Server
        ("HTTP 500 Internal Server Error", AgentErrorReason.SERVER),
        ("HTTP 502 Bad Gateway", AgentErrorReason.SERVER),
        ("HTTP 503 Service Unavailable", AgentErrorReason.SERVER),
        # Upstream
        ("Tool returned unparseable response", AgentErrorReason.UPSTREAM),
        ("malformed json from provider", AgentErrorReason.UPSTREAM),
        # Precondition
        ("no subproblem or goal to search on", AgentErrorReason.PRECONDITION),
        ("agent missing precondition: subproblem", AgentErrorReason.PRECONDITION),
        # Unknown — explicit catch-all
        ("something completely unexpected happened", AgentErrorReason.UNKNOWN),
        ("", AgentErrorReason.UNKNOWN),
    ])
    def test_classification(self, err: str, expected: AgentErrorReason) -> None:
        assert classify_error(err) == expected

    def test_none_input(self) -> None:
        assert classify_error(None) == AgentErrorReason.UNKNOWN

    def test_case_insensitive(self) -> None:
        assert classify_error("RATE LIMIT EXCEEDED") == AgentErrorReason.RATE_LIMITED
        assert classify_error("Http 401 Unauthorized") == AgentErrorReason.AUTH

    def test_rate_limit_takes_precedence_over_quota(self) -> None:
        """Critical chunk-9 finding: Exa returns 402 with 'credits limit'
        on burst traffic. That's a *throttle* signal, not real exhaustion.
        Classifier MUST recognize it as RATE_LIMITED so the substrate
        doesn't halt prematurely."""
        msg = "exa request failed: HTTP 402 You have exceeded your credits limit"
        assert classify_error(msg) == AgentErrorReason.RATE_LIMITED


# --------------------------------------------------------------------------- #
# is_terminal — predicate for halt-on-detection decisions
# --------------------------------------------------------------------------- #


class TestIsTerminal:
    def test_auth_is_terminal(self) -> None:
        assert is_terminal(AgentErrorReason.AUTH)

    def test_quota_is_terminal(self) -> None:
        assert is_terminal(AgentErrorReason.QUOTA)

    def test_rate_limited_not_terminal(self) -> None:
        # Rate-limit is transient — Layer 0's wrapper retries, governance
        # only sees it after retry exhaustion. Even then, it MIGHT recover
        # for the next run, so not terminal.
        assert not is_terminal(AgentErrorReason.RATE_LIMITED)

    def test_timeout_not_terminal(self) -> None:
        assert not is_terminal(AgentErrorReason.TIMEOUT)

    def test_schema_not_terminal(self) -> None:
        # Schema failures are usually transient (LLM produces a bad
        # response once, the next attempt at the same agent might work).
        assert not is_terminal(AgentErrorReason.SCHEMA)

    def test_terminal_reasons_set_matches_predicate(self) -> None:
        """is_terminal and TERMINAL_REASONS must agree."""
        for reason in AgentErrorReason:
            assert is_terminal(reason) == (reason in TERMINAL_REASONS)


# --------------------------------------------------------------------------- #
# GovernancePolicy
# --------------------------------------------------------------------------- #


class TestGovernancePolicy:
    def test_defaults(self) -> None:
        p = GovernancePolicy()
        assert p.max_consecutive_failures_per_agent == 5
        assert p.max_calls_per_agent == 12
        assert p.halt_on_terminal_errors is True

    def test_validation_max_consecutive_failures_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_failures_per_agent"):
            GovernancePolicy(max_consecutive_failures_per_agent=0)

    def test_validation_max_calls_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_calls_per_agent"):
            GovernancePolicy(max_calls_per_agent=0)

    def test_mutable_by_mechanism(self) -> None:
        # GovernancePolicy is intentionally NOT frozen so OmegaConf can
        # merge YAML overrides onto its dataclass schema. Immutability
        # is by convention only — see governance/README.md design notes.
        # This test pins the design choice so we notice if someone adds
        # frozen=True back without considering the OmegaConf consequence.
        p = GovernancePolicy()
        p.max_calls_per_agent = 100  # would raise FrozenInstanceError if frozen
        assert p.max_calls_per_agent == 100


# --------------------------------------------------------------------------- #
# GovernanceState — terminal-error halt
# --------------------------------------------------------------------------- #


class TestTerminalErrorHalt:
    def test_auth_error_halts_immediately(self) -> None:
        state = GovernanceState(policy=GovernancePolicy())
        with pytest.raises(BrokenAgentError) as exc_info:
            state.check_event(
                agent="solver_haiku",
                error_reason=AgentErrorReason.AUTH,
                error_detail="HTTP 401 invalid api key",
            )
        v = exc_info.value.violation
        assert v.agent == "solver_haiku"
        assert v.reason == "terminal_error:auth"
        assert v.error_reason == AgentErrorReason.AUTH
        assert v.suggested_fix is not None
        assert "OPENROUTER_API_KEY" in v.suggested_fix or "API key" in v.suggested_fix

    def test_quota_error_halts_immediately(self) -> None:
        state = GovernanceState(policy=GovernancePolicy())
        with pytest.raises(BrokenAgentError) as exc_info:
            state.check_event(
                agent="web_search_exa",
                error_reason=AgentErrorReason.QUOTA,
                error_detail="HTTP 402 payment required",
            )
        assert "exa" in exc_info.value.violation.suggested_fix.lower()
        # Provider-specific hint should mention dashboard.exa.ai
        assert "dashboard.exa.ai" in exc_info.value.violation.suggested_fix

    def test_rate_limited_does_not_halt_immediately(self) -> None:
        """RATE_LIMITED is transient — no immediate halt; counts toward
        consecutive-failure threshold instead."""
        state = GovernanceState(policy=GovernancePolicy())
        # Single rate-limit event shouldn't raise.
        state.check_event(
            agent="web_search_exa",
            error_reason=AgentErrorReason.RATE_LIMITED,
        )
        assert state.agent_consecutive_failures["web_search_exa"] == 1
        assert state.violations == []

    def test_terminal_halt_disabled_by_policy(self) -> None:
        """halt_on_terminal_errors=False → AUTH errors don't halt
        immediately; they only count toward consecutive-failure cap."""
        state = GovernanceState(policy=GovernancePolicy(halt_on_terminal_errors=False))
        # Should NOT raise on first AUTH error.
        state.check_event(
            agent="solver_haiku",
            error_reason=AgentErrorReason.AUTH,
        )
        assert state.agent_consecutive_failures["solver_haiku"] == 1


# --------------------------------------------------------------------------- #
# GovernanceState — consecutive-failure threshold
# --------------------------------------------------------------------------- #


class TestConsecutiveFailureThreshold:
    def test_threshold_fires_at_max(self) -> None:
        state = GovernanceState(
            policy=GovernancePolicy(max_consecutive_failures_per_agent=3)
        )
        # 2 failures: no halt yet
        for _ in range(2):
            state.check_event(
                agent="solver_haiku",
                error_reason=AgentErrorReason.SCHEMA,  # non-terminal
            )
        # 3rd failure: halt
        with pytest.raises(BrokenAgentError) as exc_info:
            state.check_event(
                agent="solver_haiku",
                error_reason=AgentErrorReason.SCHEMA,
            )
        v = exc_info.value.violation
        assert v.reason == "consecutive_failures"
        assert v.policy_value == 3
        assert v.actual_value == 3

    def test_success_event_resets_counter(self) -> None:
        state = GovernanceState(
            policy=GovernancePolicy(max_consecutive_failures_per_agent=3)
        )
        # 2 failures
        for _ in range(2):
            state.check_event(agent="x", error_reason=AgentErrorReason.SCHEMA)
        # 1 success — resets the counter
        state.check_event(agent="x", error_reason=None)
        assert state.agent_consecutive_failures["x"] == 0
        # 2 more failures should NOT raise (counter restarted)
        for _ in range(2):
            state.check_event(agent="x", error_reason=AgentErrorReason.SCHEMA)
        # No exception raised through these 5 events.

    def test_per_agent_independent(self) -> None:
        """Two agents failing independently shouldn't pool toward one
        consecutive-failure counter."""
        state = GovernanceState(
            policy=GovernancePolicy(max_consecutive_failures_per_agent=3)
        )
        # 3 failures spread across two agents — neither hits the threshold.
        state.check_event(agent="a", error_reason=AgentErrorReason.SCHEMA)
        state.check_event(agent="b", error_reason=AgentErrorReason.SCHEMA)
        state.check_event(agent="a", error_reason=AgentErrorReason.SCHEMA)
        state.check_event(agent="b", error_reason=AgentErrorReason.SCHEMA)
        assert state.agent_consecutive_failures == {"a": 2, "b": 2}
        # No exception raised.


# --------------------------------------------------------------------------- #
# GovernanceState — max-calls cap
# --------------------------------------------------------------------------- #


class TestMaxCallsPerAgent:
    def test_total_calls_cap_fires(self) -> None:
        state = GovernanceState(
            policy=GovernancePolicy(max_calls_per_agent=3)
        )
        # 3 successful calls — fine
        for _ in range(3):
            state.check_event(agent="planner", error_reason=None)
        # 4th call exceeds cap → halt
        with pytest.raises(BrokenAgentError) as exc_info:
            state.check_event(agent="planner", error_reason=None)
        v = exc_info.value.violation
        assert v.reason == "max_calls_per_agent"
        assert v.policy_value == 3
        assert v.actual_value == 4

    def test_call_counts_include_errored_attempts(self) -> None:
        """Both success and error events count toward the per-agent cap.
        An agent that fails 3 times then succeeds 1 time has 4 invocations."""
        state = GovernanceState(
            policy=GovernancePolicy(
                max_calls_per_agent=4,
                # high so consecutive-failure check doesn't fire first
                max_consecutive_failures_per_agent=10,
            )
        )
        for _ in range(3):
            state.check_event(agent="x", error_reason=AgentErrorReason.SCHEMA)
        state.check_event(agent="x", error_reason=None)  # 4th call ok
        assert state.agent_call_counts["x"] == 4
        # 5th call exceeds cap (4) → halt
        with pytest.raises(BrokenAgentError):
            state.check_event(agent="x", error_reason=None)


# --------------------------------------------------------------------------- #
# GovernanceViolation — structure + serialization
# --------------------------------------------------------------------------- #


class TestGovernanceViolation:
    def test_dataclass_immutable(self) -> None:
        v = GovernanceViolation(
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            agent="x", reason="r", detail="d",
            policy_value=1, actual_value=1,
        )
        with pytest.raises(Exception):
            v.agent = "changed"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# BrokenAgentError — message contents
# --------------------------------------------------------------------------- #


class TestBrokenAgentError:
    def test_message_includes_agent_and_reason(self) -> None:
        v = GovernanceViolation(
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            agent="web_search_exa",
            reason="consecutive_failures",
            detail="failed 5x in a row",
            policy_value=5,
            actual_value=5,
            suggested_fix="Top up at dashboard.exa.ai.",
        )
        e = BrokenAgentError(v)
        msg = str(e)
        assert "web_search_exa" in msg
        assert "consecutive_failures" in msg
        assert "5" in msg  # the actual value
        assert "Top up" in msg  # suggested fix surfaced

    def test_violation_attribute_accessible(self) -> None:
        v = GovernanceViolation(
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            agent="a", reason="r", detail="d", policy_value=1, actual_value=1,
        )
        e = BrokenAgentError(v)
        assert e.violation is v


# --------------------------------------------------------------------------- #
# Layer 3 — bind_governance_to_trace integration
# --------------------------------------------------------------------------- #


def _make_event(agent: str, error: str | None = None, tokens: int = 100):
    """Helper: construct a minimal TraceEvent for governance integration tests."""
    from agensflow.runtime.trace import TraceEvent
    return TraceEvent(
        agent=agent,
        model="test-model",
        input_state={},
        output_update={},
        prompt_tokens=tokens // 2,
        completion_tokens=tokens // 2,
        total_tokens=tokens,
        latency_seconds=0.1,
        error=error,
    )


class TestBindGovernanceToTrace:
    def test_successful_event_does_not_raise(self) -> None:
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector
        trace = TraceCollector()
        state = GovernanceState(policy=GovernancePolicy())
        bind_governance_to_trace(trace, state, log_events=False)
        # Recording a success event should NOT raise.
        trace.record(_make_event("planner"))
        assert state.agent_call_counts == {"planner": 1}
        assert state.agent_consecutive_failures.get("planner", 0) == 0

    def test_terminal_error_raises_via_record(self) -> None:
        """The whole point: trace.record() raises BrokenAgentError when
        the bound governance state's policy is breached. This is what
        propagates through agent factories → LangGraph → harness."""
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector
        trace = TraceCollector()
        state = GovernanceState(policy=GovernancePolicy())
        bind_governance_to_trace(trace, state, log_events=False)
        with pytest.raises(BrokenAgentError) as exc_info:
            trace.record(_make_event(
                "web_search_exa",
                error="HTTP 401 Unauthorized: invalid api key",
            ))
        assert exc_info.value.violation.error_reason == AgentErrorReason.AUTH

    def test_consecutive_failures_via_record(self) -> None:
        """Multiple non-terminal failures accumulate via record() and
        eventually trip the consecutive-failure threshold."""
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector
        trace = TraceCollector()
        state = GovernanceState(
            policy=GovernancePolicy(max_consecutive_failures_per_agent=3)
        )
        bind_governance_to_trace(trace, state, log_events=False)
        # Two SCHEMA failures: don't raise yet.
        for _ in range(2):
            trace.record(_make_event("solver", error="ValidationError: schema"))
        # Third failure: raises.
        with pytest.raises(BrokenAgentError) as exc_info:
            trace.record(_make_event("solver", error="ValidationError: schema"))
        assert exc_info.value.violation.reason == "consecutive_failures"

    def test_logging_emits_records(self, caplog) -> None:
        """Each record() call emits one structured log entry at the
        agensflow.trace logger. Successful events at INFO; errors at WARNING.
        Verifies the structured `extra` fields are attached."""
        import logging
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector

        trace = TraceCollector()
        state = GovernanceState(policy=GovernancePolicy())
        bind_governance_to_trace(trace, state, log_events=True)

        with caplog.at_level(logging.INFO, logger="agensflow.trace"):
            trace.record(_make_event("planner", tokens=200))

        rec = caplog.records[-1]
        assert rec.name == "agensflow.trace"
        assert rec.levelno == logging.INFO
        assert getattr(rec, "agent", None) == "planner"
        assert getattr(rec, "tokens", None) == 200
        assert getattr(rec, "error_reason", "missing") is None  # success

    def test_logging_emits_warning_on_governance_violation(self, caplog) -> None:
        """When the bound state raises BrokenAgentError, an
        agensflow.governance WARNING is emitted before the exception
        propagates — so dashboards/log tails see the violation even when
        the run terminates abruptly."""
        import logging
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector

        trace = TraceCollector()
        state = GovernanceState(policy=GovernancePolicy())
        bind_governance_to_trace(trace, state, log_events=True)

        with caplog.at_level(logging.WARNING, logger="agensflow.governance"):
            with pytest.raises(BrokenAgentError):
                trace.record(_make_event(
                    "web_search_exa",
                    error="HTTP 402 unsupported payment method",  # QUOTA
                ))

        # There should be a governance warning record with structured fields.
        gov_records = [r for r in caplog.records if r.name == "agensflow.governance"]
        assert len(gov_records) >= 1
        rec = gov_records[-1]
        assert getattr(rec, "violation_reason", None) == "terminal_error:quota"
        assert getattr(rec, "agent", None) == "web_search_exa"
        assert getattr(rec, "suggested_fix", None)  # non-empty fix string

    def test_log_events_disabled_skips_logging(self, caplog) -> None:
        """log_events=False should suppress the per-event INFO records
        but governance still fires (and logs the WARNING on violation)."""
        import logging
        from agensflow.runtime.governance import bind_governance_to_trace
        from agensflow.runtime.trace import TraceCollector

        trace = TraceCollector()
        state = GovernanceState(policy=GovernancePolicy())
        bind_governance_to_trace(trace, state, log_events=False)

        with caplog.at_level(logging.DEBUG, logger="agensflow.trace"):
            trace.record(_make_event("planner"))

        trace_records = [r for r in caplog.records if r.name == "agensflow.trace"]
        assert trace_records == []  # nothing logged at agensflow.trace
