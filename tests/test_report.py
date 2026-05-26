"""
Tests for Layer 6 — RunReport + SessionReport.

Pure-function tests using fabricated trace events + governance state.
Verifies:
  - AgentActivitySummary rollup (counts, tokens, latency, error reasons)
  - RunReport.from_run_artifacts builds correctly across success/halt/errored
  - format_human produces readable output
  - to_dict round-trips through JSON
  - SessionReport aggregates per-agent activity across runs correctly
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agensflow.runtime.governance import (
    AgentErrorReason,
    GovernancePolicy,
    GovernanceState,
    GovernanceViolation,
)
from agensflow.runtime.report import (
    AgentActivitySummary,
    RunReport,
    SessionReport,
    _summarize_agents,
)
from agensflow.runtime.trace import TraceCollector, TraceEvent


def _evt(agent: str, *, error: str | None = None, tokens: int = 100,
         latency: float = 0.5, model: str = "haiku-4.5") -> TraceEvent:
    return TraceEvent(
        agent=agent, model=model,
        input_state={}, output_update={},
        prompt_tokens=tokens // 2,
        completion_tokens=tokens // 2,
        total_tokens=tokens,
        latency_seconds=latency,
        error=error,
    )


# --------------------------------------------------------------------------- #
# AgentActivitySummary rollup
# --------------------------------------------------------------------------- #


class TestSummarizeAgents:
    def test_single_agent_one_call(self) -> None:
        out = _summarize_agents([_evt("planner")])
        assert len(out) == 1
        s = out[0]
        assert s.agent == "planner"
        assert s.n_invocations == 1
        assert s.n_successes == 1
        assert s.n_failures == 0
        assert s.total_tokens == 100

    def test_multiple_agents_sorted_by_invocations(self) -> None:
        events = [_evt("a")] * 1 + [_evt("b")] * 3 + [_evt("c")] * 2
        out = _summarize_agents(events)
        # Most-invoked first
        assert [s.agent for s in out] == ["b", "c", "a"]

    def test_skip_events_excluded(self) -> None:
        """Skip decisions are routing, not invocations — they're tracked
        separately via RunReport.skip_count, not in the per-agent table."""
        events = [
            _evt("planner"),
            _evt("skip:web_search_exa"),
            _evt("skip:verifier_haiku"),
            _evt("solver"),
        ]
        out = _summarize_agents(events)
        agent_names = {s.agent for s in out}
        assert "skip:web_search_exa" not in agent_names
        assert "skip:verifier_haiku" not in agent_names
        assert agent_names == {"planner", "solver"}

    def test_error_reasons_classified(self) -> None:
        events = [
            _evt("solver", error="ValidationError: missing field", tokens=0),
            _evt("solver", error="ValidationError: schema", tokens=0),
            _evt("solver"),  # eventual success
        ]
        out = _summarize_agents(events)
        assert out[0].n_failures == 2
        assert out[0].n_successes == 1
        # Both errors classify as SCHEMA
        assert out[0].error_reasons == {"schema": 2}

    def test_mixed_error_reasons_counted(self) -> None:
        events = [
            _evt("web_search_exa", error="HTTP 429 Too Many Requests"),
            _evt("web_search_exa", error="HTTP 401 Unauthorized"),
            _evt("web_search_exa", error="HTTP 429"),
        ]
        out = _summarize_agents(events)
        assert out[0].error_reasons == {"rate_limited": 2, "auth": 1}

    def test_total_tokens_sums(self) -> None:
        events = [
            _evt("solver", tokens=100),
            _evt("solver", tokens=200),
            _evt("solver", tokens=50),
        ]
        out = _summarize_agents(events)
        assert out[0].total_tokens == 350

    def test_mean_latency(self) -> None:
        events = [
            _evt("solver", latency=0.4),
            _evt("solver", latency=0.6),
        ]
        out = _summarize_agents(events)
        assert out[0].mean_latency_seconds == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# RunReport.from_run_artifacts
# --------------------------------------------------------------------------- #


def _trace_with_events(*events: TraceEvent) -> TraceCollector:
    t = TraceCollector()
    for e in events:
        t.events.append(e)
    return t


class TestRunReportFromArtifacts:
    def test_completed_run(self) -> None:
        trace = _trace_with_events(
            _evt("planner", tokens=200),
            _evt("solver", tokens=1500),
            _evt("evaluator", tokens=400),
        )
        report = RunReport.from_run_artifacts(
            task_id="C1.1",
            trace=trace,
            governance_state=None,
            status="completed",
            policy_graph_backed_up=True,
            ruler_score=0.85,
            hybrid_reward=0.62,
        )
        assert report.task_id == "C1.1"
        assert report.status == "completed"
        assert report.total_tokens == 2100
        assert report.n_calls == 3
        assert report.skip_count == 0
        assert len(report.agents) == 3
        assert report.policy_graph_backed_up is True
        assert report.ruler_score == 0.85
        assert report.hybrid_reward == 0.62
        assert report.governance_violations == []
        assert report.halt_reason is None

    def test_halted_run_carries_halt_reason_and_fix(self) -> None:
        violation = GovernanceViolation(
            timestamp=datetime.now(timezone.utc),
            agent="web_search_exa",
            reason="terminal_error:quota",
            detail="quota exhausted",
            policy_value=True,
            actual_value="quota",
            error_reason=AgentErrorReason.QUOTA,
            suggested_fix="Top up at dashboard.exa.ai",
        )
        state = GovernanceState(policy=GovernancePolicy())
        state.violations.append(violation)
        trace = _trace_with_events(_evt("planner"))
        report = RunReport.from_run_artifacts(
            task_id="C2.1",
            trace=trace,
            governance_state=state,
            status="halted_by_policy",
            policy_graph_backed_up=False,
        )
        assert report.status == "halted_by_policy"
        assert report.policy_graph_backed_up is False
        assert "web_search_exa" in report.halt_reason
        assert report.suggested_fix == "Top up at dashboard.exa.ai"

    def test_skip_count_separate_from_n_calls(self) -> None:
        trace = _trace_with_events(
            _evt("planner"),
            _evt("skip:memory"),
            _evt("skip:web_search_exa"),
            _evt("solver_concise_haiku"),
            _evt("evaluator"),
        )
        report = RunReport.from_run_artifacts(
            task_id="C8.1",
            trace=trace, governance_state=None,
            status="completed", policy_graph_backed_up=True,
        )
        # 3 real invocations + 2 skips
        assert report.n_calls == 3
        assert report.skip_count == 2


# --------------------------------------------------------------------------- #
# RunReport.format_human
# --------------------------------------------------------------------------- #


class TestRunReportFormatHuman:
    def test_completed_run_layout(self) -> None:
        trace = _trace_with_events(
            _evt("planner", tokens=200),
            _evt("solver_concise_haiku", tokens=1500, model="haiku-4.5"),
            _evt("evaluator", tokens=400),
        )
        report = RunReport.from_run_artifacts(
            task_id="C1.5",
            trace=trace, governance_state=None,
            status="completed", policy_graph_backed_up=True,
            ruler_score=0.92, hybrid_reward=0.71,
        )
        out = report.format_human()
        assert "AgensFlow run report" in out
        assert "C1.5" in out
        assert "completed" in out
        assert "planner" in out
        assert "solver_concise_haiku" in out
        assert "Governance: clean" in out
        assert "RULER 0.92" in out
        assert "reward +0.71" in out

    def test_halted_layout_includes_violation_and_fix(self) -> None:
        violation = GovernanceViolation(
            timestamp=datetime.now(timezone.utc),
            agent="web_search_exa",
            reason="consecutive_failures",
            detail="failed 5x in a row with no successes",
            policy_value=5,
            actual_value=5,
            error_reason=AgentErrorReason.RATE_LIMITED,
            suggested_fix="Lower concurrency or top up quota.",
        )
        state = GovernanceState(policy=GovernancePolicy())
        state.violations.append(violation)
        trace = _trace_with_events(
            _evt("planner"),
            _evt("web_search_exa", error="HTTP 429", tokens=0),
        )
        report = RunReport.from_run_artifacts(
            task_id="C5.4",
            trace=trace, governance_state=state,
            status="halted_by_policy", policy_graph_backed_up=False,
        )
        out = report.format_human()
        assert "halted by policy" in out
        assert "consecutive_failures" in out
        assert "web_search_exa" in out
        assert "Lower concurrency" in out
        assert "backup SKIPPED" in out

    def test_failed_agent_marker(self) -> None:
        trace = _trace_with_events(
            _evt("solver", error="ValidationError"),
            _evt("solver"),  # success
        )
        report = RunReport.from_run_artifacts(
            task_id="x",
            trace=trace, governance_state=None,
            status="completed", policy_graph_backed_up=True,
        )
        out = report.format_human()
        assert "1/2 failed" in out


# --------------------------------------------------------------------------- #
# RunReport.to_dict / JSON round-trip
# --------------------------------------------------------------------------- #


class TestRunReportToDict:
    def test_completed_round_trip(self) -> None:
        trace = _trace_with_events(_evt("planner"))
        report = RunReport.from_run_artifacts(
            task_id="x",
            trace=trace, governance_state=None,
            status="completed", policy_graph_backed_up=True,
            metadata={"epoch": 5, "scenario_class": "C1"},
        )
        d = report.to_dict()
        # Round-trip through JSON; default=str handles datetime + StrEnum.
        s = json.dumps(d, default=str)
        parsed = json.loads(s)
        assert parsed["task_id"] == "x"
        assert parsed["metadata"]["epoch"] == 5
        assert parsed["metadata"]["scenario_class"] == "C1"


# --------------------------------------------------------------------------- #
# SessionReport
# --------------------------------------------------------------------------- #


def _make_run(task_id: str, status: str = "completed",
              tokens: int = 1000, agents: list[tuple[str, int, int]] | None = None,
              **kwargs) -> RunReport:
    """Helper: construct a RunReport directly for SessionReport tests."""
    agent_summaries = []
    if agents:
        for agent_name, n_calls, total_tok in agents:
            agent_summaries.append(AgentActivitySummary(
                agent=agent_name,
                n_invocations=n_calls,
                n_successes=n_calls,
                n_failures=0,
                total_tokens=total_tok,
                mean_latency_seconds=0.5,
                error_reasons={},
                models=["m"],
            ))
    return RunReport(
        task_id=task_id, status=status, runtime_seconds=1.0,
        total_tokens=tokens, n_calls=len(agent_summaries),
        skip_count=0, agents=agent_summaries,
        governance_violations=[],
        policy_graph_backed_up=(status == "completed"),
        **kwargs,
    )


class TestSessionReport:
    def test_status_counts(self) -> None:
        s = SessionReport(runs=[
            _make_run("a", "completed"),
            _make_run("b", "completed"),
            _make_run("c", "halted_by_policy"),
            _make_run("d", "errored"),
        ])
        assert s.status_counts == {"completed": 2, "halted_by_policy": 1, "errored": 1}
        assert s.n_governance_halts == 1
        assert s.n_runs == 4

    def test_total_tokens_sums(self) -> None:
        s = SessionReport(runs=[
            _make_run("a", tokens=1000),
            _make_run("b", tokens=2500),
            _make_run("c", tokens=500),
        ])
        assert s.total_tokens == 4000

    def test_per_agent_aggregate_rolls_up(self) -> None:
        """Two runs each invoking planner once → aggregate planner = 2 calls."""
        s = SessionReport(runs=[
            _make_run("a", agents=[("planner", 1, 200), ("solver", 1, 1500)]),
            _make_run("b", agents=[("planner", 1, 220), ("solver", 2, 3000)]),
        ])
        agg = s.per_agent_aggregate
        agg_by_name = {a.agent: a for a in agg}
        assert agg_by_name["planner"].n_invocations == 2
        assert agg_by_name["planner"].total_tokens == 420
        assert agg_by_name["solver"].n_invocations == 3
        assert agg_by_name["solver"].total_tokens == 4500
        # Sorted by n_invocations desc — solver first.
        assert agg[0].agent == "solver"
        assert agg[1].agent == "planner"

    def test_format_human(self) -> None:
        s = SessionReport(
            label="chunk-9 sustained traffic",
            runs=[
                _make_run("a", tokens=1000, agents=[("planner", 1, 200)]),
                _make_run("b", "halted_by_policy", tokens=500),
            ],
        )
        out = s.format_human()
        assert "session report" in out
        assert "chunk-9 sustained traffic" in out
        assert "Runs: 2" in out
        assert "1 ok" in out
        assert "1 halted-by-policy" in out
        assert "Governance halts: 1" in out

    def test_to_dict_serializable(self) -> None:
        s = SessionReport(runs=[_make_run("a", tokens=100)])
        d = s.to_dict()
        json.dumps(d, default=str)  # must not raise
