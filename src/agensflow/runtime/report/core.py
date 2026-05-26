"""
RunReport + SessionReport — structured run artifacts.

Produced at end-of-run from the trace + governance state + outcome flag.
Two consumers:

  1. **User-facing**: `format_human()` pretty-prints a 10-second-readable
     stdout summary the user sees after every run. On halt, includes the
     specific violation + suggested fix.
  2. **Programmatic / dashboards**: `to_dict()` JSON-serializes everything
     so structured log handlers, observability platforms, or downstream
     analysis can consume runs uniformly.

A `SessionReport` aggregates many `RunReport`s (used by sustained-traffic
experiment runners) — per-agent activity rolled up across runs,
status counts, total cost, top governance issues.

The reports are pure-data — they don't import from harness or runner,
so they can be built from any source of trace + governance state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agensflow.runtime.governance import (
    AgentErrorReason,
    GovernanceState,
    GovernanceViolation,
    classify_error,
)
from agensflow.runtime.report.config import ReportConfig

if TYPE_CHECKING:
    from agensflow.runtime.trace import TraceCollector


RunStatus = Literal["completed", "halted_by_policy", "errored"]


# --------------------------------------------------------------------------- #
# AgentActivitySummary — per-agent rollup from a trace
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentActivitySummary:
    """One agent's activity within a run (or aggregated across many).

    Skip events (`skip:X`) are excluded from per-agent rollups by
    convention — they're routing decisions, not agent invocations, and
    grouping them under their original agent's row would conflate
    distinct things. They're surfaced separately via the parent
    `RunReport.skip_count` instead.
    """

    agent: str
    n_invocations: int
    n_successes: int
    n_failures: int
    total_tokens: int
    mean_latency_seconds: float
    # Mapping of AgentErrorReason.value → count, e.g. {"rate_limited": 3}
    error_reasons: dict[str, int] = field(default_factory=dict)
    # Models the agent ran on (e.g. for variant agents). Usually a single
    # entry, but kept as a list to handle per-call model overrides.
    models: list[str] = field(default_factory=list)


def _summarize_agents(events: list[Any]) -> list[AgentActivitySummary]:
    """Roll up trace events into per-agent summaries."""
    by_agent: dict[str, list[Any]] = defaultdict(list)
    for e in events:
        if e.agent.startswith("skip:"):
            continue  # tracked separately
        by_agent[e.agent].append(e)

    summaries: list[AgentActivitySummary] = []
    for agent, agent_events in by_agent.items():
        n_total = len(agent_events)
        n_failures = sum(1 for e in agent_events if e.error is not None)
        n_successes = n_total - n_failures
        total_tokens = sum(e.total_tokens for e in agent_events)
        mean_latency = (
            sum(e.latency_seconds for e in agent_events) / max(1, n_total)
        )
        error_counter: Counter[str] = Counter()
        for e in agent_events:
            if e.error is None:
                continue
            reason = classify_error(e.error, agent=agent)
            error_counter[reason.value] += 1
        models = list({e.model for e in agent_events})
        summaries.append(AgentActivitySummary(
            agent=agent,
            n_invocations=n_total,
            n_successes=n_successes,
            n_failures=n_failures,
            total_tokens=total_tokens,
            mean_latency_seconds=mean_latency,
            error_reasons=dict(error_counter),
            models=models,
        ))
    # Stable order: sort by n_invocations desc so the most-active agent
    # appears first in human reports.
    summaries.sort(key=lambda s: -s.n_invocations)
    return summaries


# --------------------------------------------------------------------------- #
# RunReport — one run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunReport:
    """End-of-run artifact summarizing what the framework did."""

    task_id: str
    status: RunStatus
    runtime_seconds: float
    total_tokens: int
    n_calls: int
    skip_count: int  # number of skip:X decisions (routing, not invocation)
    agents: list[AgentActivitySummary] = field(default_factory=list)
    governance_violations: list[GovernanceViolation] = field(default_factory=list)
    # Whether the policy graph was updated for this run. False on
    # halt-by-policy (substrate protected from infrastructure noise) or
    # on error before reward computation.
    policy_graph_backed_up: bool = False
    # Optional per-task numerics.
    ruler_score: float | None = None
    hybrid_reward: float | None = None
    # If the run halted by policy, the most-recent violation's halt
    # reason + suggested fix are surfaced here for quick scanning
    # without unpacking the violations list.
    halt_reason: str | None = None
    suggested_fix: str | None = None
    # Free-form metadata — experiments can stash extra fields here
    # without forcing schema changes (epoch number, scenario class, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_run_artifacts(
        cls,
        *,
        task_id: str,
        trace: "TraceCollector",
        governance_state: GovernanceState | None,
        status: RunStatus,
        policy_graph_backed_up: bool,
        ruler_score: float | None = None,
        hybrid_reward: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RunReport":
        """Build a RunReport from a completed (or halted) run's artifacts.

        Used by the harness/runner to produce reports uniformly across
        success/halt/error outcomes. Agent activity comes from the trace
        events; governance violations come from the state (empty list if
        no governance was attached or no violations fired).
        """
        agents = _summarize_agents(list(trace.events))
        skip_count = sum(
            1 for e in trace.events if e.agent.startswith("skip:")
        )
        violations = (
            list(governance_state.violations) if governance_state else []
        )
        # Pull halt reason / fix from the most recent violation, if any.
        halt_reason: str | None = None
        suggested_fix: str | None = None
        if status == "halted_by_policy" and violations:
            v = violations[-1]
            halt_reason = f"{v.reason} on {v.agent}: {v.detail}"
            suggested_fix = v.suggested_fix
        return cls(
            task_id=task_id,
            status=status,
            runtime_seconds=trace.total_latency_seconds,
            total_tokens=trace.total_tokens,
            n_calls=sum(s.n_invocations for s in agents),
            skip_count=skip_count,
            agents=agents,
            governance_violations=violations,
            policy_graph_backed_up=policy_graph_backed_up,
            ruler_score=ruler_score,
            hybrid_reward=hybrid_reward,
            halt_reason=halt_reason,
            suggested_fix=suggested_fix,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form. Datetimes / enums in the violations
        list need `default=str` when calling json.dumps; everything
        else is plain types."""
        return asdict(self)

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #

    def format_human(self, config: ReportConfig | None = None) -> str:
        """Multi-line human-readable summary for stdout.

        Layout aimed at <10 seconds to read: status + runtime + tokens
        on the header line, per-agent table below, governance section
        only if relevant, policy-graph action on the closing line.

        `config` controls table widths, row caps, and detail truncation.
        Pass `cfg.report` from `load_config(...)` for YAML-driven
        formatting; defaults give the original layout when omitted.
        """
        cfg = config if config is not None else ReportConfig()
        status_glyph = {
            "completed": "✓ completed",
            "halted_by_policy": "⛔ halted by policy",
            "errored": "✗ errored",
        }[self.status]
        lines: list[str] = []
        lines.append("═══ AgensFlow run report ═══")
        # Header: task + status + runtime + tokens + skip-count
        meta_chunks = [
            f"Task: {self.task_id}",
            f"Status: {status_glyph}",
            f"Runtime: {self.runtime_seconds:.1f}s",
            f"Tokens: {self.total_tokens:,}",
        ]
        if self.skip_count > 0:
            meta_chunks.append(f"Skips: {self.skip_count}")
        lines.append("  ".join(meta_chunks))

        # Per-agent activity
        if self.agents:
            lines.append("")
            lines.append("Agent activity:")
            agent_w = cfg.run_agent_col_width
            model_w = cfg.run_model_col_width
            agents_to_show = (
                self.agents[: cfg.run_max_agents_in_table]
                if cfg.run_max_agents_in_table > 0
                else self.agents
            )
            for s in agents_to_show:
                model_str = (
                    s.models[0] if len(s.models) == 1
                    else ",".join(s.models)
                )
                status_mark = "✓" if s.n_failures == 0 else (
                    f"⚠ {s.n_failures}/{s.n_invocations} failed"
                )
                lines.append(
                    f"  {s.agent:<{agent_w}s} {model_str:<{model_w}s} "
                    f"{s.n_invocations:>2} call{'s' if s.n_invocations != 1 else ''}"
                    f"  {s.total_tokens:>6,} tok  {status_mark}"
                )
                if s.error_reasons and cfg.include_agent_error_detail:
                    detail = ", ".join(
                        f"{r}={n}" for r, n in s.error_reasons.items()
                    )
                    lines.append(f"      ↳ errors: {detail}")
            if (
                cfg.run_max_agents_in_table > 0
                and len(self.agents) > cfg.run_max_agents_in_table
            ):
                hidden = len(self.agents) - cfg.run_max_agents_in_table
                lines.append(f"  … {hidden} more agent(s) omitted")

        # Governance
        if self.governance_violations:
            lines.append("")
            lines.append(
                f"Governance violations: {len(self.governance_violations)}"
            )
            for v in self.governance_violations:
                lines.append(
                    f"  ⚠ {v.reason} on {v.agent}"
                )
                lines.append(f"      Policy: {v.policy_value}, "
                             f"observed: {v.actual_value}")
                if v.error_reason:
                    lines.append(f"      Error class: {v.error_reason.value}")
                detail_text = v.detail
                if (
                    cfg.violation_detail_max_chars > 0
                    and len(detail_text) > cfg.violation_detail_max_chars
                ):
                    detail_text = (
                        detail_text[: cfg.violation_detail_max_chars] + "…"
                    )
                lines.append(f"      Detail: {detail_text}")
                if v.suggested_fix:
                    lines.append(f"      → Suggested: {v.suggested_fix}")
        else:
            lines.append("")
            lines.append("Governance: clean (0 violations)")

        # Policy graph + RelativeJudge + reward
        lines.append("")
        if self.policy_graph_backed_up:
            ruler_str = f", RelativeJudge {self.ruler_score:.2f}" if self.ruler_score is not None else ""
            reward_str = f", reward {self.hybrid_reward:+.2f}" if self.hybrid_reward is not None else ""
            lines.append(f"Policy graph: backed up{ruler_str}{reward_str}")
        else:
            if self.status == "halted_by_policy":
                lines.append(
                    "Policy graph: backup SKIPPED — failure pattern indicates "
                    "broken infrastructure rather than learnable signal. "
                    "Substrate value estimates were NOT updated."
                )
            else:
                lines.append("Policy graph: not updated for this run")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SessionReport — many runs aggregated
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionReport:
    """Aggregate of many RunReports across a sustained-traffic session.

    Used by experiment runners to summarize a whole sweep at end-of-run.
    Per-agent activity rolls up across runs (so the user sees total
    invocations / total tokens / per-agent reliability across the
    entire experiment, not just any single run).
    """

    runs: list[RunReport] = field(default_factory=list)
    label: str = ""  # optional label, e.g. "chunk-9 sustained traffic"

    # ------------------------------------------------------------------ #
    # Aggregates (computed, not stored)
    # ------------------------------------------------------------------ #

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(Counter(r.status for r in self.runs))

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)

    @property
    def total_runtime_seconds(self) -> float:
        return sum(r.runtime_seconds for r in self.runs)

    @property
    def n_governance_halts(self) -> int:
        return sum(1 for r in self.runs if r.status == "halted_by_policy")

    @property
    def per_agent_aggregate(self) -> list[AgentActivitySummary]:
        """Roll up per-agent activity across all runs in the session."""
        # Combine each agent's per-run summaries.
        bucket: dict[str, dict[str, Any]] = {}
        for r in self.runs:
            for s in r.agents:
                b = bucket.setdefault(s.agent, {
                    "n_invocations": 0, "n_successes": 0, "n_failures": 0,
                    "total_tokens": 0, "latency_sum": 0.0,
                    "error_reasons": Counter(), "models": set(),
                })
                b["n_invocations"] += s.n_invocations
                b["n_successes"] += s.n_successes
                b["n_failures"] += s.n_failures
                b["total_tokens"] += s.total_tokens
                b["latency_sum"] += s.mean_latency_seconds * s.n_invocations
                b["error_reasons"].update(s.error_reasons)
                b["models"].update(s.models)
        out: list[AgentActivitySummary] = []
        for agent, b in bucket.items():
            out.append(AgentActivitySummary(
                agent=agent,
                n_invocations=b["n_invocations"],
                n_successes=b["n_successes"],
                n_failures=b["n_failures"],
                total_tokens=b["total_tokens"],
                mean_latency_seconds=(
                    b["latency_sum"] / max(1, b["n_invocations"])
                ),
                error_reasons=dict(b["error_reasons"]),
                models=sorted(b["models"]),
            ))
        out.sort(key=lambda s: -s.n_invocations)
        return out

    # ------------------------------------------------------------------ #
    # Pretty-printing + serialization
    # ------------------------------------------------------------------ #

    def format_human(self, config: ReportConfig | None = None) -> str:
        cfg = config if config is not None else ReportConfig()
        lines: list[str] = []
        title = f"AgensFlow session report{f' — {self.label}' if self.label else ''}"
        lines.append(f"═══ {title} ═══")
        sc = self.status_counts
        lines.append(
            f"Runs: {self.n_runs}  "
            f"({sc.get('completed', 0)} ok, "
            f"{sc.get('halted_by_policy', 0)} halted-by-policy, "
            f"{sc.get('errored', 0)} errored)"
        )
        lines.append(
            f"Total tokens: {self.total_tokens:,}  "
            f"Runtime: {self.total_runtime_seconds:.1f}s"
        )
        if self.n_governance_halts > 0:
            lines.append(
                f"Governance halts: {self.n_governance_halts}  "
                f"(see per-run reports for details)"
            )

        agg = self.per_agent_aggregate
        if agg:
            lines.append("")
            lines.append("Per-agent activity (rolled up across runs):")
            agent_w = cfg.session_agent_col_width
            agg_to_show = (
                agg[: cfg.session_max_agents_in_table]
                if cfg.session_max_agents_in_table > 0
                else agg
            )
            for s in agg_to_show:
                ok_rate = (
                    s.n_successes / s.n_invocations if s.n_invocations else 0.0
                )
                lines.append(
                    f"  {s.agent:<{agent_w}s} {s.n_invocations:>5} calls  "
                    f"{s.total_tokens:>9,} tok  {ok_rate:>5.1%} ok"
                )
                if s.error_reasons and cfg.include_agent_error_detail:
                    detail = ", ".join(
                        f"{r}={n}" for r, n in s.error_reasons.items()
                    )
                    lines.append(f"      ↳ errors: {detail}")
            if (
                cfg.session_max_agents_in_table > 0
                and len(agg) > cfg.session_max_agents_in_table
            ):
                hidden = len(agg) - cfg.session_max_agents_in_table
                lines.append(f"  … {hidden} more agent(s) omitted")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_runs": self.n_runs,
            "status_counts": self.status_counts,
            "total_tokens": self.total_tokens,
            "total_runtime_seconds": self.total_runtime_seconds,
            "n_governance_halts": self.n_governance_halts,
            "per_agent_aggregate": [
                asdict(s) for s in self.per_agent_aggregate
            ],
            "runs": [r.to_dict() for r in self.runs],
        }
