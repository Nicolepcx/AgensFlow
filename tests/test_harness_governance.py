"""
Tests for Layer 4 — harness integration of BrokenAgentError.

Verifies the harness's `run_one_task` correctly catches `BrokenAgentError`
*separately* from generic exceptions and:
  - Returns a TrajectoryRecord with `governance_halted=True`
  - Carries the structured violation as a JSON-serializable dict
  - Does NOT pollute the policy graph (skips the manual backup path)
  - Surfaces a clean error with the suggested fix in the trajectory

Mocks `run()` directly so we don't need real LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agensflow.runtime.governance import (
    AgentErrorReason,
    BrokenAgentError,
    GovernancePolicy,
    GovernanceViolation,
)


def _violation(agent: str = "web_search_exa") -> GovernanceViolation:
    return GovernanceViolation(
        timestamp=datetime.now(timezone.utc),
        agent=agent,
        reason="terminal_error:quota",
        detail=f"Agent {agent!r} failed with terminal error reason 'quota'.",
        policy_value=True,
        actual_value="quota",
        error_reason=AgentErrorReason.QUOTA,
        suggested_fix="Top up at dashboard.exa.ai.",
    )


def _make_minimal_task():
    """Construct a minimal ProductionTask for harness invocation."""
    from experiments.e03_production_traffic.tasks import ALL_TASKS
    # Reuse an existing task to avoid duplicating constructor surface.
    return ALL_TASKS[0]


def _make_minimal_state():
    """Construct a minimal HarnessState for harness invocation. The
    policy_graph is a real PolicyGraph so we can verify that NO edges
    were backed up after governance halt."""
    from agensflow import (
        DEFAULT_RUBRIC,
        PolicyGraph,
        RewardConfig,
    )
    from agensflow.runtime.client import OpenRouterClient
    from experiments.e03_production_traffic.harness import HarnessState

    graph = PolicyGraph()
    # Mock the client — harness calls it for RULER scoring; if our mocked
    # run() raises before that path, the client mock is never invoked.
    client = MagicMock(spec=OpenRouterClient)
    return HarnessState(
        policy_graph=graph,
        client=client,
        reward_config=RewardConfig(),
        judge_model="anthropic/claude-haiku-4.5",
        rubric=DEFAULT_RUBRIC,
    )


class TestHarnessGovernanceHalt:
    def test_broken_agent_error_yields_governance_halted_record(self) -> None:
        """BrokenAgentError raised inside run() should be caught by the
        harness's special-case handler. The returned TrajectoryRecord
        carries governance_halted=True + the structured violation."""
        from experiments.e03_production_traffic.harness import run_one_task

        task = _make_minimal_task()
        state = _make_minimal_state()

        with patch("experiments.e03_production_traffic.harness.run") as mock_run:
            mock_run.side_effect = BrokenAgentError(_violation())

            record = run_one_task(
                task, state=state, run_index=1, epoch=1,
                governance_policy=GovernancePolicy(),
            )

        assert record.governance_halted is True
        assert record.governance_violation is not None
        assert record.governance_violation["agent"] == "web_search_exa"
        assert record.governance_violation["reason"] == "terminal_error:quota"
        assert "Top up at dashboard.exa.ai" in record.governance_violation["suggested_fix"]
        # Error string surfaces the violation summary
        assert "GovernanceHalt" in (record.error or "")
        assert "web_search_exa" in (record.error or "")

    def test_governance_halt_does_not_backup_to_graph(self) -> None:
        """The whole point: governance halt MUST skip policy-graph backup
        so infrastructure-failure data doesn't pollute the substrate."""
        from experiments.e03_production_traffic.harness import run_one_task

        task = _make_minimal_task()
        state = _make_minimal_state()
        # Sanity: graph is empty before the run.
        assert len(state.policy_graph) == 0

        with patch("experiments.e03_production_traffic.harness.run") as mock_run:
            mock_run.side_effect = BrokenAgentError(_violation())
            run_one_task(
                task, state=state, run_index=1, epoch=1,
                governance_policy=GovernancePolicy(),
            )

        # Graph MUST still be empty — no edges added during a halted run.
        assert len(state.policy_graph) == 0

    def test_generic_exception_still_uses_legacy_path(self) -> None:
        """Non-governance exceptions should hit the existing generic
        except handler, NOT the new BrokenAgentError special case.
        Backward-compat: chunks 6/7/8 record-on-error semantics preserved."""
        from experiments.e03_production_traffic.harness import run_one_task

        task = _make_minimal_task()
        state = _make_minimal_state()

        with patch("experiments.e03_production_traffic.harness.run") as mock_run:
            mock_run.side_effect = ValueError("not a governance error")
            record = run_one_task(
                task, state=state, run_index=1, epoch=1,
            )

        assert record.governance_halted is False
        assert record.governance_violation is None
        assert "ValueError" in (record.error or "")
        assert "not a governance error" in (record.error or "")

    def test_governance_violation_dict_is_json_serializable(self) -> None:
        """The serialized violation must round-trip through JSON cleanly
        — the harness writes records to JSONL, and we don't want
        TrajectoryRecord serialization to choke on datetime or enum
        fields in the violation dict."""
        import json
        from experiments.e03_production_traffic.harness import run_one_task

        task = _make_minimal_task()
        state = _make_minimal_state()

        with patch("experiments.e03_production_traffic.harness.run") as mock_run:
            mock_run.side_effect = BrokenAgentError(_violation())
            record = run_one_task(
                task, state=state, run_index=1, epoch=1,
                governance_policy=GovernancePolicy(),
            )

        # asdict(GovernanceViolation) returns a dict with datetime + enum
        # values that need `default=str` to serialize. The harness's
        # jsonl-dumper already uses default=str; verify here for safety.
        s = json.dumps(record.governance_violation, default=str)
        # Round-trip works.
        parsed = json.loads(s)
        assert parsed["agent"] == "web_search_exa"
        assert parsed["reason"] == "terminal_error:quota"
