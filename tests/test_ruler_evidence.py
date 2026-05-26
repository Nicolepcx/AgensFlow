"""
Tests for chunk 11.A1 — budgeted TrajectoryEvidence + judge prompt rendering.

Covers:
  - TrajectoryEvidence + SolverContribution + VerifierContribution dataclass shape.
  - `_render_evidence_section` budget honoring (truncation, top-K).
  - `_render_evidence_section` "full" mode = no truncation.
  - `_build_judge_prompt` falls back to path_summary when evidence is None
    (legacy backward-compat path).
  - `_build_judge_prompt` ALSO renders evidence when present.
  - `build_trajectory_evidence(events)` extracts the right per-agent fields
    from a synthetic trace and skips skip:X / errored events.
  - Re-export at top-level `agensflow` resolves.
"""

from __future__ import annotations

import json

from agensflow.learning.relative_judge import (
    RelativeJudgeConfig,
    SolverContribution,
    TrajectoryEvidence,
    TrajectoryToScore,
    VerifierContribution,
    build_trajectory_evidence,
)
from agensflow.learning.relative_judge.core import (
    _build_judge_prompt,
    _render_evidence_section,
    _truncate,
)
from agensflow.runtime.trace import TraceEvent


# --------------------------------------------------------------------------- #
# _truncate
# --------------------------------------------------------------------------- #


class TestTruncate:
    def test_under_cap_unchanged(self) -> None:
        assert _truncate("hello", 10) == "hello"

    def test_at_cap_unchanged(self) -> None:
        assert _truncate("hello", 5) == "hello"

    def test_over_cap_truncated(self) -> None:
        out = _truncate("hello world", 5)
        assert out.startswith("hello")
        assert "truncated" in out

    def test_zero_cap_means_no_truncation(self) -> None:
        # Convention: max_chars <= 0 disables truncation. Used by "full" mode.
        big = "x" * 100_000
        assert _truncate(big, 0) == big
        assert _truncate(big, -1) == big


# --------------------------------------------------------------------------- #
# _render_evidence_section — budgeted vs full
# --------------------------------------------------------------------------- #


class TestRenderEvidenceBudgeted:
    def _evidence(self) -> TrajectoryEvidence:
        return TrajectoryEvidence(
            planner={"goal": "g", "subproblem": "s", "constraints": ["c1", "c2"]},
            memory_evidence=["snippet1", "snippet2", "snippet3", "snippet4", "snippet5"],
            solvers=[
                SolverContribution(
                    skill_name="solver_concise_haiku",
                    model="anthropic/claude-haiku-4.5",
                    draft="A" * 5000,  # over default 2000-char cap
                ),
                SolverContribution(
                    skill_name="solver_cot_fast",
                    model="openai/gpt-5.4-nano",
                    draft="short draft",
                ),
            ],
            verifiers=[
                VerifierContribution(
                    skill_name="verifier_haiku",
                    verdict="supported",
                    reasoning="all claims grounded",
                ),
            ],
            evaluator={"done": True, "reasoning": "looks good"},
        )

    def test_topk_caps_memory_snippets(self) -> None:
        cfg = RelativeJudgeConfig(evidence_mode="budgeted", evidence_topk=2)
        lines = _render_evidence_section(self._evidence(), cfg)
        memory_lines = [l for l in lines if l.strip().startswith("- ")]
        assert len(memory_lines) == 2  # only top-2 of 5 shipped

    def test_topk_header_shows_K_and_total(self) -> None:
        cfg = RelativeJudgeConfig(evidence_mode="budgeted", evidence_topk=2)
        lines = _render_evidence_section(self._evidence(), cfg)
        # Memory header: "Memory (top-2 of 5):"
        memory_header = next(l for l in lines if l.startswith("Memory"))
        assert "top-2 of 5" in memory_header

    def test_solver_drafts_truncated(self) -> None:
        cfg = RelativeJudgeConfig(
            evidence_mode="budgeted", solver_draft_max_chars=100,
        )
        lines = _render_evidence_section(self._evidence(), cfg)
        solver_lines = [l for l in lines if "solver_concise_haiku" in l]
        assert len(solver_lines) == 1
        # 100 chars + "… [truncated]" marker — total ~115 chars max.
        assert "truncated" in solver_lines[0]
        assert len(solver_lines[0]) < 200

    def test_short_draft_not_truncated(self) -> None:
        cfg = RelativeJudgeConfig(
            evidence_mode="budgeted", solver_draft_max_chars=100,
        )
        lines = _render_evidence_section(self._evidence(), cfg)
        solver_lines = [l for l in lines if "solver_cot_fast" in l]
        assert "truncated" not in solver_lines[0]
        assert "short draft" in solver_lines[0]

    def test_planner_evaluator_capped(self) -> None:
        cfg = RelativeJudgeConfig(
            evidence_mode="budgeted",
            evidence_max_chars_per_agent=20,  # tiny, forces truncation
        )
        ev = self._evidence()
        lines = _render_evidence_section(ev, cfg)
        planner_line = next(l for l in lines if l.startswith("Planner:"))
        assert "truncated" in planner_line


class TestRenderEvidenceFull:
    def test_full_mode_no_truncation(self) -> None:
        # In "full" mode every cap is bypassed.
        cfg = RelativeJudgeConfig(
            evidence_mode="full",
            evidence_topk=1,                  # ignored
            solver_draft_max_chars=10,        # ignored
            evidence_max_chars_per_agent=10,  # ignored
        )
        ev = TrajectoryEvidence(
            memory_evidence=["a", "b", "c", "d"],
            solvers=[SolverContribution(
                skill_name="solver_x",
                model="m",
                draft="X" * 1000,
            )],
        )
        lines = _render_evidence_section(ev, cfg)
        # All 4 memory snippets ship, no truncation marker on solver.
        memory_items = [l for l in lines if l.strip().startswith("- ")]
        assert len(memory_items) == 4
        solver_line = next(l for l in lines if "solver_x" in l)
        assert "truncated" not in solver_line


# --------------------------------------------------------------------------- #
# _build_judge_prompt — backward compat + evidence flow
# --------------------------------------------------------------------------- #


class TestBuildJudgePrompt:
    def test_legacy_path_summary_only(self) -> None:
        # When evidence is None, the prompt must NOT contain any of the
        # structured-evidence section headers (Planner:, Memory, etc.).
        traj = TrajectoryToScore(
            trajectory_id="t1",
            final_answer="Paris",
            path_summary="planner → memory → solver",
        )
        out = _build_judge_prompt("Where is the Eiffel Tower?", "rubric...", [traj])
        assert "Coordination path: planner → memory → solver" in out
        assert "Planner:" not in out
        assert "Memory (top-" not in out
        assert "Solver attempts" not in out

    def test_evidence_included_when_present(self) -> None:
        ev = TrajectoryEvidence(
            planner={"goal": "find_the_tower"},
            memory_evidence=["paris_doc", "europe_doc"],
            solvers=[SolverContribution(
                skill_name="solver_haiku", model="anthropic/x", draft="Paris.",
            )],
        )
        traj = TrajectoryToScore(
            trajectory_id="t1",
            final_answer="Paris",
            path_summary="planner → memory → solver",
            evidence=ev,
        )
        out = _build_judge_prompt("Where?", "rubric", [traj])
        assert "Planner:" in out
        assert "Memory (top-" in out
        assert "solver_haiku" in out
        assert "Paris." in out

    def test_mixed_trajectories_render_correctly(self) -> None:
        # Some trajectories carry evidence, others only path_summary.
        # The prompt must handle both gracefully in one call.
        t_legacy = TrajectoryToScore(
            trajectory_id="t_legacy",
            final_answer="A",
            path_summary="legacy_path",
        )
        t_evidence = TrajectoryToScore(
            trajectory_id="t_evidence",
            final_answer="B",
            path_summary="evidence_path",
            evidence=TrajectoryEvidence(planner={"goal": "g"}),
        )
        out = _build_judge_prompt("task", "rubric", [t_legacy, t_evidence])
        assert "legacy_path" in out
        assert "evidence_path" in out
        # Planner: section appears once (only for the evidence trajectory).
        assert out.count("Planner:") == 1


# --------------------------------------------------------------------------- #
# build_trajectory_evidence — extraction from trace events
# --------------------------------------------------------------------------- #


def _make_event(agent: str, output_update: dict, *, model: str = "m", error: str | None = None) -> TraceEvent:
    return TraceEvent(
        agent=agent, model=model,
        input_state={}, output_update=output_update,
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        latency_seconds=0.0, error=error,
    )


class TestBuildTrajectoryEvidence:
    def test_extracts_planner(self) -> None:
        events = [_make_event("planner", {
            "goal": "g", "subproblem": "s", "constraints": ["c1"],
        })]
        ev = build_trajectory_evidence(events)
        assert ev.planner == {"goal": "g", "subproblem": "s", "constraints": ["c1"]}

    def test_extracts_memory_evidence(self) -> None:
        events = [_make_event("memory", {
            "evidence": ["snip1", "snip2", "snip3"],
        })]
        ev = build_trajectory_evidence(events)
        assert ev.memory_evidence == ["snip1", "snip2", "snip3"]

    def test_extracts_solver_drafts(self) -> None:
        events = [
            _make_event("solver_concise_haiku", {"draft_answer": "answer1"},
                        model="anthropic/claude-haiku-4.5"),
            _make_event("solver_cot_fast", {"draft_answer": "answer2"},
                        model="openai/gpt-5.4-nano"),
        ]
        ev = build_trajectory_evidence(events)
        assert len(ev.solvers) == 2
        names = [s.skill_name for s in ev.solvers]
        assert "solver_concise_haiku" in names
        assert "solver_cot_fast" in names
        # Models propagate from the trace event.
        haiku = next(s for s in ev.solvers if s.skill_name == "solver_concise_haiku")
        assert haiku.model == "anthropic/claude-haiku-4.5"

    def test_extracts_verifier_verdict(self) -> None:
        verifier_output = json.dumps({
            "verdict": "supported", "reasoning": "all claims OK",
        })
        events = [_make_event("verifier_haiku", {"verification": verifier_output})]
        ev = build_trajectory_evidence(events)
        assert len(ev.verifiers) == 1
        assert ev.verifiers[0].verdict == "supported"
        assert ev.verifiers[0].reasoning == "all claims OK"

    def test_extracts_evaluator(self) -> None:
        events = [_make_event("evaluator", {
            "metadata": {"evaluator": {
                "done": True, "reasoning": "looks complete",
            }},
        })]
        ev = build_trajectory_evidence(events)
        assert ev.evaluator == {"done": True, "reasoning": "looks complete"}

    def test_skips_skip_events(self) -> None:
        # `skip:X` events have no content for the judge to reason about.
        events = [
            _make_event("skip:verifier_haiku", {}),
            _make_event("planner", {"goal": "g"}),
        ]
        ev = build_trajectory_evidence(events)
        assert ev.planner == {"goal": "g", "subproblem": None, "constraints": []}
        # No verifiers extracted from the skip event.
        assert ev.verifiers == []

    def test_skips_errored_events(self) -> None:
        # Failed-attempt trace events have output_update like
        # {"_validation_error": "..."}; they shouldn't contribute content.
        events = [
            _make_event("solver_concise_haiku", {"_validation_error": "bad"},
                        error="ValidationError"),
            _make_event("solver_concise_haiku", {"draft_answer": "good"},
                        error=None),
        ]
        ev = build_trajectory_evidence(events)
        # Only the successful attempt contributes a SolverContribution.
        assert len(ev.solvers) == 1
        assert ev.solvers[0].draft == "good"

    def test_empty_events_yields_empty_evidence(self) -> None:
        ev = build_trajectory_evidence([])
        assert ev.planner is None
        assert ev.memory_evidence == []
        assert ev.solvers == []
        assert ev.verifiers == []
        assert ev.evaluator is None

    def test_malformed_verifier_json_silently_skipped(self) -> None:
        # Defensive: a verifier whose output isn't valid JSON shouldn't
        # crash extraction. Skip it; let the rest of the trace go through.
        events = [
            _make_event("verifier_haiku", {"verification": "not-valid-json{"}),
            _make_event("planner", {"goal": "g"}),
        ]
        ev = build_trajectory_evidence(events)
        assert ev.verifiers == []  # malformed → skipped
        assert ev.planner == {"goal": "g", "subproblem": None, "constraints": []}


# --------------------------------------------------------------------------- #
# Top-level re-exports
# --------------------------------------------------------------------------- #


class TestPublicReExports:
    def test_top_level_agensflow_namespace(self) -> None:
        # The new symbols should propagate up to `agensflow.X` per the
        # canonical pattern. Tests this once at the import boundary.
        from agensflow import (
            TrajectoryEvidence as Evid,
            SolverContribution as SC,
            VerifierContribution as VC,
            build_trajectory_evidence as bte,
        )
        assert Evid is TrajectoryEvidence
        assert SC is SolverContribution
        assert VC is VerifierContribution
        assert bte is build_trajectory_evidence
