"""
Production-traffic harness for chunk 6.

Runs each task once through the AgensFlow runtime with the shared persistent
policy graph and the chunk-6 activation plan (5 solver variants + 2 verifier
variants + corpus-memory + 2 web search providers). After each run:

  1. Reconstruct the trajectory and add it to the per-scenario-class
     rolling RULER buffer.
  2. Score the new trajectory via RULER against the prior K-1 trajectories
     of the same class — apples-to-apples relative ranking within class.
  3. Compute the hybrid reward: RULER score (primary anchor) + cost penalty
     + retry penalty.
  4. Backup the reward through the policy graph manually (run() was called
     with defer_backup=True).

This is the closed loop: policy chooses a route, RULER judges the
trajectory's quality relative to recent peers in the same class, the reward
flows back into the policy graph, the next run informs from the accumulated
value estimates.
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from agensflow import (
    PolicyGraph,
    RewardConfig,
    RewardInputs,
    RelativeJudgeScoreGroup,
    TrajectoryEvidence,
    TrajectoryToScore,
    build_trajectory_evidence,
    compute_hybrid_reward,
    run,
    relative_judge_score_group,
)
from agensflow.learning.relative_judge import RelativeJudgeConfig
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.runner import _record_failures_to_graph

from experiments.e03_production_traffic.activation import (
    build_chunk6_activation_plan,
)
from experiments.e03_production_traffic.tasks import ProductionTask


# How many recent trajectories per scenario class to keep for RULER comparison.
# K=4 gives meaningful relative ranking once 4 trajectories of a class have
# accumulated; runs 1-3 of each class score against fewer peers (RULER falls
# back to neutral 0.5 for groups of 1).
ROLLING_BUFFER_SIZE = 4

# Default judge model for RULER. Different family from the variant pool's
# fast tiers, to limit same-model bias on the rubric.
DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-4.5"


@dataclass
class TrajectoryRecord:
    """One run's recorded trajectory + per-run telemetry."""

    task_id: str
    scenario_class: str
    run_index: int  # 1-based, cumulative across the whole experiment
    final_answer: str
    path: list[str]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    validation_retries: int
    latency_seconds: float
    n_calls: int
    regime_used: str | None
    expected_optimal_variant: str
    expected_verifier_value: str
    expected_web_value: str
    corpus_has_answer: bool
    # RULER + hybrid reward fields (filled in after scoring).
    ruler_score: float | None = None
    ruler_explanation: str = ""
    ruler_group_size: int = 0
    judge_tokens: int = 0
    hybrid_reward: float | None = None
    # Chunk 11.A2/A4: cross-judge disagreement → confidence telemetry.
    # Single-judge runs report confidence=1.0 + zero disagreement.
    # Cross-judge runs report population std + range across judges and
    # the derived confidence used to weight the backup gradient.
    ruler_confidence: float = 1.0
    ruler_disagreement_std: float = 0.0
    ruler_disagreement_range: float = 0.0
    # The reward AFTER confidence weighting (i.e. the value that was
    # actually backed up to the policy graph). When confidence=1.0 this
    # equals hybrid_reward; when <1.0, it's `hybrid_reward * confidence`.
    backed_up_reward: float | None = None
    # Chunk 11.A3: per-axis rubric scoring. When the judge returns
    # axis_scores, these fields surface the cross-judge per-axis means
    # AND per-axis disagreement std. Lets downstream analysis answer
    # "which axis did this trajectory excel/struggle on, and how much
    # did the judges agree?" Empty when the rubric/judge skipped axes
    # (chunk-2..10 holistic-only path).
    ruler_axis_scores: dict[str, float] = field(default_factory=dict)
    ruler_per_axis_disagreement_std: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    # Multi-epoch experiments (chunk 7) annotate which sweep this record came
    # from. Defaults to 0 so chunk-6 records remain backward-compatible
    # without rewrites; chunk-7 records use 1..N.
    epoch: int = 0
    # Governance fields. `governance_halted=True` means the run terminated
    # because GovernancePolicy was breached (e.g. agent failing repeatedly,
    # AUTH/QUOTA error). When True, policy-graph backup was SKIPPED for
    # this run so infrastructure failures don't pollute the substrate.
    # `governance_violation` carries the structured violation as a dict
    # (JSON-serializable) for the RunReport / dashboards.
    governance_halted: bool = False
    governance_violation: dict[str, Any] | None = None


def _path_summary(path: list[str]) -> str:
    return " → ".join(path) if path else "(no path)"


def _trajectory_to_score(
    task: ProductionTask,
    record: TrajectoryRecord,
    evidence: TrajectoryEvidence | None = None,
) -> TrajectoryToScore:
    """Build a TrajectoryToScore from a record. When `evidence` is
    supplied (chunk 11.A1+), it's attached for the judge to see
    structured per-agent contributions; when None, falls back to the
    chunk-2..10 path-summary-only flow."""
    return TrajectoryToScore(
        trajectory_id=f"{task.id}#run{record.run_index}",
        final_answer=record.final_answer,
        path_summary=_path_summary(record.path),
        evidence=evidence,
    )


@dataclass
class HarnessState:
    """Mutable state shared across runs in one full benchmark."""

    policy_graph: PolicyGraph
    client: OpenRouterClient
    reward_config: RewardConfig
    judge_model: str
    rubric: str
    # RULER configuration (chunk 11.A1). When evidence_mode != "off",
    # the harness builds TrajectoryEvidence per run and attaches it to
    # the rolling buffer so RULER's judge sees structured per-agent
    # contributions. Defaults to a fresh `RelativeJudgeConfig()` for backward
    # compat with chunk-9 callers that don't supply one.
    ruler_config: RelativeJudgeConfig = field(default_factory=RelativeJudgeConfig)
    # Backup discount factor (chunk 11.C1). 1.0 (default) preserves
    # chunk-2..10 undiscounted behavior; <1.0 weights earlier path
    # decisions less than later ones during backup.
    backup_gamma: float = 1.0
    # Rolling buffer of (task, trajectory, evidence) per scenario class.
    # Evidence is None when the run was scored before A1 wiring, or
    # when `ruler_config.evidence_mode == "off"`.
    class_buffers: dict[
        str,
        deque[tuple[ProductionTask, TrajectoryRecord, TrajectoryEvidence | None]],
    ] = field(default_factory=dict)

    def buffer_for(
        self, scenario_class: str,
    ) -> deque[tuple[ProductionTask, TrajectoryRecord, TrajectoryEvidence | None]]:
        if scenario_class not in self.class_buffers:
            self.class_buffers[scenario_class] = deque(maxlen=ROLLING_BUFFER_SIZE)
        return self.class_buffers[scenario_class]


# --------------------------------------------------------------------------- #
# Run report writer
# --------------------------------------------------------------------------- #


def _write_run_report(
    *,
    task: ProductionTask,
    trace: "Any | None",
    governance_state: "Any | None",
    record: TrajectoryRecord,
    status: str,
    policy_graph_backed_up: bool,
    report_dir: "Any | None",
) -> None:
    """Build a RunReport from the run's artifacts and persist it to disk.

    Called on every terminal path of run_one_task (success, governance
    halt, generic exception). The report aggregates trace events into
    per-agent summaries and pulls the governance violations from the
    state — so even a halted run leaves a structured forensic artifact
    next to the JSONL trajectory record.

    No-op when report_dir is None (chunks 6/7/8 don't enable reports).
    """
    if report_dir is None or trace is None:
        return
    import json
    from pathlib import Path
    from agensflow.runtime.report import RunReport

    report = RunReport.from_run_artifacts(
        task_id=task.id,
        trace=trace,
        governance_state=governance_state,
        status=status,  # type: ignore[arg-type]
        policy_graph_backed_up=policy_graph_backed_up,
        ruler_score=record.ruler_score,
        hybrid_reward=record.hybrid_reward,
        metadata={
            "scenario_class": task.scenario_class,
            "epoch": record.epoch,
            "run_index": record.run_index,
            "expected_optimal_variant": task.expected_optimal_variant,
        },
    )
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (
        f"run_report_ep{record.epoch:02d}"
        f"_run{record.run_index:04d}_{task.id}.json"
    )
    with path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Per-run execution
# --------------------------------------------------------------------------- #


def run_one_task(
    task: ProductionTask,
    *,
    state: HarnessState,
    run_index: int,
    max_steps: int = 14,
    confidence_threshold: int = 5,
    epoch: int = 0,
    reliability_weight: float = 0.5,
    enable_skip: bool = False,
    plan_builder: "Any | None" = None,
    enable_router_logging: bool = False,
    router_log_dir: "Any | None" = None,
    governance_policy: "Any | None" = None,
    report_dir: "Any | None" = None,
) -> TrajectoryRecord:
    """
    Execute one task, score it via RULER against recent same-class peers,
    backup the resulting hybrid reward to the policy graph.
    """
    print(f"  [{task.id}] {task.scenario_class} ...", end=" ", flush=True)
    start = time.monotonic()
    # Construct trace + governance state OUTSIDE run() so the harness
    # holds references that survive exceptions. The trace's router_log
    # gets populated when enable_router_logging is on. The governance
    # state accumulates per-agent activity; on BrokenAgentError, the
    # harness builds a RunReport from these even though run() raised
    # before returning a RunResult.
    from agensflow.runtime.trace import TraceCollector
    captured_trace = TraceCollector() if (
        enable_router_logging or report_dir is not None
        or governance_policy is not None
    ) else None
    captured_governance = None
    if governance_policy is not None:
        from agensflow.runtime.governance import GovernanceState
        captured_governance = GovernanceState(policy=governance_policy)
    try:
        # Build the activation plan for this task. Default = chunk-6 plan
        # (full variant pool); chunk-9 passes its own `plan_builder` to
        # expose the (skill × model) action space instead.
        if plan_builder is not None:
            plan = plan_builder(task.features)
        else:
            plan = build_chunk6_activation_plan(task.features)

        result = run(
            user_task=task.user_task,
            features=task.features,
            documents=task.documents,
            regime=plan.regime,
            policy_graph=state.policy_graph,
            max_steps=max_steps,
            confidence_threshold=confidence_threshold,
            reliability_weight=reliability_weight,
            enable_skip=enable_skip,
            enable_router_logging=enable_router_logging,
            defer_backup=True,
            plan=plan,
            trace=captured_trace,
            # Pass the externally-constructed state, not the policy.
            # This way, even if run() raises BrokenAgentError, the
            # harness still has a reference to inspect the violations.
            governance_state=captured_governance,
        )

        path = [a for _, a in (result.policy_path or [])]
        n_retries = sum(1 for e in result.trace.events if e.error is not None)

        record = TrajectoryRecord(
            task_id=task.id,
            scenario_class=task.scenario_class,
            run_index=run_index,
            final_answer=result.final_answer,
            path=path,
            total_tokens=result.total_tokens,
            prompt_tokens=result.trace.total_prompt_tokens,
            completion_tokens=result.trace.total_completion_tokens,
            validation_retries=n_retries,
            latency_seconds=result.total_latency_seconds,
            n_calls=len(result.trace.events),
            regime_used=result.plan.regime.label,
            expected_optimal_variant=task.expected_optimal_variant,
            expected_verifier_value=task.expected_verifier_value,
            expected_web_value=task.expected_web_value,
            corpus_has_answer=task.corpus_has_answer,
            epoch=epoch,
        )

        # Chunk 11.A1: build structured TrajectoryEvidence from the
        # trace events when configured. The harness gates on
        # `state.ruler_config.evidence_mode` so a "off" mode skips the
        # build (legacy chunk-2..10 reproduction stays cheap). The
        # evidence is then carried through the rolling buffer alongside
        # the record so prior trajectories also score against the
        # structured form when they're re-judged for new comparisons.
        evidence_for_new: TrajectoryEvidence | None = None
        if state.ruler_config.evidence_mode != "off":
            evidence_for_new = build_trajectory_evidence(
                result.trace.events, config=state.ruler_config,
            )

        # RULER scoring: build a group from this run's trajectory + the
        # rolling buffer of prior same-class trajectories.
        buffer = state.buffer_for(task.scenario_class)
        prior = list(buffer)
        group_trajectories: list[TrajectoryToScore] = [
            _trajectory_to_score(t, r, e) for (t, r, e) in prior
        ]
        new_to_score = _trajectory_to_score(task, record, evidence_for_new)
        group_trajectories.append(new_to_score)

        ruler_result: RelativeJudgeScoreGroup = relative_judge_score_group(
            user_task=task.user_task,
            trajectories=group_trajectories,
            client=state.client,
            judge_model=state.judge_model,
            rubric=state.rubric,
            config=state.ruler_config,
        )

        # Pull this trajectory's score out of the group.
        new_score_obj = ruler_result.scores.get(new_to_score.trajectory_id)
        if new_score_obj is not None:
            record.ruler_score = new_score_obj.score
            record.ruler_explanation = new_score_obj.explanation
            # Chunk 11.A2: capture cross-judge telemetry on the record.
            record.ruler_confidence = new_score_obj.confidence
            record.ruler_disagreement_std = new_score_obj.disagreement_std
            record.ruler_disagreement_range = new_score_obj.disagreement_range
            # Chunk 11.A3: capture per-axis means + per-axis std for
            # downstream "which axis did this trajectory excel on"
            # analysis. Empty dicts when judge ran holistic-only.
            record.ruler_axis_scores = dict(new_score_obj.axis_scores)
            record.ruler_per_axis_disagreement_std = dict(
                new_score_obj.per_axis_disagreement_std
            )
        else:
            record.ruler_score = 0.5  # neutral fallback
            record.ruler_explanation = "(no score returned for this trajectory)"
            record.ruler_confidence = 0.0  # No score → zero confidence
        record.ruler_group_size = len(group_trajectories)
        record.judge_tokens = ruler_result.judge_tokens

        # Compute hybrid reward and backup to policy graph manually.
        reward_inputs = RewardInputs(
            done=False,  # ignored by hybrid reward
            verification_str=None,  # ignored by hybrid reward
            total_tokens=record.total_tokens,
            n_validation_retries=record.validation_retries,
        )
        reward = compute_hybrid_reward(
            ruler_score=record.ruler_score,
            inputs=reward_inputs,
            config=state.reward_config,
        )
        record.hybrid_reward = reward

        # Chunk 11.A4: confidence-weighted backup. When the cross-judge
        # disagreement is high (low confidence), the substrate barely
        # updates from this run — both the reward and any implicit
        # penalties shrink proportionally, so low-confidence runs
        # contribute less to the policy gradient without introducing a
        # cost-only bias. Single-judge runs have confidence=1.0 →
        # weighted_reward == hybrid_reward (chunk-2..10 behavior).
        weighted_reward = reward * record.ruler_confidence
        record.backed_up_reward = weighted_reward

        # Backup reward through the (signature, action) path.
        if result.policy_path:
            # Chunk-9: feed per-action tokens into Welford variance
            # tracking. Compute from the trace by summing total_tokens
            # per agent name across this run's events. Errors / skip
            # events have total_tokens=0 so they don't pollute the
            # action's token mean.
            action_tokens: dict[str, int] = {}
            for ev in result.trace.events:
                if ev.error is not None:
                    continue  # validation retries — skip
                action_tokens[ev.agent] = action_tokens.get(ev.agent, 0) + ev.total_tokens
            state.policy_graph.backup(
                result.policy_path, weighted_reward,
                action_tokens=action_tokens,
                gamma=state.backup_gamma,
            )
            # Also remember edges so the graph carries topology info.
            for i in range(len(result.policy_path) - 1):
                from_sig, action = result.policy_path[i]
                to_sig = result.policy_path[i + 1][0]
                state.policy_graph.record_transition(from_sig, action, to_sig)

        # Mechanism A+C: tally per-edge validation failures so UCB's
        # reliability term can downweight unreliable variants over time.
        # `defer_backup=True` skips the runner's automatic call, so the
        # harness owns this side of the bookkeeping too.
        _record_failures_to_graph(
            policy_graph=state.policy_graph,
            trace_events=result.trace.events,
            regime_label=result.plan.regime.label,
        )

        # Update the rolling buffer with the (task, record, evidence)
        # triple. Evidence is None when evidence_mode is "off".
        buffer.append((task, record, evidence_for_new))

        elapsed = time.monotonic() - start
        path_str = _path_summary(path)
        print(
            f"ruler={record.ruler_score:.2f} reward={reward:+.2f}  "
            f"tok={record.total_tokens}  retries={n_retries}  "
            f"path: {path_str[:70]}  ({elapsed:.1f}s)"
        )
        # Persist a per-task RunReport (Layer 6) when report_dir is set.
        # Trace was passed externally so result.trace IS captured_trace
        # when captured_trace was constructed.
        _write_run_report(
            task=task,
            trace=captured_trace if captured_trace is not None else result.trace,
            governance_state=captured_governance,
            record=record,
            status="completed",
            policy_graph_backed_up=True,
            report_dir=report_dir,
        )
        return record

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        tb = traceback.format_exc(limit=2)

        # ----- Governance halt (BrokenAgentError) — special-case ----- #
        #
        # When the GovernancePolicy fires, we DO NOT want to backup the
        # in-flight failure data to the policy graph (would pollute the
        # substrate with infrastructure-level noise). We DO want to
        # surface the structured violation cleanly so the user knows
        # exactly what to fix.
        from agensflow.runtime.governance import BrokenAgentError
        if isinstance(exc, BrokenAgentError):
            v = exc.violation
            print(
                f"  ⛔ HALT  {elapsed:.1f}s  "
                f"(governance: {v.reason} on {v.agent})"
            )
            if v.suggested_fix:
                print(f"     → {v.suggested_fix}")
            from dataclasses import asdict
            halt_record = TrajectoryRecord(
                task_id=task.id,
                scenario_class=task.scenario_class,
                run_index=run_index,
                final_answer="",
                path=[],
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                validation_retries=0,
                latency_seconds=elapsed,
                n_calls=0,
                regime_used=None,
                expected_optimal_variant=task.expected_optimal_variant,
                expected_verifier_value=task.expected_verifier_value,
                expected_web_value=task.expected_web_value,
                corpus_has_answer=task.corpus_has_answer,
                error=f"GovernanceHalt: {v.reason} on {v.agent} — {v.detail}",
                epoch=epoch,
                governance_halted=True,
                governance_violation=asdict(v),
            )
            _write_run_report(
                task=task, trace=captured_trace,
                governance_state=captured_governance,
                record=halt_record, status="halted_by_policy",
                policy_graph_backed_up=False,  # substrate protected
                report_dir=report_dir,
            )
            return halt_record

        # ----- Generic exception path (existing behavior) ----- #
        print(f"  ERROR  {elapsed:.1f}s  ({type(exc).__name__})")
        # Dump the captured router log to disk if we have one. This is
        # the chunk-9 forensic artifact: when run() raised before
        # returning a RunResult, the trace lives only in `captured_trace`,
        # and its router_log shows what the router was attempting on
        # each iteration of the while-loop until LangGraph's recursion
        # ceiling fired.
        if (
            captured_trace is not None
            and router_log_dir is not None
            and captured_trace.router_log
        ):
            import json
            from pathlib import Path
            log_dir = Path(router_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / (
                f"router_log_ep{epoch:02d}_run{run_index:04d}_{task.id}.json"
            )
            with log_path.open("w") as f:
                json.dump({
                    "task_id": task.id,
                    "epoch": epoch,
                    "run_index": run_index,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "n_iterations": len(captured_trace.router_log),
                    "iterations": captured_trace.router_log,
                }, f, indent=2, default=str)
            print(f"    [router log → {log_path}]")
        err_record = TrajectoryRecord(
            task_id=task.id,
            scenario_class=task.scenario_class,
            run_index=run_index,
            final_answer="",
            path=[],
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            validation_retries=0,
            latency_seconds=elapsed,
            n_calls=0,
            regime_used=None,
            expected_optimal_variant=task.expected_optimal_variant,
            expected_verifier_value=task.expected_verifier_value,
            expected_web_value=task.expected_web_value,
            corpus_has_answer=task.corpus_has_answer,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
            epoch=epoch,
        )
        _write_run_report(
            task=task, trace=captured_trace,
            governance_state=captured_governance,
            record=err_record, status="errored",
            policy_graph_backed_up=False,
            report_dir=report_dir,
        )
        return err_record


# --------------------------------------------------------------------------- #
# Full benchmark orchestration
# --------------------------------------------------------------------------- #


def run_full_benchmark(
    tasks: list[ProductionTask],
    *,
    state: HarnessState,
    max_steps: int = 14,
    confidence_threshold: int = 5,
    epoch: int = 0,
    run_index_offset: int = 0,
    on_record: "Any | None" = None,
    reliability_weight: float = 0.5,
    enable_skip: bool = False,
    plan_builder: "Any | None" = None,
    enable_router_logging: bool = False,
    router_log_dir: "Any | None" = None,
    governance_policy: "Any | None" = None,
    report_dir: "Any | None" = None,
) -> list[TrajectoryRecord]:
    """
    Run all tasks sequentially through the shared policy graph.

    Multi-epoch experiments (chunk 7) call this once per epoch with a
    growing `run_index_offset` so the cumulative run_index keeps climbing
    across the experiment. `epoch` is stamped onto each record.

    `on_record` is an optional callback invoked after each record is
    produced; chunk-7's runner uses this to flush JSONL incrementally so
    a long-running experiment is crash-safe.
    """
    records: list[TrajectoryRecord] = []
    for i, task in enumerate(tasks, start=1):
        cumulative = run_index_offset + i
        print(f"=== run {cumulative} (epoch {epoch}, {i}/{len(tasks)}) ===")
        record = run_one_task(
            task,
            state=state,
            run_index=cumulative,
            max_steps=max_steps,
            confidence_threshold=confidence_threshold,
            epoch=epoch,
            reliability_weight=reliability_weight,
            enable_skip=enable_skip,
            plan_builder=plan_builder,
            enable_router_logging=enable_router_logging,
            router_log_dir=router_log_dir,
            governance_policy=governance_policy,
            report_dir=report_dir,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)
        print()
    return records


def record_to_dict(record: TrajectoryRecord) -> dict[str, Any]:
    return asdict(record)
