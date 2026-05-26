"""
Multi-agent retry-stack baseline.

This is the "production-shape" baseline AgensFlow's economic claim is
actually competing against — the way teams *currently* build MAS without
a learnable policy:

  - planner → solver → verifier as a fixed sequential pipeline
  - retry-on-failure between stages, capped at MAX_RETRIES per stage
  - every task uses the same (capable) solver — no per-task variant choice
  - every task invokes the verifier — no per-task verification skip
  - no policy learning across runs

Each task runs through this baseline once, with RULER scoring (against the
same prior trajectories AgensFlow uses) so the comparison is apples-to-apples.

This isn't a strawman — it's the real production pattern. Teams cap retries
at 2-3, use the strongest model they can afford for the solver, always run
the verifier because they don't know which queries need it. The framework's
claim is that learned routing reduces token cost vs. this pattern at
equivalent quality.
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from agensflow import (
    Document,
    Handoff,
    RewardConfig,
    RewardInputs,
    RelativeJudgeScoreGroup,
    TaskFeatures,
    TrajectoryToScore,
    compute_hybrid_reward,
    detect_regime,
    relative_judge_score_group,
)
from agensflow.runtime.agents import (
    make_evaluator,
    make_memory,
    make_planner,
    make_solver,
    make_verifier,
)
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.trace import TraceCollector

from experiments.e03_production_traffic.tasks import ProductionTask


# Retry caps mirroring common production patterns.
MAX_PLANNER_RETRIES = 2
MAX_SOLVER_RETRIES = 3
MAX_VERIFIER_RETRIES = 2

# Baseline uses the most-capable solver in the *actively-tested* variant
# pool — production teams that don't have routing infrastructure pick the
# strongest model they can afford to avoid losing on quality. Comparing
# learned routing against the cheapest model would be unfair to the
# framework; comparing against the most capable is the honest test of
# "is it worth routing at all."
#
# We use `solver_haiku` rather than `solver_qwen_max` because the Qwen
# variants are excluded from chunk-6's plan (OpenRouter tool_choice
# compatibility — see activation.py for context). Using a variant the
# AgensFlow path can also reach keeps the comparison apples-to-apples.
BASELINE_SOLVER_SKILL = "solver_haiku"
BASELINE_VERIFIER_SKILL = "verifier_haiku"


@dataclass
class BaselineRecord:
    """One baseline run's record — same shape as TrajectoryRecord for parity."""

    task_id: str
    scenario_class: str
    run_index: int
    final_answer: str
    path: list[str]  # the fixed pipeline stages actually executed
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    n_retries_total: int  # validation retries summed across all stages
    n_stage_retries: int  # how many *stages* needed any retry
    latency_seconds: float
    n_calls: int
    expected_optimal_variant: str
    expected_verifier_value: str
    expected_web_value: str
    corpus_has_answer: bool
    ruler_score: float | None = None
    ruler_explanation: str = ""
    ruler_group_size: int = 0
    judge_tokens: int = 0
    hybrid_reward: float | None = None
    error: str | None = None


def _trajectory_to_score(
    task: ProductionTask, record: BaselineRecord
) -> TrajectoryToScore:
    return TrajectoryToScore(
        trajectory_id=f"{task.id}#baseline_run{record.run_index}",
        final_answer=record.final_answer,
        path_summary=" → ".join(record.path),
    )


def _try_stage(
    *,
    name: str,
    factory_fn,
    state: Handoff,
    max_retries: int,
) -> tuple[Handoff, int, str | None]:
    """
    Run a stage with retry-on-failure. Returns (new_state, retries_used, error).

    "Failure" here means a Python-level exception from the agent factory's
    node (which itself uses Instructor's bounded validation retry internally).
    A stage-level retry calls the entire node afresh.
    """
    last_error: str | None = None
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            update = factory_fn(state)
            new_state = state.model_copy(update=update)
            return new_state, retries, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            retries = attempt + 1  # we used `attempt+1` attempts that failed
            if attempt < max_retries:
                continue
    return state, retries, last_error


def run_baseline_for_task(
    task: ProductionTask,
    *,
    client: OpenRouterClient,
    run_index: int,
    rolling_buffer: deque[tuple[ProductionTask, BaselineRecord]],
    reward_config: RewardConfig,
    judge_model: str,
    rubric: str,
) -> BaselineRecord:
    """Execute the baseline pipeline for one task."""
    print(f"  [{task.id}] {task.scenario_class} BASELINE ...", end=" ", flush=True)
    start = time.monotonic()
    trace = TraceCollector()

    try:
        # Build the fixed pipeline: planner → memory → capable solver → verifier → evaluator.
        planner = make_planner(client, task.user_task, trace)
        memory = make_memory(client, task.documents, trace)
        solver = make_solver(client, trace, skill_name=BASELINE_SOLVER_SKILL)
        verifier = make_verifier(client, trace, skill_name=BASELINE_VERIFIER_SKILL)
        evaluator = make_evaluator(client, trace)

        state = Handoff()
        path: list[str] = []
        n_retries = 0
        n_stage_retries = 0

        for name, fn, max_retries in [
            ("planner", planner, MAX_PLANNER_RETRIES),
            ("memory", memory, 1),  # memory rarely needs retries
            (BASELINE_SOLVER_SKILL, solver, MAX_SOLVER_RETRIES),
            (BASELINE_VERIFIER_SKILL, verifier, MAX_VERIFIER_RETRIES),
            ("evaluator", evaluator, 1),
        ]:
            state, stage_retries, err = _try_stage(
                name=name, factory_fn=fn, state=state, max_retries=max_retries,
            )
            path.append(name)
            if stage_retries > 0:
                n_retries += stage_retries
                n_stage_retries += 1
            if err is not None:
                # Stage exhausted retries; bail. The trajectory still gets
                # scored — its quality will be low, which is the honest
                # behavior of a retry-stack baseline that runs out of attempts.
                break

        evaluator_output = state.metadata.get("evaluator", {}) if state.metadata else {}
        final_answer = (
            evaluator_output.get("final_answer", "")
            or state.draft_answer
            or ""
        )

        record = BaselineRecord(
            task_id=task.id,
            scenario_class=task.scenario_class,
            run_index=run_index,
            final_answer=final_answer,
            path=path,
            total_tokens=trace.total_tokens,
            prompt_tokens=trace.total_prompt_tokens,
            completion_tokens=trace.total_completion_tokens,
            n_retries_total=n_retries,
            n_stage_retries=n_stage_retries,
            latency_seconds=trace.total_latency_seconds,
            n_calls=len(trace.events),
            expected_optimal_variant=task.expected_optimal_variant,
            expected_verifier_value=task.expected_verifier_value,
            expected_web_value=task.expected_web_value,
            corpus_has_answer=task.corpus_has_answer,
        )

        # RULER scoring against the rolling buffer.
        prior = list(rolling_buffer)
        group = [_trajectory_to_score(t, r) for t, r in prior]
        new_score = _trajectory_to_score(task, record)
        group.append(new_score)

        ruler_result: RelativeJudgeScoreGroup = relative_judge_score_group(
            user_task=task.user_task,
            trajectories=group,
            client=client,
            judge_model=judge_model,
            rubric=rubric,
        )
        new_obj = ruler_result.scores.get(new_score.trajectory_id)
        if new_obj is not None:
            record.ruler_score = new_obj.score
            record.ruler_explanation = new_obj.explanation
        else:
            record.ruler_score = 0.5
            record.ruler_explanation = "(no score returned)"
        record.ruler_group_size = len(group)
        record.judge_tokens = ruler_result.judge_tokens

        record.hybrid_reward = compute_hybrid_reward(
            ruler_score=record.ruler_score,
            inputs=RewardInputs(
                done=False,
                verification_str=None,
                total_tokens=record.total_tokens,
                n_validation_retries=record.n_retries_total,
            ),
            config=reward_config,
        )

        rolling_buffer.append((task, record))

        elapsed = time.monotonic() - start
        print(
            f"ruler={record.ruler_score:.2f} reward={record.hybrid_reward:+.2f}  "
            f"tok={record.total_tokens}  retries={n_retries}  ({elapsed:.1f}s)"
        )
        return record

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        tb = traceback.format_exc(limit=2)
        print(f"  ERROR  {elapsed:.1f}s  ({type(exc).__name__})")
        return BaselineRecord(
            task_id=task.id,
            scenario_class=task.scenario_class,
            run_index=run_index,
            final_answer="",
            path=[],
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            n_retries_total=0,
            n_stage_retries=0,
            latency_seconds=elapsed,
            n_calls=0,
            expected_optimal_variant=task.expected_optimal_variant,
            expected_verifier_value=task.expected_verifier_value,
            expected_web_value=task.expected_web_value,
            corpus_has_answer=task.corpus_has_answer,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
        )


def baseline_record_to_dict(record: BaselineRecord) -> dict[str, Any]:
    return asdict(record)
