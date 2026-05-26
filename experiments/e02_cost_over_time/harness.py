"""
Cost-over-time experiment harness.

Runs each Category B task K times sequentially with a shared persistent
policy graph. Captures per-run metrics and aggregates the trajectory.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from agensflow import PolicyGraph, run

from experiments.e01_regime_validation.grader import Verdict, grade
from experiments.e01_regime_validation.tasks import CATEGORY_B_TASKS, BenchmarkTask


@dataclass
class TrajectoryCell:
    """One (task_id, run_index) outcome."""

    task_id: str
    run_index: int  # 1-based
    answer: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    validation_retries: int
    latency_seconds: float
    n_calls: int
    regime_used: str | None
    judgement: str
    grader_rationale: str
    reward: float | None
    policy_path: list[str]
    policy_graph_size: int | None
    n_confident_nodes: int | None
    error: str | None = None


def _extract_path_actions(policy_path: Any) -> list[str]:
    if not policy_path:
        return []
    return [action for _, action in policy_path]


def run_task_trajectory(
    task: BenchmarkTask,
    *,
    policy_graph: PolicyGraph,
    n_runs: int,
    confidence_threshold: int = 3,
    max_steps: int = 12,
) -> list[TrajectoryCell]:
    """
    Run `task` `n_runs` times sequentially through the shared policy graph.
    Returns one TrajectoryCell per run.
    """
    cells: list[TrajectoryCell] = []
    for run_index in range(1, n_runs + 1):
        print(f"  [{task.id}] run {run_index:>2d}/{n_runs} ...",
              end=" ", flush=True)
        cell = _run_single_cell(
            task,
            run_index=run_index,
            policy_graph=policy_graph,
            confidence_threshold=confidence_threshold,
            max_steps=max_steps,
        )
        cells.append(cell)
        # Concise per-run line so progress is visible during the run.
        print(
            f"{cell.judgement:>8s}  "
            f"{cell.total_tokens:>5d} tok  "
            f"r={(cell.reward if cell.reward is not None else 0.0):+.3f}  "
            f"graph={cell.policy_graph_size or 0}n/"
            f"{cell.n_confident_nodes or 0}c  "
            f"{cell.latency_seconds:>5.2f}s"
            + (f"  ERROR" if cell.error else "")
        )
    return cells


def _run_single_cell(
    task: BenchmarkTask,
    *,
    run_index: int,
    policy_graph: PolicyGraph,
    confidence_threshold: int,
    max_steps: int,
) -> TrajectoryCell:
    start = time.monotonic()
    try:
        result = run(
            user_task=task.user_task,
            features=task.features,
            documents=task.documents,
            policy_graph=policy_graph,
            confidence_threshold=confidence_threshold,
            max_steps=max_steps,
        )

        verdict: Verdict = grade(
            user_task=task.user_task,
            documents=task.documents,
            answer=result.final_answer,
            category=task.category,
            ground_truth_answer=task.ground_truth_answer,
            grading_notes=task.grading_notes,
        )

        confident = sum(
            1 for n in policy_graph.nodes.values() if n.visits >= confidence_threshold
        )

        return TrajectoryCell(
            task_id=task.id,
            run_index=run_index,
            answer=result.final_answer,
            total_tokens=result.total_tokens,
            prompt_tokens=result.trace.total_prompt_tokens,
            completion_tokens=result.trace.total_completion_tokens,
            validation_retries=sum(
                1 for e in result.trace.events if e.error is not None
            ),
            latency_seconds=result.total_latency_seconds,
            n_calls=len(result.trace.events),
            regime_used=result.plan.regime.label,
            judgement=verdict.judgement,
            grader_rationale=verdict.rationale,
            reward=result.reward,
            policy_path=_extract_path_actions(result.policy_path),
            policy_graph_size=result.policy_graph_size,
            n_confident_nodes=confident,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        tb = traceback.format_exc(limit=2)
        return TrajectoryCell(
            task_id=task.id,
            run_index=run_index,
            answer="",
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            validation_retries=0,
            latency_seconds=elapsed,
            n_calls=0,
            regime_used=None,
            judgement="failure",
            grader_rationale="(execution error)",
            reward=None,
            policy_path=[],
            policy_graph_size=None,
            n_confident_nodes=None,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
        )


def cell_to_dict(cell: TrajectoryCell) -> dict[str, Any]:
    return asdict(cell)
