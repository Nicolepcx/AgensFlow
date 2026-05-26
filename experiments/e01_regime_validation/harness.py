"""
Run-all harness.

For each (task, configuration) cell: execute the configuration, capture
metrics, grade the answer. Returns a structured result table.

The four configurations:
  1. naive  — single LLM call (claude-haiku-4.5).
  2. agensflow_forced_straightforward — full pipeline, regime forced.
  3. agensflow_forced_evidence_heavy  — full pipeline, regime forced.
  4. agensflow_auto                   — full pipeline, regime auto-detected
                                         from features.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agensflow import RegimeEstimate, RunResult, run
from agensflow.runtime.errors import InvalidAgentOutputError

from experiments.e01_regime_validation.baselines import NaiveResult, run_naive
from experiments.e01_regime_validation.grader import Verdict, grade
from experiments.e01_regime_validation.tasks import ALL_TASKS, BenchmarkTask

ConfigName = Literal[
    "naive",
    "agensflow_forced_straightforward",
    "agensflow_forced_evidence_heavy",
    "agensflow_auto",
]

CONFIGURATIONS: list[ConfigName] = [
    "naive",
    "agensflow_forced_straightforward",
    "agensflow_forced_evidence_heavy",
    "agensflow_auto",
]


@dataclass
class CellResult:
    """One (task, configuration) outcome."""

    task_id: str
    category: str
    configuration: ConfigName
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
    flagged_missing_evidence: bool
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _extract_run_result(rr: RunResult) -> dict[str, Any]:
    failed_events = [e for e in rr.trace.events if e.error is not None]
    return {
        "answer": rr.final_answer or rr.final_state.draft_answer or "",
        "total_tokens": rr.total_tokens,
        "prompt_tokens": rr.trace.total_prompt_tokens,
        "completion_tokens": rr.trace.total_completion_tokens,
        "validation_retries": len(failed_events),
        "latency_seconds": rr.total_latency_seconds,
        "n_calls": len(rr.trace.events),
        "regime_used": rr.plan.regime.label,
    }


def _run_naive_for_task(task: BenchmarkTask) -> dict[str, Any]:
    nr: NaiveResult = run_naive(
        user_task=task.user_task,
        documents=task.documents,
    )
    return {
        "answer": nr.answer,
        "total_tokens": nr.total_tokens,
        "prompt_tokens": nr.prompt_tokens,
        "completion_tokens": nr.completion_tokens,
        "validation_retries": 0,
        "latency_seconds": nr.latency_seconds,
        "n_calls": 1,
        "regime_used": None,
    }


def _run_agensflow_for_task(
    task: BenchmarkTask, *, forced_regime: str | None
) -> dict[str, Any]:
    regime: RegimeEstimate | None = None
    if forced_regime is not None:
        regime = RegimeEstimate(label=forced_regime, confidence=1.0)  # type: ignore[arg-type]
    rr: RunResult = run(
        user_task=task.user_task,
        features=task.features,
        documents=task.documents or None,
        regime=regime,
    )
    return _extract_run_result(rr)


def _execute_configuration(
    task: BenchmarkTask,
    config: ConfigName,
) -> dict[str, Any]:
    if config == "naive":
        return _run_naive_for_task(task)
    if config == "agensflow_forced_straightforward":
        return _run_agensflow_for_task(task, forced_regime="straightforward")
    if config == "agensflow_forced_evidence_heavy":
        return _run_agensflow_for_task(task, forced_regime="evidence_heavy")
    if config == "agensflow_auto":
        return _run_agensflow_for_task(task, forced_regime=None)
    raise ValueError(f"unknown configuration: {config}")


def run_cell(task: BenchmarkTask, config: ConfigName) -> CellResult:
    """Run one (task, configuration) cell and grade the result."""
    print(f"  [{config:<37s}] {task.id} ...", end=" ", flush=True)
    start = time.monotonic()
    try:
        run_data = _execute_configuration(task, config)
        verdict: Verdict = grade(
            user_task=task.user_task,
            documents=task.documents,
            answer=run_data["answer"],
            category=task.category,
            ground_truth_answer=task.ground_truth_answer,
            grading_notes=task.grading_notes,
        )
        elapsed = time.monotonic() - start
        print(
            f"{verdict.judgement:>8s}  "
            f"{run_data['total_tokens']:>5d} tok  "
            f"{run_data['n_calls']} call(s)  "
            f"{elapsed:.1f}s"
        )
        return CellResult(
            task_id=task.id,
            category=task.category,
            configuration=config,
            answer=run_data["answer"],
            total_tokens=run_data["total_tokens"],
            prompt_tokens=run_data["prompt_tokens"],
            completion_tokens=run_data["completion_tokens"],
            validation_retries=run_data["validation_retries"],
            latency_seconds=run_data["latency_seconds"],
            n_calls=run_data["n_calls"],
            regime_used=run_data["regime_used"],
            judgement=verdict.judgement,
            grader_rationale=verdict.rationale,
            flagged_missing_evidence=verdict.flagged_missing_evidence,
        )
    except (InvalidAgentOutputError, Exception) as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        tb = traceback.format_exc(limit=2)
        print(f"  ERROR  {elapsed:.1f}s  ({type(exc).__name__})")
        return CellResult(
            task_id=task.id,
            category=task.category,
            configuration=config,
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
            flagged_missing_evidence=False,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
        )


def run_full_benchmark(
    tasks: list[BenchmarkTask] | None = None,
    configurations: list[ConfigName] | None = None,
) -> list[CellResult]:
    """Run every (task, configuration) cell in sequence."""
    tasks = tasks or list(ALL_TASKS)
    configurations = configurations or list(CONFIGURATIONS)
    results: list[CellResult] = []
    for task in tasks:
        print(f"\n=== Task {task.id} (Category {task.category}) ===")
        for config in configurations:
            results.append(run_cell(task, config))
    return results


def cell_to_dict(cell: CellResult) -> dict[str, Any]:
    return asdict(cell)
