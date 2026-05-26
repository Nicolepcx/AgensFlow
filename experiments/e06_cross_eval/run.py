"""
Chunk 8.5 entry point — cross-condition RULER evaluation.

Compares chunk-7 (no-skip) vs chunk-8 (skip-on) trajectories *head-to-head*,
using three judges to control for same-family bias and within-condition
peer-group calibration drift.

Per-task workflow:

  1. Pull the last 2 epochs of successful runs for the task from chunk 7
     and from chunk 8 (up to 2 + 2 = 4 trajectories per task).
  2. Build a single RULER comparison group containing all 4.
  3. For each judge model, call `relative_judge_score_group` on that group.
  4. Record each trajectory's score from each judge, plus which condition
     "won" the head-to-head per judge.

Output:

  - results.jsonl                : one row per (task, judge) with the
                                   per-trajectory scores + win flag
  - aggregates.json              : win/loss/tie matrix per judge,
                                   per-class breakdown, inter-judge
                                   agreement
  - RESULTS.md                   : human-readable summary

Usage:

    python -m experiments.e06_cross_eval.run                          # full run
    python -m experiments.e06_cross_eval.run --tasks C1.1,C2.1        # subset
    python -m experiments.e06_cross_eval.run --judges haiku,gpt       # judge subset
    python -m experiments.e06_cross_eval.run --reset                  # fresh start

Cost estimate (full run): 59 tasks × 3 judges = 177 judge calls,
~12k input + ~2k output tokens each. Roughly $15-25 total.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from agensflow import (
    DEFAULT_RUBRIC,
    RelativeJudgeScoreGroup,
    TrajectoryToScore,
    relative_judge_score_group,
)
from agensflow.runtime.client import OpenRouterClient


THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
AGGREGATES_PATH = RESULTS_DIR / "aggregates.json"
RESULTS_PATH = RESULTS_DIR / "RESULTS.md"

# Source experiments. We pull trajectories from both.
CHUNK7_JSONL = (
    THIS_DIR.parent / "e04_sustained_traffic" / "results_agensflow.jsonl"
)
CHUNK8_JSONL = (
    THIS_DIR.parent / "e05_topology_skip" / "results_agensflow.jsonl"
)

# Judges. The original is intentionally first so its column lines up with
# the other experiments' RULER scores.
#
# Chunk-11 update: the chunk-6 caveats here were partially wrong. The
# original analysis concluded "qwen + gemini fail under Instructor's
# TOOLS mode" and dropped them entirely. The chunk-11 probe
# (`scripts/probe_qwen_judge.py`) showed that qwen-flash + qwen-max
# DO work — they need `Mode.JSON` + `extra_body={"provider":
# {"require_parameters": True}}`, both documented in the
# Instructor+OpenRouter integration guide. Similarly, grok-4.3 needs
# `Mode.JSON` (it doesn't comply with TOOLS-mode strict schema).
#
# Updated 3-family validated set (chunk-11.A2):
#   - anthropic/claude-haiku-4.5  TOOLS + extra_body{provider:require_parameters}
#   - openai/gpt-5.4-mini         TOOLS + NO extra_body (require_parameters
#                                 paradoxically breaks OpenAI's primary
#                                 OpenRouter route)
#   - qwen/qwen3.6-flash          JSON  + extra_body
#
# Each judge needs per-call (mode, extra_body) configuration. See
# `RelativeJudgeConfig.cross_judge_modes` + `cross_judge_extra_body` in
# `agensflow/learning/ruler/config.py`. Probe new judges with
# `scripts/probe_qwen_judge.py` before adding them.
#
# Failure modes seen in the probe (all distinguishable by error
# fingerprint):
#   1. OpenRouter routing 404 (instant, <0.1s) — qwen-all under TOOLS,
#      gemini-flash. Fix: Mode.JSON.
#   2. Model-side validation timeout (10-20s) — gemini-pro-preview.
#      No known fix; gemini-pro genuinely doesn't populate the schema.
#   3. Schema compliance refusal — grok-4.3 under TOOLS (won't populate
#      axis_scores). Fix: Mode.JSON; grok-4.3 does comply under JSON.
DEFAULT_JUDGES = {
    "haiku":  "anthropic/claude-haiku-4.5",
    "gpt":    "openai/gpt-5.4",
    "sonnet": "anthropic/claude-sonnet-4.6",
    "grok":   "x-ai/grok-4.3",
}

# Per-judge (mode, extra_body) configuration for chunk-11 onwards. The
# e06 cross-eval methodology pre-dates chunk-11's per-judge plumbing,
# so this dict is informational here; downstream callers using
# `RelativeJudgeConfig.cross_judge_modes` + `cross_judge_extra_body` thread it
# through `relative_judge_score_group` correctly. See `replay_rescore.py` in
# e07_skill_variants for the canonical config example.
DEFAULT_JUDGE_CONFIGS = {
    "haiku":  {"mode": "tools", "extra_body": {"provider": {"require_parameters": True}}},
    "gpt":    {"mode": "tools", "extra_body": {}},   # explicit no extra_body
    "sonnet": {"mode": "tools", "extra_body": {"provider": {"require_parameters": True}}},
    "grok":   {"mode": "json",  "extra_body": {"provider": {"require_parameters": True}}},
}

# How many epochs of recent runs to pull per condition per task.
EPOCHS_PER_CONDITION = 2

RESULTS_BEGIN = "<!-- RESULTS:BEGIN (auto-generated; do not edit) -->"
RESULTS_END = "<!-- RESULTS:END -->"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _last_n_epoch_trajectories(
    records: list[dict[str, Any]],
    n_epochs: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Group records by task_id and return only the last `n_epochs` epochs of
    successful runs. Errored runs are dropped — we can't compare a missing
    answer head-to-head against a real one.
    """
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by_task[r["task_id"]].append(r)
    out: dict[str, list[dict[str, Any]]] = {}
    for task_id, recs in by_task.items():
        # Sort ascending by epoch then by run_index so "last n" is well defined.
        recs.sort(key=lambda r: (r.get("epoch", 0), r.get("run_index", 0)))
        if not recs:
            continue
        max_epoch = recs[-1].get("epoch", 0)
        cutoff = max_epoch - n_epochs + 1
        out[task_id] = [r for r in recs if r.get("epoch", 0) >= cutoff]
    return out


def _path_summary(path: list[str] | None) -> str:
    if not path:
        return "(no path)"
    return " → ".join(path)


def _to_trajectory_score(
    rec: dict[str, Any],
    condition_label: str,
) -> TrajectoryToScore:
    """One record → one TrajectoryToScore. ID encodes condition + epoch + run."""
    tid = (
        f"{condition_label}#ep{rec.get('epoch', 0)}"
        f"#run{rec.get('run_index', 0)}#{rec['task_id']}"
    )
    return TrajectoryToScore(
        trajectory_id=tid,
        final_answer=rec.get("final_answer", ""),
        path_summary=_path_summary(rec.get("path")),
    )


# --------------------------------------------------------------------------- #
# Per-task evaluation
# --------------------------------------------------------------------------- #


def _eval_one_task(
    *,
    task_id: str,
    user_task: str,
    chunk7_recs: list[dict[str, Any]],
    chunk8_recs: list[dict[str, Any]],
    judge_models: dict[str, str],
    client: OpenRouterClient,
) -> dict[str, Any]:
    """
    Score the cross-condition group for one task with each judge.

    Returns a dict with the per-judge results and the head-to-head winner
    decision, plus the raw per-trajectory scores.
    """
    trajectories: list[TrajectoryToScore] = []
    for r in chunk7_recs:
        trajectories.append(_to_trajectory_score(r, "chunk7"))
    for r in chunk8_recs:
        trajectories.append(_to_trajectory_score(r, "chunk8"))

    if len(trajectories) < 2:
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": (
                f"Insufficient trajectories: chunk7 has {len(chunk7_recs)}, "
                f"chunk8 has {len(chunk8_recs)}; need ≥2 total."
            ),
        }

    # Tasks at this point have a record from at least one condition. Check that
    # both conditions are represented — otherwise the head-to-head is moot.
    if not chunk7_recs or not chunk8_recs:
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": (
                f"Asymmetric: chunk7 has {len(chunk7_recs)}, "
                f"chunk8 has {len(chunk8_recs)}. Need both for head-to-head."
            ),
        }

    judge_results: dict[str, dict[str, Any]] = {}
    for short_name, model_id in judge_models.items():
        start = time.monotonic()
        try:
            group: RelativeJudgeScoreGroup = relative_judge_score_group(
                user_task=user_task,
                trajectories=trajectories,
                client=client,
                judge_model=model_id,
                rubric=DEFAULT_RUBRIC,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=2)
            judge_results[short_name] = {
                "model": model_id,
                "error": f"{type(exc).__name__}: {exc}\n{tb}",
                "elapsed": time.monotonic() - start,
            }
            continue

        # Per-trajectory scores, separated by condition.
        per_traj: dict[str, float] = {
            tid: res.score for tid, res in group.scores.items()
        }
        per_traj_explanation: dict[str, str] = {
            tid: res.explanation for tid, res in group.scores.items()
        }

        chunk7_scores = [
            per_traj[t.trajectory_id]
            for t in trajectories
            if t.trajectory_id.startswith("chunk7#")
            and t.trajectory_id in per_traj
        ]
        chunk8_scores = [
            per_traj[t.trajectory_id]
            for t in trajectories
            if t.trajectory_id.startswith("chunk8#")
            and t.trajectory_id in per_traj
        ]
        c7_mean = mean(chunk7_scores) if chunk7_scores else 0.0
        c8_mean = mean(chunk8_scores) if chunk8_scores else 0.0
        if abs(c8_mean - c7_mean) < 0.05:  # tie threshold
            winner = "tie"
        elif c8_mean > c7_mean:
            winner = "chunk8"
        else:
            winner = "chunk7"

        judge_results[short_name] = {
            "model": model_id,
            "scores": per_traj,
            "explanations": per_traj_explanation,
            "chunk7_mean": c7_mean,
            "chunk8_mean": c8_mean,
            "delta": c8_mean - c7_mean,
            "winner": winner,
            "judge_tokens": group.judge_tokens,
            "judge_latency": group.judge_latency_seconds,
            "elapsed": time.monotonic() - start,
        }

    return {
        "task_id": task_id,
        "n_chunk7": len(chunk7_recs),
        "n_chunk8": len(chunk8_recs),
        "judges": judge_results,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-judge win/loss/tie matrix + per-class breakdown + agreement."""
    judges_seen = sorted({
        j for r in rows if not r.get("skipped") for j in r["judges"]
    })

    per_judge: dict[str, Any] = {}
    for j in judges_seen:
        wins: dict[str, int] = Counter()
        for r in rows:
            if r.get("skipped"):
                continue
            jr = r["judges"].get(j)
            if jr is None or jr.get("error"):
                continue
            wins[jr["winner"]] += 1
        per_judge[j] = {
            "chunk8_wins": wins.get("chunk8", 0),
            "chunk7_wins": wins.get("chunk7", 0),
            "ties": wins.get("tie", 0),
            "total_scored": sum(wins.values()),
        }

    # Per-class breakdown for chunk-8 win rate.
    per_class_per_judge: dict[str, dict[str, dict[str, int]]] = {}
    for r in rows:
        if r.get("skipped"):
            continue
        cls = r["task_id"].split(".")[0]  # "C2.1" → "C2"
        for j, jr in r["judges"].items():
            if jr.get("error"):
                continue
            per_class_per_judge.setdefault(cls, {}).setdefault(j, Counter())
            per_class_per_judge[cls][j][jr["winner"]] += 1

    # Inter-judge agreement: for each task, how many judges agreed on the
    # winner? Higher = more robust head-to-head signal.
    agreement_dist: Counter[int] = Counter()
    for r in rows:
        if r.get("skipped"):
            continue
        winners = [
            jr["winner"] for jr in r["judges"].values()
            if not jr.get("error")
        ]
        if not winners:
            continue
        majority = Counter(winners).most_common(1)[0][1]
        agreement_dist[majority] += 1

    return {
        "per_judge": per_judge,
        "per_class": {
            cls: {j: dict(c) for j, c in per_judge_dict.items()}
            for cls, per_judge_dict in per_class_per_judge.items()
        },
        "agreement": dict(agreement_dist),
        "n_tasks_scored": sum(
            1 for r in rows if not r.get("skipped")
            and any(not jr.get("error") for jr in r["judges"].values())
        ),
        "n_skipped": sum(1 for r in rows if r.get("skipped")),
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_results(agg: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(RESULTS_BEGIN)
    lines.append("")
    lines.append("## Cross-evaluation results")
    lines.append("")
    lines.append(
        f"Tasks scored: {agg['n_tasks_scored']}, "
        f"skipped (asymmetric coverage): {agg['n_skipped']}"
    )
    lines.append("")
    lines.append("### Per-judge head-to-head (chunk-8 vs chunk-7)")
    lines.append("")
    lines.append(
        "| judge | chunk-8 wins | chunk-7 wins | ties | chunk-8 win rate |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for judge_name, d in agg["per_judge"].items():
        total = d["total_scored"]
        win_rate = d["chunk8_wins"] / total if total else 0.0
        lines.append(
            f"| {judge_name} | {d['chunk8_wins']} | {d['chunk7_wins']} | "
            f"{d['ties']} | {win_rate:.0%} |"
        )
    lines.append("")
    lines.append("**Tie threshold: |Δ score| < 0.05.**")
    lines.append("")

    lines.append("### Inter-judge agreement on the winner")
    lines.append("")
    lines.append(
        "Distribution of *majority size* across tasks (3 judges per task)."
    )
    lines.append("")
    lines.append("| majority size | n tasks | meaning |")
    lines.append("|---:|---:|---|")
    agree = agg["agreement"]
    if agree.get(3):
        lines.append(f"| 3 | {agree.get(3, 0)} | unanimous agreement |")
    if agree.get(2):
        lines.append(f"| 2 | {agree.get(2, 0)} | 2 of 3 agree |")
    if agree.get(1):
        lines.append(f"| 1 | {agree.get(1, 0)} | all three disagree (split decision) |")
    lines.append("")

    lines.append("### Per-class breakdown")
    lines.append("")
    lines.append(
        "| class | judge | chunk-8 wins | chunk-7 wins | ties |"
    )
    lines.append("|---|---|---:|---:|---:|")
    for cls in sorted(agg["per_class"].keys()):
        for judge_name, c in agg["per_class"][cls].items():
            lines.append(
                f"| {cls} | {judge_name} | {c.get('chunk8', 0)} | "
                f"{c.get('chunk7', 0)} | {c.get('tie', 0)} |"
            )
    lines.append("")
    lines.append(RESULTS_END)
    return "\n".join(lines)


def _splice_results(rendered: str) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(_predictions_stub() + "\n\n" + rendered + "\n")
        return
    text = RESULTS_PATH.read_text()
    if RESULTS_BEGIN in text and RESULTS_END in text:
        before, _, rest = text.partition(RESULTS_BEGIN)
        _, _, after = rest.partition(RESULTS_END)
        new_text = before.rstrip() + "\n\n" + rendered + "\n" + after.lstrip()
    else:
        new_text = text.rstrip() + "\n\n" + rendered + "\n"
    RESULTS_PATH.write_text(new_text)


def _predictions_stub() -> str:
    return """# Experiment 06 — Cross-condition quality evaluation

Pre-registered notes, written before the run. Predictions are preserved
across runs; only the auto-generated results section is overwritten.

## Purpose

Test whether chunk-8 (skip-on)'s headline +0.02 RULER lift over chunk-7
(no-skip) survives independent scrutiny. Per-condition RULER ranks each
condition's trajectories against its own peer group; this experiment scores
the two conditions' trajectories *together*, head-to-head, against a single
rubric, with three judges across three model families.

## Setup

- For each of 59 tasks: pull last 2 epochs of successful runs from chunk 7
  and chunk 8 (~2+2 = up to 4 trajectories per task).
- Build a single RULER group containing all 4 trajectories.
- Score with three judges in parallel:
  - `anthropic/claude-haiku-4.5` (same family as the original chunk-7/8 judge)
  - `openai/gpt-5.4`              (cross-family check, OpenAI)
  - `google/gemini-3.1-pro-preview` (cross-family check, Google)
- Compute per-trajectory scores, condition means, head-to-head winner,
  and inter-judge agreement.

## What "the chunk-8 quality claim survives" looks like

- chunk-8 wins or ties ≥50% of head-to-heads under all 3 judges
- Inter-judge agreement ≥2/3 on a majority of tasks (i.e. judges aren't
  randomly disagreeing)
- Per-class breakdown shows wins concentrated in the *expected* classes
  (C1, C6, C8 — simple extraction) and competitive (not catastrophic) on
  the hard classes (C5, C7)

## What would falsify the chunk-8 quality claim

- chunk-7 wins majority head-to-heads under cross-family judges
  (`gpt-5.4`, `gemini-3.1-pro-preview`) even if `claude-haiku-4.5` favors
  chunk-8 — that would diagnose same-family bias as the source of the
  +0.02 RULER lift
- Catastrophic chunk-7 wins on C5/C7 (the hard classes), suggesting
  chunk-8's skip-mechanism cuts material it shouldn't cut
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-condition RULER evaluation: chunk-7 vs chunk-8."
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task ids to score (default: all 59).",
    )
    parser.add_argument(
        "--judges", type=str, default=",".join(DEFAULT_JUDGES.keys()),
        help=(
            f"Comma-separated judge short-names (default: "
            f"{','.join(DEFAULT_JUDGES.keys())}). Available: "
            f"{','.join(DEFAULT_JUDGES.keys())}."
        ),
    )
    parser.add_argument(
        "--epochs-per-condition", type=int, default=EPOCHS_PER_CONDITION,
        help=f"How many recent epochs of trajectories to pull from each "
             f"condition (default {EPOCHS_PER_CONDITION}).",
    )
    parser.add_argument("--reset", action="store_true",
                        help="Delete results.jsonl + aggregates before starting.")
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write to RESULTS.md; print instead.")
    args = parser.parse_args()

    judges_to_use = {
        name: model
        for name, model in DEFAULT_JUDGES.items()
        if name in set(args.judges.split(","))
    }
    if not judges_to_use:
        print(f"No judges selected; available: {list(DEFAULT_JUDGES)}",
              file=sys.stderr)
        return 2

    if args.reset:
        for p in [RESULTS_JSONL, AGGREGATES_PATH]:
            if p.exists():
                p.unlink()
                print(f"Reset: removed {p}")

    # Load source records.
    chunk7_recs = _load_jsonl(CHUNK7_JSONL)
    chunk8_recs = _load_jsonl(CHUNK8_JSONL)
    if not chunk7_recs:
        print(f"ERROR: no records at {CHUNK7_JSONL}", file=sys.stderr)
        return 1
    if not chunk8_recs:
        print(f"ERROR: no records at {CHUNK8_JSONL}", file=sys.stderr)
        return 1

    chunk7_by_task = _last_n_epoch_trajectories(
        chunk7_recs, args.epochs_per_condition
    )
    chunk8_by_task = _last_n_epoch_trajectories(
        chunk8_recs, args.epochs_per_condition
    )

    # Pick tasks.
    if args.tasks:
        wanted = set(args.tasks.split(","))
    else:
        wanted = set(chunk7_by_task.keys()) | set(chunk8_by_task.keys())

    # Need both conditions to have at least one record for the task.
    eligible_tasks = sorted(
        t for t in wanted
        if t in chunk7_by_task and t in chunk8_by_task
    )
    print(f"Cross-eval setup:")
    print(f"  judges:                 {list(judges_to_use)}")
    print(f"  epochs per condition:   {args.epochs_per_condition}")
    print(f"  tasks with both arms:   {len(eligible_tasks)}")
    print(f"  total judge calls:      {len(eligible_tasks) * len(judges_to_use)}")
    print()

    # Build a task_id → user_task lookup. Use chunk-8's records since they're
    # the most recent; both conditions used the same task pool.
    user_task_lookup: dict[str, str] = {}
    for r in chunk8_recs + chunk7_recs:
        if r["task_id"] not in user_task_lookup:
            # Records don't carry the user_task string directly; pull it
            # from the source task pool.
            pass
    # Source-of-truth: load from tasks module.
    from experiments.e03_production_traffic.tasks import ALL_TASKS
    task_text_lookup = {t.id: t.user_task for t in ALL_TASKS}

    load_dotenv()
    client = OpenRouterClient()

    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for i, tid in enumerate(eligible_tasks, start=1):
        user_task = task_text_lookup.get(tid)
        if user_task is None:
            print(f"  [{tid}] SKIP — task not found in ALL_TASKS")
            continue
        c7r = chunk7_by_task.get(tid, [])
        c8r = chunk8_by_task.get(tid, [])
        print(f"  [{tid}] {i}/{len(eligible_tasks)}  "
              f"chunk7×{len(c7r)} vs chunk8×{len(c8r)} ...",
              end=" ", flush=True)
        t0 = time.monotonic()
        row = _eval_one_task(
            task_id=tid,
            user_task=user_task,
            chunk7_recs=c7r,
            chunk8_recs=c8r,
            judge_models=judges_to_use,
            client=client,
        )
        rows.append(row)
        # Append to JSONL incrementally so a long run is crash-safe.
        with RESULTS_JSONL.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # Print per-judge winner summary line.
        if row.get("skipped"):
            print(f"skipped: {row.get('reason', '?')}  ({time.monotonic()-t0:.1f}s)")
        else:
            verdicts = []
            for j, jr in row["judges"].items():
                if jr.get("error"):
                    verdicts.append(f"{j}=ERR")
                else:
                    verdicts.append(f"{j}={jr['winner']}")
            print(f"{', '.join(verdicts)}  ({time.monotonic()-t0:.1f}s)")

    elapsed = time.monotonic() - start
    print(f"\nFinished {len(rows)} tasks in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    # Aggregate + write outputs.
    agg = _aggregate(rows)
    AGGREGATES_PATH.write_text(json.dumps(agg, indent=2))

    rendered = _render_results(agg)
    if args.no_write:
        print()
        print(rendered)
    else:
        _splice_results(rendered)
        print(f"\nResults written to {RESULTS_PATH}")
        print(f"Aggregates at {AGGREGATES_PATH}")

    # Print the headline at the end for convenience.
    print()
    print("=== Headline ===")
    for j, d in agg["per_judge"].items():
        total = d["total_scored"]
        wr = d["chunk8_wins"] / total if total else 0.0
        print(f"  {j:<8} chunk-8 {d['chunk8_wins']}, chunk-7 {d['chunk7_wins']}, "
              f"ties {d['ties']}  →  chunk-8 win rate {wr:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
