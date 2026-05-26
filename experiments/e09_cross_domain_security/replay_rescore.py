"""
Offline 3-judge re-score for e09 trajectories.

Re-scores any e09 `results_*.jsonl` under the validated chunk-11
cross-family judge ensemble + per-axis rubric, WITHOUT re-executing
any agents and WITHOUT updating any policy graph.

Why a separate judging pass:

  The main / ablation / baseline runs of e09 use the chunk-9 single
  judge (anthropic/claude-haiku-4.5) for substrate reward consistency.
  Single-judge RULER is the right reward signal during a sweep
  (cheap + stable for UCB updates), but it carries known judge bias
  and noise. For the *reported* cross-domain comparison — and for
  the prediction-#4 baseline-vs-main RULER tolerance test — we want
  the upgraded measurement stack from chunk-11:

    anthropic/claude-haiku-4.5  + openai/gpt-5.4-mini  + qwen/qwen3.6-flash

  Three families ⇒ tie-breaking + outlier detection in cross-judge
  averaging, plus per-axis decomposition.

This script is parametric over `--source-jsonl` / `--out` so a single
implementation handles every e09 arm:

  - main run                  results_agensflow.jsonl
  - no-skip ablation          ablation_no_skip/results_agensflow.jsonl
  - fixed-pipeline baseline   baseline_fixed/results_baseline.jsonl
  - warm-start arm            warm_start/results_agensflow.jsonl

Adapted from `experiments/e07_skill_variants/replay_rescore.py`.
Differences:
  - Source defaults to e09's main-run JSONL.
  - Task lookup pulls from e09's task pool (SecurityTask), not e03's.
  - Output paths land under e09's directory tree.
  - Summary skips the chunk-9-specific "epoch1→epoch8" verdict
    (since baseline and warm-start are single-epoch) — instead reports
    per-epoch deltas and the headline mean-delta vs original RULER.

Cost / time (same shape as e07): ~$15-30 / ~60-90 min per arm
(60 tasks × 8 epochs × 3 judges ≈ 1,440 calls; baseline is 60 × 3 ≈ 180).

Usage:

    # Main run (default source):
    python -m experiments.e09_cross_domain_security.replay_rescore

    # Ablation:
    python -m experiments.e09_cross_domain_security.replay_rescore \\
        --source-jsonl experiments/e09_cross_domain_security/ablation_no_skip/results_agensflow.jsonl \\
        --out experiments/e09_cross_domain_security/ablation_no_skip/replay_rescore_results.jsonl

    # Baseline:
    python -m experiments.e09_cross_domain_security.replay_rescore \\
        --source-jsonl experiments/e09_cross_domain_security/baseline_fixed/results_baseline.jsonl \\
        --out experiments/e09_cross_domain_security/baseline_fixed/replay_rescore_results.jsonl

    # Smoke test:
    python -m experiments.e09_cross_domain_security.replay_rescore --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from agensflow.config import load_config
from agensflow.learning.relative_judge import (
    DEFAULT_RUBRIC,
    TrajectoryToScore,
    relative_judge_score_group,
)
from agensflow.runtime.client import OpenRouterClient


THIS_DIR = Path(__file__).parent
DEFAULT_SOURCE = THIS_DIR / "results_agensflow.jsonl"
DEFAULT_OUT = THIS_DIR / "replay_rescore_results.jsonl"
DEFAULT_CONFIG = THIS_DIR / "example_config.yaml"

# Match the e03/e09 harness rolling buffer for apples-to-apples
# relative scoring (most recent 3 same-class trajectories + the new one).
ROLLING_BUFFER_SIZE = 4

# Validated chunk-11 cross-family judge triple.
# Modes + extra_body settings carried verbatim from e07's
# replay_rescore.py — these are the configs that survived the
# chunk-11 probe (qwen needs JSON mode; OpenAI's primary OpenRouter
# route breaks under require_parameters=True).
DEFAULT_CROSS_JUDGE = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "qwen/qwen3.6-flash",
]
DEFAULT_CROSS_JUDGE_MODES: dict[str, str] = {
    "qwen/qwen3.6-flash": "json",
}
DEFAULT_CROSS_JUDGE_EXTRA_BODY: dict[str, dict] = {
    "openai/gpt-5.4-mini": {},
}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Source JSONL not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_user_task_lookup() -> dict[str, str]:
    """Return {task_id → user_task} so the judge sees the original
    task string. Pulls from e09's task pool (SecurityTask)."""
    from experiments.e09_cross_domain_security.tasks import ALL_TASKS
    return {t.id: t.user_task for t in ALL_TASKS}


# --------------------------------------------------------------------------- #
# Re-scoring loop
# --------------------------------------------------------------------------- #


def _trajectory_id_of(record: dict[str, Any]) -> str:
    return f"{record['task_id']}#run{record['run_index']}"


def _replay(
    records: list[dict[str, Any]],
    *,
    cfg_rj,
    client: OpenRouterClient,
    task_lookup: dict[str, str],
    rubric: str,
    progress_every: int = 20,
) -> list[dict[str, Any]]:
    """Walk records in run_index order, replay the original rolling-
    buffer-of-4 scoring discipline per scenario class, return new
    score objects keyed by trajectory_id.
    """
    class_buffers: dict[str, deque[TrajectoryToScore]] = defaultdict(
        lambda: deque(maxlen=ROLLING_BUFFER_SIZE)
    )

    out: list[dict[str, Any]] = []
    sorted_records = sorted(records, key=lambda r: int(r.get("run_index", 0)))

    started = time.monotonic()
    n_judge_calls = 0
    for i, rec in enumerate(sorted_records, start=1):
        traj_id = _trajectory_id_of(rec)
        scenario_class = rec.get("scenario_class") or "unknown"
        task_id = rec.get("task_id") or ""
        user_task = task_lookup.get(task_id, "(task text unavailable)")
        path = rec.get("path") or []
        final_answer = rec.get("final_answer") or ""

        new_traj = TrajectoryToScore(
            trajectory_id=traj_id,
            final_answer=final_answer,
            path_summary=" → ".join(path) if path else "(no path)",
        )

        buffer = class_buffers[scenario_class]
        group = list(buffer) + [new_traj]

        try:
            result = relative_judge_score_group(
                user_task=user_task,
                trajectories=group,
                client=client,
                judge_model=cfg_rj.judge_model,  # ignored when cross_judge_models is set
                rubric=rubric,
                max_tokens=cfg_rj.max_tokens,
                config=cfg_rj,
            )
            n_judge_calls += max(1, len(cfg_rj.cross_judge_models or [1]))
            new_score = result.scores.get(traj_id)
        except Exception as exc:  # noqa: BLE001
            new_score = None
            print(f"  [{traj_id}] re-score ERROR: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

        original_score = rec.get("ruler_score")
        comparison: dict[str, Any] = {
            "trajectory_id": traj_id,
            "task_id": task_id,
            "scenario_class": scenario_class,
            "epoch": rec.get("epoch"),
            "run_index": rec.get("run_index"),
            "original_ruler_score": original_score,
            "original_judge": "anthropic/claude-haiku-4.5",  # e09 single-judge default
            "group_size": len(group),
            "path": path,
            "total_tokens": rec.get("total_tokens"),  # carry forward for cost/quality plots
        }
        if new_score is not None:
            judges_with_axes = [
                j for j, axes in (new_score.per_judge_axis_scores or {}).items()
                if axes
            ]
            comparison.update({
                "new_score": new_score.score,
                "new_explanation": new_score.explanation,
                "new_confidence": new_score.confidence,
                "new_disagreement_std": new_score.disagreement_std,
                "new_disagreement_range": new_score.disagreement_range,
                "new_per_judge_scores": new_score.per_judge_scores,
                "new_per_judge_axis_scores": new_score.per_judge_axis_scores,
                "new_axis_scores": new_score.axis_scores,
                "new_per_axis_disagreement_std": new_score.per_axis_disagreement_std,
                "new_judges": list(result.per_judge_models),
                "n_judges_returning_axes": len(judges_with_axes),
                "judges_returning_axes": judges_with_axes,
            })
            if original_score is not None:
                comparison["delta"] = new_score.score - original_score
        else:
            comparison["new_score"] = None
            comparison["error"] = "(scoring failed; see stderr)"

        out.append(comparison)
        buffer.append(new_traj)

        if i % progress_every == 0 or i == len(sorted_records):
            elapsed = time.monotonic() - started
            rate = i / max(1.0, elapsed)
            remaining = (len(sorted_records) - i) / max(1.0, rate)
            print(
                f"  [{i}/{len(sorted_records)}]  "
                f"judge_calls~{n_judge_calls}  "
                f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s",
                file=sys.stderr,
            )

    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _summarize(out: list[dict[str, Any]]) -> str:
    """Per-class + per-epoch summary of original vs new score, plus
    cross-judge axis-compliance telemetry."""
    lines: list[str] = []
    valid = [r for r in out if r.get("new_score") is not None
             and r.get("original_ruler_score") is not None]
    n_total = len(out)
    n_valid = len(valid)
    n_failed = n_total - n_valid

    lines.append("═" * 80)
    lines.append("e09 3-judge offline re-score — summary")
    lines.append("═" * 80)
    lines.append(f"Trajectories re-scored: {n_valid}/{n_total} (failed: {n_failed})")
    if not valid:
        lines.append("No valid comparisons; cannot summarize.")
        return "\n".join(lines)

    # Axis-compliance: which judges actually populated axis_scores?
    # Without this, axis_σ=0 looks like "judges agree" when it can
    # actually mean "only one judge returned axes."
    multi_traj_records = [r for r in valid if r.get("group_size", 0) >= 2]
    if multi_traj_records:
        compliance: dict[str, int] = defaultdict(int)
        seen_attempts: dict[str, int] = defaultdict(int)
        all_judges: set[str] = set()
        for r in multi_traj_records:
            judge_axis_map = r.get("new_per_judge_axis_scores") or {}
            for j in judge_axis_map:
                all_judges.add(j)
                seen_attempts[j] += 1
            for j in r.get("judges_returning_axes") or []:
                compliance[j] += 1
        lines.append("")
        lines.append(f"Judge axis-compliance ({len(multi_traj_records)} multi-trajectory groups):")
        for j in sorted(all_judges):
            attempts = max(seen_attempts.get(j, 0), 1)
            n = compliance.get(j, 0)
            pct = n / attempts * 100
            tag = " " if pct >= 80 else "  ⚠"
            lines.append(
                f"  {tag} {j:<40s}  {n:>3}/{attempts}  ({pct:.0f}%) returned axes"
            )

    # Headline delta.
    deltas = [r["new_score"] - r["original_ruler_score"] for r in valid]
    delta_mean = mean(deltas)
    orig_mean = mean(r["original_ruler_score"] for r in valid)
    new_mean = mean(r["new_score"] for r in valid)
    lines.append("")
    lines.append(f"Overall mean RULER:")
    lines.append(f"  original (single-judge haiku):   {orig_mean:.3f}")
    lines.append(f"  new      (3-judge cross-family): {new_mean:.3f}")
    lines.append(f"  Δ (new − original):              {delta_mean:+.3f}")

    # Per-class.
    lines.append("")
    lines.append("Per scenario class:")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in valid:
        by_class[r["scenario_class"]].append(r)
    for cls in sorted(by_class.keys()):
        recs = by_class[cls]
        orig = mean(r["original_ruler_score"] for r in recs)
        new = mean(r["new_score"] for r in recs)
        delta = new - orig
        n = len(recs)
        all_axis_stds: list[float] = []
        for r in recs:
            for ax_std in (r.get("new_per_axis_disagreement_std") or {}).values():
                all_axis_stds.append(ax_std)
        mean_axis_std = mean(all_axis_stds) if all_axis_stds else 0.0
        mean_conf = mean(r.get("new_confidence", 1.0) for r in recs)
        lines.append(
            f"  {cls:<5s}  n={n:>3}  "
            f"orig={orig:.2f}  new={new:.2f}  Δ={delta:+.2f}  "
            f"axis_σ={mean_axis_std:.2f}  conf={mean_conf:.2f}"
        )

    # Per-epoch (skipped quietly if all records belong to one epoch,
    # e.g. baseline_fixed).
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in valid:
        by_epoch[int(r.get("epoch") or 0)].append(r)
    if len(by_epoch) > 1:
        lines.append("")
        lines.append("Per epoch:")
        for epoch in sorted(by_epoch.keys()):
            recs = by_epoch[epoch]
            orig = mean(r["original_ruler_score"] for r in recs)
            new = mean(r["new_score"] for r in recs)
            delta = new - orig
            mean_conf = mean(r.get("new_confidence", 1.0) for r in recs)
            lines.append(
                f"  epoch {epoch:>1}  n={len(recs):>3}  "
                f"orig={orig:.2f}  new={new:.2f}  Δ={delta:+.2f}  "
                f"conf={mean_conf:.2f}"
            )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline 3-judge cross-family re-score of e09 trajectories. "
                    "No agent re-execution, no graph updates.",
    )
    parser.add_argument(
        "--source-jsonl", type=Path, default=DEFAULT_SOURCE,
        help="Path to results_agensflow.jsonl from an e09 run.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="YAML config (defaults to e09's example_config.yaml).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the per-trajectory comparison JSONL. "
             "Default: <source_dir>/replay_rescore_results.jsonl.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on trajectories re-scored (for smoke tests).",
    )
    parser.add_argument(
        "--cross-judge", nargs="+", default=None,
        help="Override config.relative_judge.cross_judge_models. "
             "Default: cross-family triple "
             "[anthropic/claude-haiku-4.5, openai/gpt-5.4-mini, qwen/qwen3.6-flash].",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    # Default --out to sit next to the source file.
    if args.out is None:
        args.out = args.source_jsonl.parent / "replay_rescore_results.jsonl"

    cfg = load_config(args.config) if args.config.exists() else load_config()
    if args.cross_judge:
        cfg.relative_judge.cross_judge_models = list(args.cross_judge)
    if not cfg.relative_judge.cross_judge_models:
        cfg.relative_judge.cross_judge_models = list(DEFAULT_CROSS_JUDGE)
        for j, m in DEFAULT_CROSS_JUDGE_MODES.items():
            cfg.relative_judge.cross_judge_modes[j] = m
        for j, eb in DEFAULT_CROSS_JUDGE_EXTRA_BODY.items():
            cfg.relative_judge.cross_judge_extra_body[j] = eb

    print(f"Source:  {args.source_jsonl}")
    print(f"Output:  {args.out}")
    print(f"Judges:  {cfg.relative_judge.cross_judge_models}")
    print(f"Per-judge modes: {dict(cfg.relative_judge.cross_judge_modes)}")
    print(f"Per-judge extra_body overrides: {dict(cfg.relative_judge.cross_judge_extra_body)}")
    print(f"Default extra_body: {dict(cfg.relative_judge.extra_body_default)}")
    print(f"Axis weights: {dict(cfg.relative_judge.axis_weights)}")
    print()

    records = _load_records(args.source_jsonl)
    print(f"Loaded {len(records)} records.")
    if args.limit is not None:
        records = records[: args.limit]
        print(f"Limited to first {len(records)} records (smoke mode).")
    print()

    try:
        task_lookup = _build_user_task_lookup()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: couldn't load e09 task pool ({exc}); judge will see "
              f"placeholder task text.", file=sys.stderr)
        task_lookup = {}

    rubric = cfg.relative_judge.rubric or DEFAULT_RUBRIC
    client = OpenRouterClient()

    print(f"Re-scoring {len(records)} trajectories sequentially...")
    print(f"Estimated time: ~{len(records) * 4 / 60:.0f}-{len(records) * 6 / 60:.0f} min "
          f"at typical judge latency.")
    print()

    out = _replay(
        records,
        cfg_rj=cfg.ruler,
        client=client,
        task_lookup=task_lookup,
        rubric=rubric,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print()
    print(f"Wrote {len(out)} comparison records to {args.out}")
    print()
    print(_summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
