"""
Chunk-11 reward replay: re-score existing chunk-9 trajectories under the
upgraded RULER stack (cross-judge + per-axis), without re-executing any
agents and without updating the policy graph.

Cheap, focused validation of one question:

    Does the chunk-9 +0.09 RULER finding survive bias mitigation
    (cross-judge averaging) + per-axis decomposition?

What this script does and does NOT do:

  ✓ Loads the chunk-9 sweep's `results_agensflow.jsonl` (one record
    per trajectory, with final_answer + path_summary).
  ✓ Replays the rolling-buffer scoring discipline (group size 4 per
    scenario class, ordered by run_index) so each trajectory is judged
    against the SAME peers it was originally judged against.
  ✓ Runs the configured cross-judge stack (default: anthropic +
    openai + qwen at one tier each) and per-axis rubric.
  ✓ Writes a per-trajectory comparison JSONL (original vs new score,
    per-axis breakdown, per-axis disagreement, confidence).
  ✓ Prints a summary table: per-class mean score deltas + epoch
    deltas (original epoch-1→8 trajectory vs new).

  ✗ Does NOT re-execute any agents (no LLM cost on the agent side).
  ✗ Does NOT update the policy graph (input graph stays untouched).
  ✗ Does NOT exercise A1 (decompressed evidence) — JSONL records
    don't carry trace events. A1's effect is exercised separately
    during a small subset re-execution and during chunk 12.

Cost: ~$15-30 in judge calls (1,400 calls × ~$0.01-0.02). Time:
~60-90 min sequential at typical judge latency.

Usage:

    # Default config + default cross-family judge triple:
    python -m experiments.e07_skill_variants.replay_rescore

    # Smoke test with 20 trajectories:
    python -m experiments.e07_skill_variants.replay_rescore --limit 20

    # Custom source/output:
    python -m experiments.e07_skill_variants.replay_rescore \\
        --source-jsonl path/to/results_agensflow.jsonl \\
        --out path/to/replay_results.jsonl
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
DEFAULT_SOURCE = THIS_DIR / "postconfig_v1" / "results_agensflow.jsonl"
DEFAULT_OUT = THIS_DIR / "postconfig_v1" / "replay_rescore_results.jsonl"
DEFAULT_CONFIG = THIS_DIR / "example_config.yaml"

# Match the chunk-9 harness's rolling-buffer group size for apples-to-
# apples relative scoring (recent 3 same-class trajectories + the new one).
ROLLING_BUFFER_SIZE = 4

# Validated 3-family judge set from chunk-11 probe
# (`scripts/probe_qwen_judge.py`). Each judge has specific mode and
# extra_body requirements:
#
#   anthropic/claude-haiku-4.5  — TOOLS mode + require_parameters
#   openai/gpt-5.4-mini         — TOOLS mode + NO extra_body
#                                 (require_parameters paradoxically
#                                 breaks OpenAI's primary OpenRouter
#                                 route; pass empty dict to disable)
#   qwen/qwen3.6-flash          — JSON mode + require_parameters
#
# Three families covered: Anthropic, OpenAI, Qwen. n=3 enables tie-
# breaking + outlier detection in cross-judge averaging.
DEFAULT_CROSS_JUDGE = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "qwen/qwen3.6-flash",
]
DEFAULT_CROSS_JUDGE_MODES: dict[str, str] = {
    # anthropic + openai use the default TOOLS mode (no entry needed).
    "qwen/qwen3.6-flash": "json",
}
DEFAULT_CROSS_JUDGE_EXTRA_BODY: dict[str, dict] = {
    # Empty dict explicitly disables extra_body for openai (
    # require_parameters=True breaks gpt-5.4-* OpenRouter routing).
    "openai/gpt-5.4-mini": {},
    # Anthropic and qwen pick up the global `extra_body_default`
    # from RelativeJudgeConfig, which is `{provider: {require_parameters: True}}`.
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
    """Return {task_id → user_task} so we can pass the original task
    string to the judge. Sourced from the chunk-9 task pool."""
    from experiments.e03_production_traffic.tasks import ALL_TASKS
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
    # Per-class rolling buffer matching the original harness state.
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
            # No evidence — JSONL doesn't carry trace events. A1 is
            # exercised separately via the subset-reexecution script.
        )

        buffer = class_buffers[scenario_class]
        group = list(buffer) + [new_traj]

        # Single-trajectory groups: short-circuit per existing semantics
        # (returns neutral). We still record so downstream sees something
        # for every trajectory, but flag it explicitly.
        try:
            result = relative_judge_score_group(
                user_task=user_task,
                trajectories=group,
                client=client,
                # judge_model is ignored when cross_judge_models is set.
                judge_model=cfg_rj.judge_model,
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

        # Build the comparison record for this trajectory.
        original_score = rec.get("ruler_score")
        comparison = {
            "trajectory_id": traj_id,
            "task_id": task_id,
            "scenario_class": scenario_class,
            "epoch": rec.get("epoch"),
            "run_index": rec.get("run_index"),
            "original_ruler_score": original_score,
            "original_judge": "anthropic/claude-haiku-4.5",  # chunk-9 default
            "group_size": len(group),
            "path": path,
        }
        if new_score is not None:
            # Per-judge axis compliance: which judges actually returned
            # populated axis_scores vs empty. Cheap + critical telemetry
            # — without it, "axis_σ=0" looks like "judges agree" when
            # it actually means "only one judge returned axes."
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
    """Per-class + per-epoch summary of original vs new score."""
    lines: list[str] = []
    valid = [r for r in out if r.get("new_score") is not None
             and r.get("original_ruler_score") is not None]
    n_total = len(out)
    n_valid = len(valid)
    n_failed = n_total - n_valid

    lines.append("═" * 80)
    lines.append("Chunk-11 reward replay — summary")
    lines.append("═" * 80)
    lines.append(f"Trajectories re-scored: {n_valid}/{n_total} (failed: {n_failed})")
    if not valid:
        lines.append("No valid comparisons; cannot summarize.")
        return "\n".join(lines)

    # Per-judge axis compliance: critical telemetry — without it,
    # "per-axis disagreement std = 0" can look like agreement when
    # it's actually "only one judge populated axes."
    #
    # KEY: aggregate over the SAME identifier space as
    # `judges_returning_axes` was populated against — i.e., the
    # input config names (keys of per_judge_axis_scores), NOT the
    # API-echoed model names (`new_judges`). Mixing those caused
    # smoke-v2's "0% across the board" false alarm.
    multi_traj_records = [r for r in valid if r.get("group_size", 0) >= 2]
    if multi_traj_records:
        compliance: dict[str, int] = defaultdict(int)
        seen_attempts: dict[str, int] = defaultdict(int)
        all_judges: set[str] = set()
        for r in multi_traj_records:
            # Universe = judges seen as keys in per_judge_axis_scores
            # (this is the same naming that judges_returning_axes uses).
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
        non_compliant = [
            j for j in all_judges
            if compliance.get(j, 0) / max(seen_attempts.get(j, 0), 1) < 0.8
        ]
        if non_compliant:
            lines.append("")
            lines.append(
                f"  ⚠ Per-axis cross-judge averaging is NOT meaningful — "
                f"these judges did not populate axis_scores reliably:"
            )
            for j in non_compliant:
                lines.append(f"     - {j}")
            lines.append(
                f"     Replace with validated judges (see "
                f"experiments/e06_cross_eval/run.py:DEFAULT_JUDGES)."
            )

    # Headline delta.
    deltas = [r["new_score"] - r["original_ruler_score"] for r in valid]
    delta_mean = mean(deltas)
    lines.append("")
    lines.append(f"Mean Δ (new − original) across all trajectories: {delta_mean:+.3f}")

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
        # Mean per-axis disagreement std as a quick "judges agreed?" signal.
        all_axis_stds: list[float] = []
        for r in recs:
            for ax_std in (r.get("new_per_axis_disagreement_std") or {}).values():
                all_axis_stds.append(ax_std)
        mean_axis_std = mean(all_axis_stds) if all_axis_stds else 0.0
        # Mean confidence.
        mean_conf = mean(r.get("new_confidence", 1.0) for r in recs)
        lines.append(
            f"  {cls:<5s}  n={n:>3}  "
            f"orig={orig:.2f}  new={new:.2f}  Δ={delta:+.2f}  "
            f"axis_σ={mean_axis_std:.2f}  conf={mean_conf:.2f}"
        )

    # Per-epoch.
    lines.append("")
    lines.append("Per epoch (Δ = new − original mean RULER):")
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in valid:
        by_epoch[int(r.get("epoch") or 0)].append(r)
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

    # Headline epoch-1 → epoch-N delta comparison (the chunk-9 +0.09 claim).
    if 1 in by_epoch and 8 in by_epoch:
        ep1_orig = mean(r["original_ruler_score"] for r in by_epoch[1])
        ep8_orig = mean(r["original_ruler_score"] for r in by_epoch[8])
        ep1_new = mean(r["new_score"] for r in by_epoch[1])
        ep8_new = mean(r["new_score"] for r in by_epoch[8])
        lines.append("")
        lines.append("Headline: chunk-9 quality-improvement claim under upgraded reward")
        lines.append(f"  Original (single-judge haiku): epoch1={ep1_orig:.2f} → epoch8={ep8_orig:.2f}  (Δ {ep8_orig - ep1_orig:+.2f})")
        lines.append(f"  New (cross-judge + per-axis):  epoch1={ep1_new:.2f} → epoch8={ep8_new:.2f}  (Δ {ep8_new - ep1_new:+.2f})")
        lines.append("")
        if (ep8_new - ep1_new) > 0.02:
            verdict = "SURVIVES — quality-improvement claim holds under bias mitigation."
        elif (ep8_new - ep1_new) > -0.02:
            verdict = "MARGINAL — original claim was tight under single-judge; under cross-judge it's roughly flat."
        else:
            verdict = "OVERTURNED — original quality-improvement claim was bias-driven; substrate cost-compresses but doesn't actually improve quality."
        lines.append(f"  Verdict: {verdict}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score chunk-9 trajectories under chunk-11's "
                    "upgraded RULER (cross-judge + per-axis).",
    )
    parser.add_argument(
        "--source-jsonl", type=Path, default=DEFAULT_SOURCE,
        help="Path to results_agensflow.jsonl from a chunk-9 sweep.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="YAML config (defaults to example_config.yaml).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Where to write the per-trajectory comparison JSONL.",
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

    # Load config; inject cross-judge if not already configured.
    cfg = load_config(args.config) if args.config.exists() else load_config()
    if args.cross_judge:
        cfg.relative_judge.cross_judge_models = list(args.cross_judge)
    if not cfg.relative_judge.cross_judge_models:
        cfg.relative_judge.cross_judge_models = list(DEFAULT_CROSS_JUDGE)
        # Apply the validated mode + extra_body per judge (chunk-11
        # probe). Without these, qwen/grok routes 404 and OpenAI's
        # primary route breaks.
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

    # Build user_task lookup so we hand the judge the same task string
    # the original sweep saw.
    try:
        task_lookup = _build_user_task_lookup()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: couldn't load task pool ({exc}); judge will see "
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

    # Persist the comparison JSONL.
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
