"""
Chunk 6 entry point: production-traffic benchmark.

Runs the 60-task distributed-systems benchmark through:
  1. AgensFlow with the chunk-6 activation plan (full variant pool) and
     the shared persistent policy graph + RULER-anchored hybrid reward.
  2. The multi-agent retry-stack baseline on the same tasks.

Compares the two on tokens-per-task at equivalent RULER quality.

Usage:
  python -m experiments.e03_production_traffic.run                       # full benchmark
  python -m experiments.e03_production_traffic.run --tasks C1.1,C2.1     # subset
  python -m experiments.e03_production_traffic.run --skip-baseline       # AgensFlow only
  python -m experiments.e03_production_traffic.run --skip-agensflow      # baseline only
  python -m experiments.e03_production_traffic.run --resume              # continue from saved graph
  python -m experiments.e03_production_traffic.run --reset               # delete graph + results, fresh start
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from agensflow import (
    DEFAULT_RUBRIC,
    PolicyGraph,
    RewardConfig,
    load_policy_graph,
    save_policy_graph,
)
from agensflow.runtime.client import OpenRouterClient

from experiments.e03_production_traffic.baseline import (
    BaselineRecord,
    baseline_record_to_dict,
    run_baseline_for_task,
)
from experiments.e03_production_traffic.harness import (
    DEFAULT_JUDGE_MODEL,
    HarnessState,
    TrajectoryRecord,
    record_to_dict,
    run_full_benchmark,
)
from experiments.e03_production_traffic.tasks import ALL_TASKS, ProductionTask

THIS_DIR = Path(__file__).parent
GRAPH_PATH = THIS_DIR / "policy_graph.pkl"
TRACE_PATH_AGENSFLOW = THIS_DIR / "results_agensflow.jsonl"
TRACE_PATH_BASELINE = THIS_DIR / "results_baseline.jsonl"
RESULTS_PATH = THIS_DIR / "RESULTS.md"

RESULTS_BEGIN = "<!-- RESULTS:BEGIN (auto-generated; do not edit) -->"
RESULTS_END = "<!-- RESULTS:END -->"

DEFAULT_SHUFFLE_SEED = 20260505


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _aggregate_agensflow(records: list[TrajectoryRecord]) -> dict[str, Any]:
    by_class: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for r in records:
        by_class[r.scenario_class].append(r)

    per_class: dict[str, Any] = {}
    for cls, recs in sorted(by_class.items()):
        n = len(recs)
        if n == 0:
            continue
        successful = [r for r in recs if r.error is None]
        per_class[cls] = {
            "n": n,
            "n_errors": n - len(successful),
            "tokens_avg": mean(r.total_tokens for r in successful) if successful else 0.0,
            "ruler_avg": mean(r.ruler_score for r in successful if r.ruler_score is not None) if successful else 0.0,
            "reward_avg": mean(r.hybrid_reward for r in successful if r.hybrid_reward is not None) if successful else 0.0,
            "retries_avg": mean(r.validation_retries for r in successful) if successful else 0.0,
            "variant_distribution": dict(
                Counter(_solver_variant_in_path(r.path) for r in successful)
            ),
            "verifier_invocation_rate": (
                sum(1 for r in successful if _has_verifier(r.path))
                / max(1, len(successful))
            ),
            "web_invocation_rate": (
                sum(1 for r in successful if _has_web(r.path))
                / max(1, len(successful))
            ),
            "expected_variant_match_rate": (
                sum(
                    1 for r in successful
                    if _solver_variant_in_path(r.path) == r.expected_optimal_variant
                ) / max(1, len(successful))
            ),
        }

    # Trajectory across runs (run_index ordering).
    sorted_recs = sorted(records, key=lambda r: r.run_index)
    return {
        "n_records": len(records),
        "per_class": per_class,
        "tokens_trajectory": [r.total_tokens for r in sorted_recs],
        "ruler_trajectory": [r.ruler_score for r in sorted_recs],
        "reward_trajectory": [r.hybrid_reward for r in sorted_recs],
    }


def _aggregate_baseline(records: list[BaselineRecord]) -> dict[str, Any]:
    by_class: dict[str, list[BaselineRecord]] = defaultdict(list)
    for r in records:
        by_class[r.scenario_class].append(r)

    per_class: dict[str, Any] = {}
    for cls, recs in sorted(by_class.items()):
        n = len(recs)
        if n == 0:
            continue
        successful = [r for r in recs if r.error is None]
        per_class[cls] = {
            "n": n,
            "n_errors": n - len(successful),
            "tokens_avg": mean(r.total_tokens for r in successful) if successful else 0.0,
            "ruler_avg": mean(r.ruler_score for r in successful if r.ruler_score is not None) if successful else 0.0,
            "reward_avg": mean(r.hybrid_reward for r in successful if r.hybrid_reward is not None) if successful else 0.0,
            "retries_avg": mean(r.n_retries_total for r in successful) if successful else 0.0,
        }

    return {
        "n_records": len(records),
        "per_class": per_class,
    }


def _solver_variant_in_path(path: list[str]) -> str:
    for action in path:
        if action.startswith("solver_"):
            return action
    return "(none)"


def _has_verifier(path: list[str]) -> bool:
    return any(a.startswith("verifier_") or a == "verifier" for a in path)


def _has_web(path: list[str]) -> bool:
    return any(a.startswith("web_search_") for a in path)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _render_results(
    af_records: list[TrajectoryRecord] | None,
    bl_records: list[BaselineRecord] | None,
) -> str:
    lines: list[str] = []
    lines.append(RESULTS_BEGIN)
    lines.append("")
    lines.append("## Results")
    lines.append("")

    if af_records is not None:
        af_agg = _aggregate_agensflow(af_records)
        lines.append("### AgensFlow with chunk-6 hybrid reward + variant pool")
        lines.append("")
        lines.append(f"Total runs: {af_agg['n_records']}")
        lines.append("")
        lines.append(
            "| class | n | tokens/run | RULER avg | reward avg | retries avg | "
            "verifier-rate | web-rate | optimal-variant match |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cls in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
            stats = af_agg["per_class"].get(cls)
            if stats is None:
                continue
            lines.append(
                f"| {cls} | {stats['n']} | "
                f"{int(stats['tokens_avg'])} | "
                f"{stats['ruler_avg']:.2f} | "
                f"{stats['reward_avg']:+.2f} | "
                f"{stats['retries_avg']:.2f} | "
                f"{_fmt_pct(stats['verifier_invocation_rate'])} | "
                f"{_fmt_pct(stats['web_invocation_rate'])} | "
                f"{_fmt_pct(stats['expected_variant_match_rate'])} |"
            )
        lines.append("")

        # Variant distribution per class (which solver did the policy converge to?)
        lines.append("#### Solver variant distribution by class")
        lines.append("")
        lines.append("| class | top variant chosen | distribution |")
        lines.append("|---|---|---|")
        for cls in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
            stats = af_agg["per_class"].get(cls)
            if stats is None:
                continue
            dist = stats["variant_distribution"]
            top = max(dist.items(), key=lambda x: x[1]) if dist else ("(none)", 0)
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
            lines.append(f"| {cls} | `{top[0]}` ({top[1]}) | {dist_str} |")
        lines.append("")

    if bl_records is not None:
        bl_agg = _aggregate_baseline(bl_records)
        lines.append("### Multi-agent retry-stack baseline")
        lines.append("")
        lines.append(f"Total runs: {bl_agg['n_records']}")
        lines.append("")
        lines.append(
            "| class | n | tokens/run | RULER avg | reward avg | retries avg |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cls in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
            stats = bl_agg["per_class"].get(cls)
            if stats is None:
                continue
            lines.append(
                f"| {cls} | {stats['n']} | "
                f"{int(stats['tokens_avg'])} | "
                f"{stats['ruler_avg']:.2f} | "
                f"{stats['reward_avg']:+.2f} | "
                f"{stats['retries_avg']:.2f} |"
            )
        lines.append("")

    if af_records is not None and bl_records is not None:
        lines.append("### Headline comparison: AgensFlow vs. retry-stack baseline")
        lines.append("")
        lines.append("| class | AF tokens/run | BL tokens/run | AF RULER | BL RULER | token gap | RULER gap |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        af_agg = _aggregate_agensflow(af_records)
        bl_agg = _aggregate_baseline(bl_records)
        for cls in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
            a = af_agg["per_class"].get(cls)
            b = bl_agg["per_class"].get(cls)
            if a is None or b is None:
                continue
            tok_gap_pct = (
                (b["tokens_avg"] - a["tokens_avg"]) / b["tokens_avg"] * 100.0
                if b["tokens_avg"] > 0 else 0.0
            )
            ruler_gap = a["ruler_avg"] - b["ruler_avg"]
            lines.append(
                f"| {cls} | {int(a['tokens_avg'])} | {int(b['tokens_avg'])} | "
                f"{a['ruler_avg']:.2f} | {b['ruler_avg']:.2f} | "
                f"{tok_gap_pct:+.0f}% | {ruler_gap:+.2f} |"
            )
        lines.append("")
        lines.append(
            "*Token gap*: positive = AgensFlow used fewer tokens (better). "
            "*RULER gap*: positive = AgensFlow scored higher quality."
        )
        lines.append("")

    lines.append(RESULTS_END)
    return "\n".join(lines)


def _splice_into_results(rendered: str) -> None:
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
    return """# Experiment 03 — Production-traffic benchmark

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-05
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 6 (variant pool + hybrid RULER reward + UCB v2)

### Setup

- 60 tasks across 8 scenario classes over a 12-document distributed-systems
  corpus.
- AgensFlow path: chunk-6 activation plan (5 solver variants spanning OpenAI /
  Anthropic / Qwen, 2 verifier variants, corpus memory + 2 web search
  providers), shared persistent policy graph, hybrid reward (RULER + cost +
  retry), UCB v2 (annealed exploration, threshold 5).
- Baseline: planner → memory → solver_qwen_max → verifier_haiku → evaluator,
  retry-on-failure between stages, no policy learning.
- Both paths are RULER-scored against rolling per-class trajectory buffers,
  so quality comparisons are apples-to-apples.

### Primary prediction (cost reduction)

Across the 60 tasks, AgensFlow's average tokens-per-task will be **at least
20% lower** than the retry-stack baseline at **equivalent or higher RULER
quality**. The savings come from the policy learning to:

- Pick `solver_fast` / `solver_qwen_flash` for C1, C6, C8 (simple lookups,
  definitions, numerical extraction).
- Skip the verifier on signature classes where it adds no quality (C1, C6).
- Skip web search on signatures the corpus answers cleanly (C1-C4, C6, C8).

### Secondary prediction (per-class variant convergence)

For each scenario class, the policy will converge to within 50% of the
expected optimal variant by end of benchmark. We track this via the
"optimal-variant match rate" column in the results table. The rate isn't
expected to be 100% with N=60 runs — that requires more sustained traffic
than chunk 6 simulates — but ≥50% indicates the learning is moving in the
right direction.

### Tertiary prediction (verifier and web invocation routing)

- C1, C6 (skip-verifier classes): verifier invocation rate ≤ 30%.
- C3, C5 (essential-verifier classes): verifier invocation rate ≥ 70%.
- C5 (no-corpus-answer): web invocation rate ≥ 50%.

### What would falsify the framework's claim on this benchmark

- *Primary fails*: tokens-per-task gap ≤ 5% or AgensFlow is more expensive
  than the baseline. The economic claim doesn't hold on this benchmark and
  needs reformulation.
- *Secondary fails*: optimal-variant match rate < 30% across all classes.
  The reward signal isn't differentiating variants enough; either the RULER
  rubric needs sharpening or the cost weights need re-tuning.
- *Tertiary fails*: verifier and web invocation rates don't differentiate
  between classes. The policy isn't learning per-class structural decisions
  even when the variant choice is right.

### Acknowledged limitations before running

- N=60 is small for population-level claims. Treat results as preliminary;
  scale up for the full paper.
- Single-trial per task. LLM variance not quantified.
- Same-family RULER bias: judge model is claude-haiku-4.5, in the variant
  pool. A stronger out-of-mix judge is future work.
- Synthetic corpus, not real papers. Reproducibility prioritized over
  authenticity.
- The chunk 6 design tests *learning trajectory*, not *converged* policy.
  Convergence to per-signature optima would require thousands of runs, which
  is the production-traffic-volume regime — out of scope here.
"""


def _dump_records(records: Iterable, path: Path, to_dict_fn) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_dict_fn(r), default=str) + "\n")


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    """Load records previously dumped to JSONL (used by --resume + merge)."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _merge_by_task_id(
    existing: list[dict[str, Any]],
    new_records,
    to_dict_fn,
) -> list[dict[str, Any]]:
    """
    Merge new records into existing by task_id. New records win on conflict —
    so re-running a failed task with chunk-6.5's bumped config replaces the
    failed record with the now-successful one.
    """
    new_dicts = [to_dict_fn(r) for r in new_records]
    by_id: dict[str, dict[str, Any]] = {r["task_id"]: r for r in existing}
    for r in new_dicts:
        by_id[r["task_id"]] = r
    return list(by_id.values())


def _dicts_to_trajectory_records(dicts: list[dict[str, Any]]) -> list[TrajectoryRecord]:
    """Deserialize dicts back into TrajectoryRecord objects for aggregation."""
    out: list[TrajectoryRecord] = []
    for d in dicts:
        # Defensive: filter to TrajectoryRecord's known fields so a stale
        # JSONL with extra keys doesn't break dataclass construction.
        valid = {f for f in TrajectoryRecord.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in valid}
        out.append(TrajectoryRecord(**kwargs))
    return out


def _dicts_to_baseline_records(dicts: list[dict[str, Any]]) -> list[BaselineRecord]:
    out: list[BaselineRecord] = []
    for d in dicts:
        valid = {f for f in BaselineRecord.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in valid}
        out.append(BaselineRecord(**kwargs))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chunk 6 production-traffic benchmark.")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task ids to run (default: all 60).")
    parser.add_argument("--skip-agensflow", action="store_true",
                        help="Skip the AgensFlow path; baseline only.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the baseline; AgensFlow only.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from saved policy graph.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete saved graph + results before starting.")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED,
                        help="Seed for shuffling task order (deterministic).")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Model id for the RULER judge.")
    parser.add_argument("--max-steps", type=int, default=14,
                        help="Max routing decisions per AgensFlow run.")
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write to RESULTS.md; print summary only.")
    args = parser.parse_args()

    if args.reset:
        for p in [GRAPH_PATH, TRACE_PATH_AGENSFLOW, TRACE_PATH_BASELINE]:
            if p.exists():
                p.unlink()
                print(f"Reset: removed {p}")

    # Pick tasks.
    tasks = list(ALL_TASKS)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"No tasks match: {sorted(wanted)}", file=sys.stderr)
            return 2

    # Shuffle deterministically.
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(tasks)

    print(f"Running chunk-6 benchmark: {len(tasks)} task(s).")
    print(f"  agensflow: {not args.skip_agensflow}")
    print(f"  baseline:  {not args.skip_baseline}")
    print(f"  judge:     {args.judge_model}")
    print()

    # Load .env from the repo root so OPENROUTER_API_KEY etc. resolve.
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenRouterClient()
    reward_config = RewardConfig()  # defaults

    af_records: list[TrajectoryRecord] | None = None
    bl_records: list[BaselineRecord] | None = None

    if not args.skip_agensflow:
        if args.resume and GRAPH_PATH.exists():
            graph = load_policy_graph(GRAPH_PATH)
            print(f"Resumed graph from {GRAPH_PATH} ({len(graph)} nodes)")
        else:
            graph = PolicyGraph()
            print("Starting with empty policy graph")

        state = HarnessState(
            policy_graph=graph,
            client=client,
            reward_config=reward_config,
            judge_model=args.judge_model,
            rubric=DEFAULT_RUBRIC,
        )
        print()
        print("=== AgensFlow path ===")
        af_records = run_full_benchmark(
            tasks, state=state, max_steps=args.max_steps,
        )
        save_policy_graph(graph, GRAPH_PATH)
        # When --resume is set, merge new records into the existing JSONL by
        # task_id (new wins on conflict). This is the chunk-6.5 workflow:
        # bump configs, re-run failed tasks, merged dataset gets re-aggregated.
        if args.resume and TRACE_PATH_AGENSFLOW.exists():
            existing = _load_existing_records(TRACE_PATH_AGENSFLOW)
            merged = _merge_by_task_id(existing, af_records, record_to_dict)
            with TRACE_PATH_AGENSFLOW.open("w") as f:
                for r in merged:
                    f.write(json.dumps(r, default=str) + "\n")
            # Use the merged dataset for the RESULTS.md re-render so the
            # aggregation reflects all 60 tasks, not just the re-run subset.
            af_records = _dicts_to_trajectory_records(merged)
            print(
                f"\nGraph saved to {GRAPH_PATH} ({len(graph)} nodes)\n"
                f"AgensFlow records merged: {len(existing)} prior + "
                f"{len([r for r in merged if r['task_id'] in {t.id for t in tasks}])} "
                f"re-runs -> {len(merged)} total at {TRACE_PATH_AGENSFLOW}"
            )
        else:
            _dump_records(af_records, TRACE_PATH_AGENSFLOW, record_to_dict)
            print(f"\nGraph saved to {GRAPH_PATH} ({len(graph)} nodes)")
            print(f"AgensFlow records dumped to {TRACE_PATH_AGENSFLOW}")

    if not args.skip_baseline:
        print()
        print("=== Multi-agent retry-stack baseline ===")
        bl_buffer: deque = deque(maxlen=4)
        bl_records = []
        for i, task in enumerate(tasks, start=1):
            print(f"=== run {i}/{len(tasks)} ===")
            rec = run_baseline_for_task(
                task,
                client=client,
                run_index=i,
                rolling_buffer=bl_buffer,
                reward_config=reward_config,
                judge_model=args.judge_model,
                rubric=DEFAULT_RUBRIC,
            )
            bl_records.append(rec)
            print()
        _dump_records(bl_records, TRACE_PATH_BASELINE, baseline_record_to_dict)
        print(f"\nBaseline records dumped to {TRACE_PATH_BASELINE}")

    # On --resume + --skip-baseline, load the prior baseline JSONL so the
    # headline comparison still spans both paths in the re-rendered RESULTS.md.
    if (
        args.resume and args.skip_baseline
        and bl_records is None and TRACE_PATH_BASELINE.exists()
    ):
        bl_records = _dicts_to_baseline_records(
            _load_existing_records(TRACE_PATH_BASELINE)
        )

    rendered = _render_results(af_records, bl_records)
    if args.no_write:
        print(rendered)
    else:
        _splice_into_results(rendered)
        print(f"\nResults written to {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
