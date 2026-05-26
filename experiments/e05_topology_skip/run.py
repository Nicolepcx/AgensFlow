"""
Chunk 8 entry point — inline `skip:X` topology learning.

Same harness as chunk 7 (warm-start from chunk-6.5 graph, 59-task pool,
8 epochs, deterministic shuffle), but with the router's `enable_skip`
mechanism turned on. At every routing step the action space expands to
include `skip:X` for every legal X, so the policy can learn to *exclude*
skills from the topology — not just choose which model variant or in
what order.

The empirical question:

    *When the framework can learn coalition membership as a coordination
     decision (not just order or model binding), does it discover
     non-trivial skip patterns per scenario class — and does the
     resulting cost-quality curve improve over chunk 7's baseline?*

Pre-registered predictions:

  1. **Selective skip-learning.** In ≥2 of 8 classes, `skip:X` becomes
     the dominant action at some signatures within that class.
  2. **Cost reduction beyond chunk 7.** Mean tokens per task in epoch 8
     drops ≥10% compared to chunk-7 epoch 8. Mechanism: skipped skills
     cost zero tokens.
  3. **No quality regression.** RULER mean stays within 0.05 of
     chunk-7 epoch 8 baseline (~0.84). UCB should route around any
     "skip everything → produce nothing" pathology after a few
     exploration runs.

Per-epoch checkpointing (graph snapshot + JSONL append) makes the
experiment crash-safe and lets the visualizer plot trajectories.

Usage:
    # Standard 8-epoch sustained run (warm-start from chunk 6.5):
    python -m experiments.e05_topology_skip.run

    # Smoke test on a 4-task subset (cold-start to avoid warm-start
    # bias for an instrumentation-only check):
    python -m experiments.e05_topology_skip.run --tasks C1.1,C2.1,C5.1,C8.1 --epochs 2 --cold-start

    # Resume from this experiment's own saved graph (after a crash):
    python -m experiments.e05_topology_skip.run --resume

    # Disable skip mechanism (degenerates to chunk-7 main run, useful
    # for sanity-checking that the rest of the pipeline is identical):
    python -m experiments.e05_topology_skip.run --no-skip
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from agensflow import (
    DEFAULT_RUBRIC,
    PolicyGraph,
    RewardConfig,
    load_policy_graph,
    save_policy_graph,
)
from agensflow.runtime.client import OpenRouterClient

from experiments.e03_production_traffic.harness import (
    DEFAULT_JUDGE_MODEL,
    HarnessState,
    TrajectoryRecord,
    record_to_dict,
    run_full_benchmark,
)
from experiments.e03_production_traffic.tasks import ALL_TASKS

# ----- Paths -----
THIS_DIR = Path(__file__).parent
# Output directory — overridable via --output-suffix so the λ=0 ablation
# writes to its own subdirectory and doesn't clobber the main λ=0.5 run.
RESULTS_DIR = THIS_DIR  # mutated in main() if --output-suffix given
RESULTS_AGENSFLOW = RESULTS_DIR / "results_agensflow.jsonl"
GRAPH_PATH = RESULTS_DIR / "policy_graph.pkl"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
RESULTS_PATH = RESULTS_DIR / "RESULTS.md"

# Warm-start source — chunk 6.5's graph.
CHUNK6_GRAPH_PATH = (
    THIS_DIR.parent / "e03_production_traffic" / "policy_graph.pkl"
)

# Tasks excluded from the chunk-7 pool. C7.1 still trips the LangGraph
# recursion limit at 200 (documented in chunk 6.5 RESULTS.md as an edge
# case unrelated to the framework's claim). Including it would just add
# 8 errors over 8 epochs and noise up the curves.
EXCLUDE_TASK_IDS = {"C7.1"}

DEFAULT_EPOCHS = 8
DEFAULT_SHUFFLE_SEED = 20260507  # different from chunk-6 seed


# --------------------------------------------------------------------------- #
# Per-epoch I/O helpers
# --------------------------------------------------------------------------- #


def _append_record(record: TrajectoryRecord, path: Path) -> None:
    """Append one record to the JSONL — used as the harness on_record callback."""
    with path.open("a") as f:
        f.write(json.dumps(record_to_dict(record), default=str) + "\n")


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _highest_completed_epoch(records: list[dict[str, Any]]) -> int:
    """The largest fully-finished epoch number — for --resume."""
    if not records:
        return 0
    by_epoch: dict[int, set[str]] = defaultdict(set)
    for r in records:
        by_epoch[int(r.get("epoch", 0))].add(r["task_id"])
    # An epoch is "complete" if it has the same task count as the largest one.
    max_count = max(len(ids) for ids in by_epoch.values()) if by_epoch else 0
    completed = [e for e, ids in by_epoch.items() if len(ids) >= max_count]
    return max(completed) if completed else 0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _aggregate_per_epoch(records: list[TrajectoryRecord]) -> list[dict[str, Any]]:
    by_epoch: dict[int, list[TrajectoryRecord]] = defaultdict(list)
    for r in records:
        by_epoch[r.epoch].append(r)
    rows: list[dict[str, Any]] = []
    for epoch in sorted(by_epoch.keys()):
        recs = by_epoch[epoch]
        ok = [r for r in recs if r.error is None]
        rows.append({
            "epoch": epoch,
            "n": len(recs),
            "n_errors": len(recs) - len(ok),
            "tokens_avg": int(mean(r.total_tokens for r in ok)) if ok else 0,
            "ruler_avg": (
                mean(r.ruler_score for r in ok if r.ruler_score is not None)
                if ok else 0.0
            ),
            "reward_avg": (
                mean(r.hybrid_reward for r in ok if r.hybrid_reward is not None)
                if ok else 0.0
            ),
            "retries_avg": mean(r.validation_retries for r in ok) if ok else 0.0,
        })
    return rows


def _render_results(records: list[TrajectoryRecord], graph: PolicyGraph) -> str:
    rows = _aggregate_per_epoch(records)
    lines: list[str] = []
    lines.append("<!-- RESULTS:BEGIN (auto-generated; do not edit) -->")
    lines.append("")
    lines.append("## Results — topology skip-X (chunk 8)")
    lines.append("")
    lines.append(f"Total runs: {len(records)}")
    lines.append(f"Final graph: {len(graph)} nodes, "
                 f"{sum(n.visits for n in graph.nodes.values())} total visits")
    lines.append("")
    lines.append("### Per-epoch trajectory")
    lines.append("")
    lines.append("| epoch | n | errors | tokens/run | RULER avg | reward avg | retries avg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['epoch']} | {r['n']} | {r['n_errors']} | "
            f"{r['tokens_avg']} | {r['ruler_avg']:.2f} | "
            f"{r['reward_avg']:+.2f} | {r['retries_avg']:.2f} |"
        )
    lines.append("")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        token_delta_pct = (
            (first["tokens_avg"] - last["tokens_avg"]) / max(1, first["tokens_avg"]) * 100
        )
        ruler_delta = last["ruler_avg"] - first["ruler_avg"]
        lines.append(
            f"**Δ epoch {first['epoch']} → {last['epoch']}:** "
            f"tokens {token_delta_pct:+.0f}% (positive = cheaper), "
            f"RULER {ruler_delta:+.2f} (positive = better quality)."
        )
        lines.append("")

    # Per-edge failure-rate top-10.
    lines.append("### Top unreliable (signature, action) edges in final graph")
    lines.append("")
    lines.append("| signature (regime) | action | visits | failures | failure rate |")
    lines.append("|---|---|---:|---:|---:|")
    edges: list[tuple[str, str, int, int, float]] = []
    for sig, node in graph.nodes.items():
        for action, n_fail in node.action_failure_count.items():
            n_visits = node.action_visits.get(action, 0)
            denom = n_visits + n_fail
            if denom == 0:
                continue
            rate = n_fail / denom
            regime = sig[0] if sig else "?"
            edges.append((str(regime), action, n_visits, n_fail, rate))
    edges.sort(key=lambda e: (-e[4], -e[3]))  # by rate desc, then failures desc
    for regime, action, n_visits, n_fail, rate in edges[:10]:
        lines.append(
            f"| {regime} | `{action}` | {n_visits} | {n_fail} | {rate:.2%} |"
        )
    if not edges:
        lines.append("| (no failures recorded yet) |  |  |  |  |")
    lines.append("")
    lines.append("<!-- RESULTS:END -->")
    return "\n".join(lines)


def _splice_into_results(rendered: str) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(_predictions_stub() + "\n\n" + rendered + "\n")
        return
    text = RESULTS_PATH.read_text()
    BEGIN = "<!-- RESULTS:BEGIN (auto-generated; do not edit) -->"
    END = "<!-- RESULTS:END -->"
    if BEGIN in text and END in text:
        before, _, rest = text.partition(BEGIN)
        _, _, after = rest.partition(END)
        new_text = before.rstrip() + "\n\n" + rendered + "\n" + after.lstrip()
    else:
        new_text = text.rstrip() + "\n\n" + rendered + "\n"
    RESULTS_PATH.write_text(new_text)


def _predictions_stub() -> str:
    return """# Experiment 05 — Topology skip-X learning (chunk 8)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-06
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 8 — inline `skip:X` action mechanism
on top of chunk 7's validated substrate (UCB-on-folded-signatures policy
graph). Builds on the chunk 7 + ablation finding that the substrate itself
delivers the cost-quality improvement.

### What's new vs chunk 7

The router's action space now expands at every step to include `skip:X`
for every legal X. This makes coalition membership a learnable
coordination decision: the policy can choose to *exclude* a skill from
the topology rather than only re-ordering / re-binding it. Same substrate,
richer action space.

### Setup

- Same 59-task pool, same shuffle seed (deterministic A/B against chunk 7).
- 8 epochs, warm-start from chunk-6.5 graph (same as chunk 7).
- Same hybrid reward, same chunk-6 activation plan (full variant pool).
- λ (reliability_weight) = 0.5 (chunk-7 default; the ablation showed this
  doesn't materially affect outcome but we keep it for consistency).
- `enable_skip = True` (the experimental knob).

### Primary prediction (selective skip-learning)

- **In ≥2 of 8 scenario classes, `skip:X` becomes the dominant action at
  some signatures within that class** by epoch 8. Most likely candidates
  per chunk-6 expectations: skip-verifier in C1, C6, C8; skip-web-search
  in C1–C4, C6, C8 where the corpus already has the answer.

### Secondary prediction (cost reduction beyond chunk 7)

- **Mean tokens per task in epoch 8 drops ≥10% compared to chunk-7
  epoch 8** (~9,010 → ~8,100 or below). Mechanism: skipped skills cost
  zero tokens.

### Tertiary prediction (no quality regression)

- **RULER mean stays within 0.05 of chunk-7 epoch 8 baseline** (~0.84).
  UCB should route around the "skip everything → produce nothing →
  terminal low reward" pathology after a few exploration runs; the
  reward signal will catch quality degradation from over-skipping.

### What would falsify

- *(1) fails*: skip actions never become dominant. Either signature
  folding is too coarse to distinguish "X helps here" from "X doesn't",
  or exploration constant is too low to ever try skips enough times to
  learn their value.
- *(2) fails*: mean tokens unchanged from chunk-7 baseline. Either skips
  aren't being explored, or the policy correctly identifies that all
  currently-included skills are necessary at all signatures.
- *(3) fails*: tokens drop but RULER drops with them (cheap-but-bad).
  Reward signal isn't catching quality degradation from over-skipping —
  the chunk-7 hybrid reward's RULER weight may need re-tuning.

### Acknowledged limitations before running

- Single-condition run (skip enabled). A no-skip ablation isn't needed
  here because chunk 7's main run already serves as that baseline (same
  seed, same warm-start, same everything except `enable_skip`).
- Skip events count toward `max_steps` (per chunk-8 design decision),
  so a runaway skip-skip loop terminates cleanly via budget exhaustion.
- The activation plan still defines the *available* skills — skip can
  only refuse skills that were already legal. Plan-level coalition
  selection (the planner-level UCB sketched for chunk 9) is a separate
  layer, not in scope here.
- Same constraints as chunk 7: one corpus, one variant pool, one judge,
  no external baseline.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run chunk 8 topology skip-X experiment."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of sweeps through the task pool (default {DEFAULT_EPOCHS}).")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task ids to run (default: full 59-task pool).")
    parser.add_argument("--cold-start", action="store_true",
                        help="Don't warm from chunk 6.5; start with empty graph.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from this experiment's own saved graph + JSONL.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete this experiment's saved graph + results before starting.")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--confidence-threshold", type=int, default=5)
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write to RESULTS.md; print summary only.")
    parser.add_argument("--reliability-weight", type=float, default=0.5,
                        help="UCB reliability term coefficient λ (default 0.5; 0 = ablation).")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Subdirectory under e05_topology_skip/ to write outputs into "
                             "(useful for ablations).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Disable the skip:X mechanism (degenerates to chunk-7 main "
                             "behavior; useful for sanity-checking the harness).")
    args = parser.parse_args()
    enable_skip = not args.no_skip

    # Re-route output paths if --output-suffix is set so the ablation
    # writes into its own subdir without colliding with the main run.
    global RESULTS_DIR, RESULTS_AGENSFLOW, GRAPH_PATH, SNAPSHOTS_DIR, RESULTS_PATH
    if args.output_suffix:
        RESULTS_DIR = THIS_DIR / args.output_suffix
        RESULTS_AGENSFLOW = RESULTS_DIR / "results_agensflow.jsonl"
        GRAPH_PATH = RESULTS_DIR / "policy_graph.pkl"
        SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
        RESULTS_PATH = RESULTS_DIR / "RESULTS.md"

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for p in [RESULTS_AGENSFLOW, GRAPH_PATH]:
            if p.exists():
                p.unlink()
                print(f"Reset: removed {p}")
        for p in SNAPSHOTS_DIR.glob("*.pkl"):
            p.unlink()
            print(f"Reset: removed {p}")

    # Pick tasks (default = full pool minus excluded).
    tasks = [t for t in ALL_TASKS if t.id not in EXCLUDE_TASK_IDS]
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"No tasks match: {sorted(wanted)}", file=sys.stderr)
            return 2

    # Load / start graph.
    if args.resume and GRAPH_PATH.exists():
        graph = load_policy_graph(GRAPH_PATH)
        print(f"Resumed from {GRAPH_PATH} ({len(graph)} nodes)")
        existing = _load_existing_records(RESULTS_AGENSFLOW)
        start_epoch = _highest_completed_epoch(existing) + 1
        prior_runs = len(existing)
        print(f"  prior epochs complete: {start_epoch - 1}, prior runs: {prior_runs}")
    elif args.cold_start:
        graph = PolicyGraph()
        print("Cold start: empty policy graph")
        if RESULTS_AGENSFLOW.exists():
            RESULTS_AGENSFLOW.unlink()
        start_epoch = 1
        prior_runs = 0
    else:
        # Warm start from chunk 6.5.
        if not CHUNK6_GRAPH_PATH.exists():
            print(
                f"ERROR: chunk-6.5 graph not found at {CHUNK6_GRAPH_PATH}. "
                f"Run experiments.e03_production_traffic.run first, "
                f"or pass --cold-start.",
                file=sys.stderr,
            )
            return 1
        graph = load_policy_graph(CHUNK6_GRAPH_PATH)
        print(f"Warm-started from chunk 6.5 graph at {CHUNK6_GRAPH_PATH} "
              f"({len(graph)} nodes)")
        if RESULTS_AGENSFLOW.exists() and not args.resume:
            RESULTS_AGENSFLOW.unlink()
        start_epoch = 1
        prior_runs = 0

    print()
    print(f"Running chunk-8 topology skip-X experiment.")
    print(f"  tasks per epoch:       {len(tasks)}")
    print(f"  epochs (this session): {start_epoch} .. {args.epochs}")
    print(f"  total runs estimated:  {len(tasks) * (args.epochs - start_epoch + 1)}")
    print(f"  judge model:           {args.judge_model}")
    print(f"  reliability weight:    {args.reliability_weight}")
    print(f"  enable_skip:           {enable_skip}"
          f"{' (degenerated to chunk-7 behavior)' if not enable_skip else ''}")
    print(f"  output dir:            {RESULTS_DIR}")
    print()

    load_dotenv()
    client = OpenRouterClient()
    reward_config = RewardConfig()

    state = HarnessState(
        policy_graph=graph,
        client=client,
        reward_config=reward_config,
        judge_model=args.judge_model,
        rubric=DEFAULT_RUBRIC,
    )

    all_records: list[TrajectoryRecord] = []
    cumulative_index = prior_runs

    for epoch in range(start_epoch, args.epochs + 1):
        # Per-epoch shuffle, deterministically seeded so reruns are reproducible.
        rng = random.Random(args.shuffle_seed + epoch)
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)

        print(f"\n========== EPOCH {epoch}/{args.epochs} ==========")
        epoch_records = run_full_benchmark(
            epoch_tasks,
            state=state,
            max_steps=args.max_steps,
            confidence_threshold=args.confidence_threshold,
            epoch=epoch,
            run_index_offset=cumulative_index,
            reliability_weight=args.reliability_weight,
            enable_skip=enable_skip,
            on_record=lambda rec: _append_record(rec, RESULTS_AGENSFLOW),
        )
        all_records.extend(epoch_records)
        cumulative_index += len(epoch_records)

        # Per-epoch checkpoint: graph snapshot + final graph.
        snapshot_path = SNAPSHOTS_DIR / f"policy_graph_epoch_{epoch:02d}.pkl"
        save_policy_graph(graph, snapshot_path)
        save_policy_graph(graph, GRAPH_PATH)
        print(f"\nEpoch {epoch} complete. Graph: {len(graph)} nodes, "
              f"snapshot saved to {snapshot_path}")

    # Re-aggregate ALL records (including any prior --resume rounds).
    all_dicts = _load_existing_records(RESULTS_AGENSFLOW)
    valid_fields = {f for f in TrajectoryRecord.__dataclass_fields__}
    merged_records = [
        TrajectoryRecord(**{k: v for k, v in d.items() if k in valid_fields})
        for d in all_dicts
    ]

    rendered = _render_results(merged_records, graph)
    if args.no_write:
        print(rendered)
    else:
        _splice_into_results(rendered)
        print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
