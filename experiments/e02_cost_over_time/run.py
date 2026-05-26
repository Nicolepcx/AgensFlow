"""
Cost-over-time experiment entry point.

Usage:
    # Smoke version
    python -m experiments.e02_cost_over_time.run --tasks B1_tcp_udp --runs 5

    # Full benchmark
    python -m experiments.e02_cost_over_time.run

    # Resume from saved graph
    python -m experiments.e02_cost_over_time.run --resume

    # Reset and start fresh
    python -m experiments.e02_cost_over_time.run --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from agensflow import PolicyGraph, load_policy_graph, save_policy_graph

from experiments.e01_regime_validation.tasks import CATEGORY_B_TASKS
from experiments.e02_cost_over_time.harness import (
    TrajectoryCell,
    cell_to_dict,
    run_task_trajectory,
)

THIS_DIR = Path(__file__).parent
GRAPH_PATH = THIS_DIR / "policy_graph.pkl"
TRACE_PATH = THIS_DIR / "results_raw.jsonl"
RESULTS_PATH = THIS_DIR / "RESULTS.md"

RESULTS_BEGIN = "<!-- RESULTS:BEGIN (auto-generated; do not edit) -->"
RESULTS_END = "<!-- RESULTS:END -->"

DEFAULT_RUNS = 15
DEFAULT_CONFIDENCE = 3
DEFAULT_MAX_STEPS = 12


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _per_task_aggregates(cells: list[TrajectoryCell]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[TrajectoryCell]] = {}
    for c in cells:
        by_task.setdefault(c.task_id, []).append(c)

    for task_id, task_cells in by_task.items():
        task_cells = sorted(task_cells, key=lambda x: x.run_index)
        n_runs = len(task_cells)
        out[task_id] = {
            "n_runs": n_runs,
            "tokens_per_run": [c.total_tokens for c in task_cells],
            "rewards": [c.reward if c.reward is not None else 0.0 for c in task_cells],
            "judgements": [c.judgement for c in task_cells],
            "validation_retries": [c.validation_retries for c in task_cells],
            "n_confident_nodes": [c.n_confident_nodes or 0 for c in task_cells],
            "policy_graph_size": [c.policy_graph_size or 0 for c in task_cells],
            "n_success": sum(1 for c in task_cells if c.judgement == "success"),
            "n_partial": sum(1 for c in task_cells if c.judgement == "partial"),
            "n_failure": sum(1 for c in task_cells if c.judgement == "failure"),
            "n_errors": sum(1 for c in task_cells if c.error is not None),
        }
    return out


def _trajectory_window_stats(
    cells: list[TrajectoryCell],
    *,
    early_window: tuple[int, int],
    late_window: tuple[int, int],
) -> dict[str, Any]:
    """
    Compute the headline early-vs-late comparison.

    Windows are 1-based, inclusive.
    """
    def _window_cells(window: tuple[int, int]) -> list[TrajectoryCell]:
        lo, hi = window
        return [c for c in cells if lo <= c.run_index <= hi]

    def _summary(window_cells: list[TrajectoryCell]) -> dict[str, Any]:
        if not window_cells:
            return {"n": 0, "tokens": 0.0, "tokens_per_success": None,
                    "success_rate": 0.0}
        total_tokens = sum(c.total_tokens for c in window_cells)
        n_success = sum(1 for c in window_cells if c.judgement == "success")
        return {
            "n": len(window_cells),
            "tokens_per_run": total_tokens / len(window_cells),
            "tokens_per_success": (
                total_tokens / n_success if n_success > 0 else None
            ),
            "success_rate": n_success / len(window_cells),
        }

    early = _summary(_window_cells(early_window))
    late = _summary(_window_cells(late_window))

    descent_pct: float | None = None
    if (
        early.get("tokens_per_run", 0) and
        late.get("tokens_per_run") is not None and
        early["tokens_per_run"] > 0
    ):
        descent_pct = (
            (early["tokens_per_run"] - late["tokens_per_run"])
            / early["tokens_per_run"] * 100.0
        )

    return {
        "early_window": list(early_window),
        "late_window": list(late_window),
        "early": early,
        "late": late,
        "descent_pct": descent_pct,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render(
    cells: list[TrajectoryCell],
    *,
    n_runs: int,
    confidence_threshold: int,
) -> str:
    per_task = _per_task_aggregates(cells)

    early_window = (1, max(1, n_runs // 3))
    late_window = (max(1, n_runs - (n_runs // 3) + 1), n_runs)

    headline = _trajectory_window_stats(
        cells, early_window=early_window, late_window=late_window
    )

    lines: list[str] = [RESULTS_BEGIN, "", "## Results", ""]
    lines.append(f"**Total cells:** {len(cells)}  ")
    lines.append(f"**Runs per task:** {n_runs}  ")
    lines.append(f"**Tasks:** {sorted(per_task.keys())}  ")
    lines.append(f"**Confidence threshold:** {confidence_threshold}")
    lines.append("")

    # --- Headline test ---
    lines.append("### Headline: early-vs-late comparison")
    lines.append("")
    lines.append(
        f"Early window: runs {headline['early_window'][0]}-{headline['early_window'][1]}.  "
        f"Late window: runs {headline['late_window'][0]}-{headline['late_window'][1]}."
    )
    lines.append("")
    lines.append("| Window | n cells | tokens/run | tokens/success | success-rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, w in [("early", headline["early"]), ("late", headline["late"])]:
        tps = w.get("tokens_per_success")
        tps_str = f"{int(tps)}" if tps is not None else "—"
        lines.append(
            f"| {label} | {w['n']} | "
            f"{int(w.get('tokens_per_run', 0))} | "
            f"{tps_str} | "
            f"{w['success_rate'] * 100:.0f}% |"
        )
    lines.append("")
    if headline["descent_pct"] is not None:
        sign = "+" if headline["descent_pct"] >= 0 else ""
        lines.append(
            f"**Descent in tokens/run from early to late: "
            f"{sign}{headline['descent_pct']:.1f}%**"
        )
        lines.append("")
        threshold_pct = 15.0
        if headline["descent_pct"] >= threshold_pct:
            lines.append(
                f"✓ Pre-registered threshold (≥{threshold_pct:.0f}% descent) **met**."
            )
        else:
            lines.append(
                f"✗ Pre-registered threshold (≥{threshold_pct:.0f}% descent) **not met**."
            )
        lines.append("")

    # --- Per-task trajectory ---
    lines.append("### Per-task trajectory")
    lines.append("")
    lines.append(
        "| task | n_runs | tokens trajectory | n_success | n_partial | n_failure | "
        "validation retries (total) |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for task_id in sorted(per_task.keys()):
        t = per_task[task_id]
        traj_str = ", ".join(str(x) for x in t["tokens_per_run"])
        retries_total = sum(t["validation_retries"])
        lines.append(
            f"| `{task_id}` | {t['n_runs']} | {traj_str} | "
            f"{t['n_success']} | {t['n_partial']} | {t['n_failure']} | {retries_total} |"
        )
    lines.append("")

    # --- Reward trajectory ---
    lines.append("### Per-task reward trajectory")
    lines.append("")
    lines.append("| task | rewards (run 1 → run N) |")
    lines.append("|---|---|")
    for task_id in sorted(per_task.keys()):
        rewards = per_task[task_id]["rewards"]
        rewards_str = ", ".join(f"{r:+.2f}" for r in rewards)
        lines.append(f"| `{task_id}` | {rewards_str} |")
    lines.append("")

    # --- Policy graph growth ---
    lines.append("### Policy graph growth")
    lines.append("")
    lines.append("| task | nodes (run 1 → run N) | confident nodes (run 1 → run N) |")
    lines.append("|---|---|---|")
    for task_id in sorted(per_task.keys()):
        nodes_str = ", ".join(str(x) for x in per_task[task_id]["policy_graph_size"])
        conf_str = ", ".join(str(x) for x in per_task[task_id]["n_confident_nodes"])
        lines.append(f"| `{task_id}` | {nodes_str} | {conf_str} |")
    lines.append("")

    # --- Errors ---
    error_cells = [c for c in cells if c.error]
    if error_cells:
        lines.append("### Execution errors")
        lines.append("")
        for c in error_cells:
            lines.append(f"- `{c.task_id}` × run {c.run_index}")
            lines.append("  ```")
            err_text = (c.error or "").strip()[:400]
            lines.append(f"  {err_text}")
            lines.append("  ```")
        lines.append("")

    lines.append(RESULTS_END)
    return "\n".join(lines)


def _splice_into_results_md(rendered: str) -> None:
    if not RESULTS_PATH.exists():
        # First run — write predictions stub + results.
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
    """Pre-registered predictions, written before the first run."""
    return """# Experiment 02 — Cost-over-time learning trajectory

Pre-registered predictions written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-04
**Author:** Nicole Königstein
**Framework version:** AgensFlow 0.1.0 (chunk 4.5 — dynamic routing)

### Primary prediction

Across the 4 Category B tasks run 15 times each (60 cells total), mean
tokens-per-task in the late window (runs 11-15) will be **at least 15%
lower** than the early window (runs 1-5). The descent is the empirical
signature of policy-driven dynamic routing learning to skip wasteful actions.

### Secondary prediction

The router will override the rule-based prior (graph_recommendation chosen
over rule_based_prior) at least once across the 60 runs. If never, either
the confidence threshold is too high or the signature folding is producing
too many singleton nodes for value to accumulate at any one signature.

### Tertiary prediction

Success rate in runs 11-15 will be **no lower** than success rate in runs
1-5 (i.e., the policy isn't degrading the system, only making it cheaper).

### What would falsify

- Primary fails: the cost-over-time claim is unsupported on this benchmark.
  Possible reasons: confidence_threshold too high, reward signal not strong
  enough to differentiate good from bad routes, signature folding too coarse
  or too fine, single-trial variance overwhelming the signal.
- Secondary fails: the routing infrastructure is correct but never engages.
  Tuning required: lower confidence_threshold or denser signatures.
- Tertiary fails: the policy is degrading the system. Reward function is
  miscalibrated or the policy graph is overfitting to early rewards.
"""


def _dump_raw(cells: list[TrajectoryCell]) -> None:
    with TRACE_PATH.open("w") as f:
        for c in cells:
            f.write(json.dumps(cell_to_dict(c), default=str) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the cost-over-time experiment."
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task ids to run (default: all Category B).",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Sequential runs per task (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--confidence-threshold", type=int, default=DEFAULT_CONFIDENCE,
        help="Min visits to a signature before graph overrides rule-based prior.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Max routing decisions per run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from a saved policy graph.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete saved graph + results before starting.",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Skip RESULTS.md write; print to stdout.",
    )
    args = parser.parse_args()

    if args.reset:
        if GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
            print(f"Reset: removed {GRAPH_PATH}")
        if TRACE_PATH.exists():
            TRACE_PATH.unlink()
            print(f"Reset: removed {TRACE_PATH}")

    # Pick tasks.
    all_b = list(CATEGORY_B_TASKS)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in all_b if t.id in wanted]
        if not tasks:
            print(f"No Category B tasks match: {sorted(wanted)}", file=sys.stderr)
            return 2
    else:
        tasks = all_b

    # Load or initialise the policy graph.
    if args.resume and GRAPH_PATH.exists():
        graph = load_policy_graph(GRAPH_PATH)
        print(f"Resumed from {GRAPH_PATH} ({len(graph)} nodes)")
    else:
        graph = PolicyGraph()
        print(f"Starting with an empty policy graph")

    print(f"\nRunning {len(tasks)} task(s) × {args.runs} sequential runs each")
    print(f"  tasks: {[t.id for t in tasks]}")
    print(f"  confidence_threshold: {args.confidence_threshold}")
    print(f"  max_steps: {args.max_steps}")
    print()

    all_cells: list[TrajectoryCell] = []
    for task in tasks:
        print(f"=== Task {task.id} ===")
        cells = run_task_trajectory(
            task,
            policy_graph=graph,
            n_runs=args.runs,
            confidence_threshold=args.confidence_threshold,
            max_steps=args.max_steps,
        )
        all_cells.extend(cells)
        # Persist after each task so a crash mid-experiment doesn't lose data.
        save_policy_graph(graph, GRAPH_PATH)
        print()

    rendered = _render(
        all_cells,
        n_runs=args.runs,
        confidence_threshold=args.confidence_threshold,
    )

    if args.no_write:
        print(rendered)
    else:
        _splice_into_results_md(rendered)
        _dump_raw(all_cells)
        print(f"\nWrote results to {RESULTS_PATH}")
        print(f"Raw cells dumped to {TRACE_PATH}")
        print(f"Graph saved to {GRAPH_PATH} ({len(graph)} nodes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
