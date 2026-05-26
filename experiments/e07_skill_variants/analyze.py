"""
Per-epoch analysis of a chunk-9 sweep.

Reads `results_agensflow.jsonl` (or any compatible jsonl) and produces an
enriched per-epoch table matching the chunks 6/7/8 RESULTS.md format:

    | ep | n | tokens | RULER | reward | retries | skip% | solvers/run |

Beyond `_render_results` in `run.py`, this adds:
  - **skip%** — fraction of all routing decisions that were `skip:X`. Low
    early (cold-start exploration), should rise as policy converges to
    confident "skip the verifier-on-easy-Q" / "skip the second search"
    decisions.
  - **solvers/run** — average count of solver_X invocations per run. Drops
    from "every variant" (~9) at cold start toward the converged 1-2 once
    the substrate locks onto per-class winners.

Safe to run while the sweep is in flight — reads jsonl line-by-line, no
locking required.

Usage:

    # Default — analyze the postconfig_v1 sweep:
    python -m experiments.e07_skill_variants.analyze

    # Custom output dir:
    python -m experiments.e07_skill_variants.analyze \\
        --jsonl experiments/e07_skill_variants/postconfig_v1/results_agensflow.jsonl

    # Render markdown table directly (for splicing into RESULTS.md):
    python -m experiments.e07_skill_variants.analyze --markdown
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


THIS_DIR = Path(__file__).parent
DEFAULT_JSONL = THIS_DIR / "postconfig_v1" / "results_agensflow.jsonl"


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Records file not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _record_skip_count(rec: dict[str, Any]) -> tuple[int, int]:
    """Return (n_skips, n_decisions) for one record's path. A "decision"
    here is any path entry — including skip:X — since each is a routing
    decision the policy made."""
    path = rec.get("path") or []
    n_skips = sum(1 for p in path if isinstance(p, str) and p.startswith("skip:"))
    return n_skips, len(path)


def _record_solver_count(rec: dict[str, Any]) -> int:
    """Count solver_X invocations (excluding skips) in one record's path.

    We count any agent name starting with "solver" but NOT "skip:" — that
    matches both the base "solver" agent and every solver_X variant.
    """
    path = rec.get("path") or []
    return sum(
        1 for p in path
        if isinstance(p, str)
        and p.startswith("solver")
        and not p.startswith("skip:")
    )


def aggregate_per_epoch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_epoch[int(r.get("epoch", 0))].append(r)
    rows: list[dict[str, Any]] = []
    for epoch in sorted(by_epoch.keys()):
        recs = by_epoch[epoch]
        ok = [r for r in recs if r.get("error") is None]
        n = len(recs)
        n_errors = n - len(ok)

        # Token / RULER / reward / retries — averaged over successful runs.
        tokens_avg = int(mean(r["total_tokens"] for r in ok)) if ok else 0
        ruler_avg = (
            mean(r["ruler_score"] for r in ok if r.get("ruler_score") is not None)
            if ok else 0.0
        )
        reward_avg = (
            mean(r["hybrid_reward"] for r in ok if r.get("hybrid_reward") is not None)
            if ok else 0.0
        )
        retries_avg = mean(r.get("validation_retries", 0) for r in ok) if ok else 0.0

        # skip% — total skip events / total path entries across all runs.
        # Pooled rather than per-run-then-averaged so single short paths
        # don't disproportionately swing the rate.
        skip_total, decisions_total = 0, 0
        for r in ok:
            ns, nd = _record_skip_count(r)
            skip_total += ns
            decisions_total += nd
        skip_pct = (skip_total / decisions_total) if decisions_total else 0.0

        # solvers/run — mean across runs of (solver_X invocation count).
        solvers_per_run = (
            mean(_record_solver_count(r) for r in ok) if ok else 0.0
        )

        rows.append({
            "epoch": epoch,
            "n": n,
            "n_errors": n_errors,
            "tokens_avg": tokens_avg,
            "ruler_avg": ruler_avg,
            "reward_avg": reward_avg,
            "retries_avg": retries_avg,
            "skip_pct": skip_pct,
            "solvers_per_run": solvers_per_run,
        })
    return rows


def render_text(rows: list[dict[str, Any]]) -> str:
    """Compact aligned-text table for stdout — easier to read at a glance
    than the markdown variant."""
    lines: list[str] = []
    lines.append(
        f"{'ep':>3}  {'n':>3}  {'err':>3}  {'tokens':>7}  {'RULER':>5}  "
        f"{'reward':>7}  {'retries':>7}  {'skip%':>5}  {'solvers/run':>11}"
    )
    lines.append("-" * 76)
    for r in rows:
        lines.append(
            f"{r['epoch']:>3}  {r['n']:>3}  {r['n_errors']:>3}  "
            f"{r['tokens_avg']:>7,}  {r['ruler_avg']:>5.2f}  "
            f"{r['reward_avg']:>+7.2f}  {r['retries_avg']:>7.2f}  "
            f"{r['skip_pct']:>4.0%}  {r['solvers_per_run']:>11.1f}"
        )
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        token_delta_pct = (
            (first["tokens_avg"] - last["tokens_avg"])
            / max(1, first["tokens_avg"]) * 100
        )
        ruler_delta = last["ruler_avg"] - first["ruler_avg"]
        skip_delta = last["skip_pct"] - first["skip_pct"]
        solvers_delta = last["solvers_per_run"] - first["solvers_per_run"]
        lines.append("")
        lines.append(
            f"Δ epoch {first['epoch']} → {last['epoch']}: "
            f"tokens {token_delta_pct:+.0f}%  "
            f"RULER {ruler_delta:+.2f}  "
            f"skip {skip_delta:+.0%}  "
            f"solvers/run {solvers_delta:+.1f}"
        )
    return "\n".join(lines)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown table suitable for splicing into RESULTS.md — same column
    set as the text variant but pipe-delimited."""
    lines: list[str] = []
    lines.append(
        "| ep | n | err | tokens | RULER | reward | retries | skip% | solvers/run |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['epoch']} | {r['n']} | {r['n_errors']} | "
            f"{r['tokens_avg']:,} | {r['ruler_avg']:.2f} | "
            f"{r['reward_avg']:+.2f} | {r['retries_avg']:.2f} | "
            f"{r['skip_pct']:.0%} | {r['solvers_per_run']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-epoch analysis of a chunk-9 sweep."
    )
    parser.add_argument(
        "--jsonl", type=Path, default=DEFAULT_JSONL,
        help="Path to results_agensflow.jsonl (default: postconfig_v1).",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Emit a markdown table instead of the aligned-text view.",
    )
    args = parser.parse_args(argv)

    records = _load_records(args.jsonl)
    rows = aggregate_per_epoch(records)
    if not rows:
        print(f"No records found in {args.jsonl}")
        return 1
    if args.markdown:
        print(render_markdown(rows))
    else:
        print(f"Source: {args.jsonl}  ({len(records)} records, "
              f"{len(rows)} epoch{'s' if len(rows) != 1 else ''} represented)")
        print()
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
