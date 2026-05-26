"""
Entry point: run the full benchmark and write results into RESULTS.md.

Usage:
    python -m experiments.e01_regime_validation.run

Optional:
    --tasks TASK_ID,TASK_ID    Run only the specified tasks (e.g., for smoke tests).
    --configs CONFIG,CONFIG    Run only the specified configurations.
    --no-write                 Print results to stdout without modifying RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.e01_regime_validation.harness import (
    CONFIGURATIONS,
    CellResult,
    cell_to_dict,
    run_full_benchmark,
)
from experiments.e01_regime_validation.tasks import ALL_TASKS

RESULTS_PATH = Path(__file__).parent / "RESULTS.md"
TRACE_DUMP_PATH = Path(__file__).parent / "results_raw.jsonl"
RESULTS_BEGIN_MARKER = "<!-- RESULTS:BEGIN (auto-generated; do not edit) -->"
RESULTS_END_MARKER = "<!-- RESULTS:END -->"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _aggregate(results: list[CellResult]) -> dict[str, Any]:
    by_cat_cfg: dict[tuple[str, str], list[CellResult]] = defaultdict(list)
    by_cfg: dict[str, list[CellResult]] = defaultdict(list)
    for r in results:
        by_cat_cfg[(r.category, r.configuration)].append(r)
        by_cfg[r.configuration].append(r)

    def _cell_summary(cells: list[CellResult]) -> dict[str, Any]:
        n = len(cells)
        n_success = sum(1 for c in cells if c.judgement == "success")
        n_partial = sum(1 for c in cells if c.judgement == "partial")
        n_failure = sum(1 for c in cells if c.judgement == "failure")
        total_tokens = sum(c.total_tokens for c in cells)
        total_calls = sum(c.n_calls for c in cells)
        total_retries = sum(c.validation_retries for c in cells)
        n_flagged_missing = sum(1 for c in cells if c.flagged_missing_evidence)
        return {
            "n": n,
            "success": n_success,
            "partial": n_partial,
            "failure": n_failure,
            "success_rate": (n_success / n) if n else 0.0,
            "total_tokens": total_tokens,
            "tokens_per_run": (total_tokens / n) if n else 0.0,
            "tokens_per_success": (total_tokens / n_success) if n_success else None,
            "total_calls": total_calls,
            "total_validation_retries": total_retries,
            "flagged_missing_evidence": n_flagged_missing,
        }

    return {
        "by_category_and_config": {
            f"{cat}::{cfg}": _cell_summary(cells)
            for (cat, cfg), cells in sorted(by_cat_cfg.items())
        },
        "by_config_overall": {
            cfg: _cell_summary(cells) for cfg, cells in sorted(by_cfg.items())
        },
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _fmt_int_or_dash(x: Any) -> str:
    return "—" if x is None else f"{int(x)}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _render_results_md(results: list[CellResult], agg: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(RESULTS_BEGIN_MARKER)
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"Total cells run: {len(results)}")
    lines.append("")

    # Per-category breakdown.
    for category in ["A", "B", "C"]:
        lines.append(f"### Category {category}")
        lines.append("")
        lines.append(
            "| Configuration | N | success | partial | failure | "
            "success-rate | tokens/run | tokens/success | retries | flagged-missing |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for cfg in CONFIGURATIONS:
            key = f"{category}::{cfg}"
            s = agg["by_category_and_config"].get(key)
            if s is None:
                continue
            lines.append(
                f"| `{cfg}` | {s['n']} | {s['success']} | {s['partial']} | "
                f"{s['failure']} | {_fmt_pct(s['success_rate'])} | "
                f"{int(s['tokens_per_run'])} | "
                f"{_fmt_int_or_dash(s['tokens_per_success'])} | "
                f"{s['total_validation_retries']} | "
                f"{s['flagged_missing_evidence']} |"
            )
        lines.append("")

    # Overall.
    lines.append("### Overall (all categories)")
    lines.append("")
    lines.append(
        "| Configuration | N | success-rate | tokens/run | tokens/success | retries |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cfg in CONFIGURATIONS:
        s = agg["by_config_overall"].get(cfg)
        if s is None:
            continue
        lines.append(
            f"| `{cfg}` | {s['n']} | {_fmt_pct(s['success_rate'])} | "
            f"{int(s['tokens_per_run'])} | "
            f"{_fmt_int_or_dash(s['tokens_per_success'])} | "
            f"{s['total_validation_retries']} |"
        )
    lines.append("")

    # Per-task table.
    lines.append("### Per-task detail")
    lines.append("")
    lines.append(
        "| task | category | configuration | judgement | tokens | calls | retries | "
        "regime used |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| `{r.task_id}` | {r.category} | `{r.configuration}` | "
            f"**{r.judgement}** | {r.total_tokens} | {r.n_calls} | "
            f"{r.validation_retries} | {r.regime_used or '—'} |"
        )
    lines.append("")

    # Errors.
    error_cells = [r for r in results if r.error]
    if error_cells:
        lines.append("### Execution errors")
        lines.append("")
        for r in error_cells:
            lines.append(f"- `{r.task_id}` × `{r.configuration}`")
            lines.append("  ```")
            lines.append(f"  {r.error.strip()[:500]}")
            lines.append("  ```")
        lines.append("")

    lines.append(RESULTS_END_MARKER)
    return "\n".join(lines)


def _splice_into_results_md(rendered: str) -> None:
    if not RESULTS_PATH.exists():
        # Should not happen — RESULTS.md ships pre-registered. Be defensive.
        RESULTS_PATH.write_text(rendered + "\n")
        return
    text = RESULTS_PATH.read_text()
    if RESULTS_BEGIN_MARKER in text and RESULTS_END_MARKER in text:
        before, _, rest = text.partition(RESULTS_BEGIN_MARKER)
        _, _, after = rest.partition(RESULTS_END_MARKER)
        new_text = before.rstrip() + "\n\n" + rendered + "\n" + after.lstrip()
    else:
        new_text = text.rstrip() + "\n\n" + rendered + "\n"
    RESULTS_PATH.write_text(new_text)


def _dump_raw(results: list[CellResult]) -> None:
    with TRACE_DUMP_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(cell_to_dict(r), default=str) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the regime-validation benchmark."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task ids to run (default: all).",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated configurations to run (default: all).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not modify RESULTS.md; print results to stdout instead.",
    )
    args = parser.parse_args()

    tasks = list(ALL_TASKS)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        missing = wanted - {t.id for t in tasks}
        if missing:
            print(f"Unknown task ids: {sorted(missing)}", file=sys.stderr)
            return 2

    configs = list(CONFIGURATIONS)
    if args.configs:
        wanted_cfg = args.configs.split(",")
        unknown = [c for c in wanted_cfg if c not in CONFIGURATIONS]
        if unknown:
            print(f"Unknown configurations: {unknown}", file=sys.stderr)
            return 2
        configs = wanted_cfg  # type: ignore[assignment]

    print(f"Running {len(tasks)} task(s) × {len(configs)} configuration(s)")
    print(f"Tasks:   {[t.id for t in tasks]}")
    print(f"Configs: {configs}")

    results = run_full_benchmark(tasks=tasks, configurations=configs)
    agg = _aggregate(results)
    rendered = _render_results_md(results, agg)

    if args.no_write:
        print("\n" + rendered)
    else:
        _splice_into_results_md(rendered)
        _dump_raw(results)
        print(f"\nWrote results to {RESULTS_PATH}")
        print(f"Raw cells dumped to {TRACE_DUMP_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
