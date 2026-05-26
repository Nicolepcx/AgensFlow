"""
Chunk 9 entry point — declarative skill-definition variants.

Extends chunk 8's substrate (sustained learning + skip-X topology
learning) by enriching the solver action space from "model variants of
one hardcoded prompt" to "(skill card × model) cross product." Three
solver SKILL.md cards (concise / chain_of_thought / evidence_first)
paired with three model bindings (haiku / fast / mini) gives 9 distinct
solver actions in the policy graph.

The systems-perspective hypothesis being tested:

  (a) **Per-class differentiation**: at least 4 of 8 scenario classes
      converge to a (skill, model) combination that is non-trivially
      different from "default skill + most-capable model" — i.e. a
      cheaper or differently-constrained combination wins on RULER ×
      cost.

  (b) **Cost optimization through skill-as-constraint**: at least one
      class converges to a (cheaper model, tighter skill) pair that
      produces ≥20% lower tokens than the same model paired with the
      default skill spec, at preserved RULER head-to-head.

  (c) **Reliability differentiation**: per-edge retry rate, reward
      variance, and token variance differ meaningfully across (skill,
      model) pairs at the same signature — the systems-level reliability
      profile is observable, not noise.

  (d) **Stable interaction surface**: re-runs of the same task pool
      from the chunk-9 final graph (frozen) produce similar per-class
      winning combinations — discovery is reproducible learning, not
      lucky exploration.

Usage:

    # Standard 8-epoch run (warm-start from chunk-8 graph):
    python -m experiments.e07_skill_variants.run

    # Smoke test:
    python -m experiments.e07_skill_variants.run --tasks C1.1,C2.1 --epochs 2

    # Cold start (don't warm from chunk 8):
    python -m experiments.e07_skill_variants.run --cold-start

    # Resume after a crash:
    python -m experiments.e07_skill_variants.run --resume

    # Stability replay against the frozen final graph (after main sweep):
    python -m experiments.e07_skill_variants.run --frozen --epochs 2 \\
        --output-suffix stability_replay
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
from agensflow.config import load_config
from agensflow.registry import default_registry
from agensflow.runtime.client import OpenRouterClient

from agensflow.runtime.governance import GovernancePolicy
from agensflow.runtime.preflight import run_preflight_checks
from agensflow.runtime.report import RunReport, SessionReport

from experiments.e03_production_traffic.harness import (
    DEFAULT_JUDGE_MODEL,
    HarnessState,
    TrajectoryRecord,
    record_to_dict,
    run_full_benchmark,
)
from experiments.e03_production_traffic.tasks import ALL_TASKS
from experiments.e07_skill_variants.activation import (
    build_chunk9_activation_plan,
)


# ----- Paths -----
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR
RESULTS_AGENSFLOW = RESULTS_DIR / "results_agensflow.jsonl"
GRAPH_PATH = RESULTS_DIR / "policy_graph.pkl"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
RESULTS_PATH = RESULTS_DIR / "RESULTS.md"

# Skills directory — repo root by default.
REPO_ROOT = THIS_DIR.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

# Warm-start sources, in priority order:
#   1. Chunk-8 final graph (the richest substrate state available).
#   2. Chunk-6.5 graph (fallback if chunk 8 hasn't been run).
CHUNK8_GRAPH_PATH = REPO_ROOT / "experiments" / "e05_topology_skip" / "policy_graph.pkl"
CHUNK6_GRAPH_PATH = (
    REPO_ROOT / "experiments" / "e03_production_traffic" / "policy_graph.pkl"
)

# Tasks excluded from the chunk-9 pool. C7.1 hits LangGraph recursion
# limit deterministically (documented in chunk 6.5 RESULTS.md).
EXCLUDE_TASK_IDS = {"C7.1"}

DEFAULT_EPOCHS = 8
DEFAULT_SHUFFLE_SEED = 20260507  # same seed family as chunk 7+8 for direct A/B


# --------------------------------------------------------------------------- #
# Per-epoch I/O helpers
# --------------------------------------------------------------------------- #


def _append_record(record: TrajectoryRecord, path: Path) -> None:
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
    if not records:
        return 0
    by_epoch: dict[int, set[str]] = defaultdict(set)
    for r in records:
        by_epoch[int(r.get("epoch", 0))].add(r["task_id"])
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

        # Pooled skip% — total skip events / total path entries across
        # all runs in this epoch. Pooled (not per-run-then-averaged) so
        # short paths don't disproportionately swing the rate.
        skip_total, decisions_total = 0, 0
        for r in ok:
            ns = sum(1 for p in r.path if p.startswith("skip:"))
            skip_total += ns
            decisions_total += len(r.path)
        skip_pct = (skip_total / decisions_total) if decisions_total else 0.0

        # solvers/run — average count of solver_X invocations per run
        # (excluding skip:solver_X). Direct measure of variant pool
        # convergence: starts ~9 at cold start, drops toward 1-2 once
        # the substrate locks onto per-class winners.
        def _solver_count(r: TrajectoryRecord) -> int:
            return sum(
                1 for p in r.path
                if p.startswith("solver") and not p.startswith("skip:")
            )
        solvers_per_run = mean(_solver_count(r) for r in ok) if ok else 0.0

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
            "skip_pct": skip_pct,
            "solvers_per_run": solvers_per_run,
        })
    return rows


def _render_results(records: list[TrajectoryRecord], graph: PolicyGraph) -> str:
    rows = _aggregate_per_epoch(records)
    lines: list[str] = []
    lines.append("<!-- RESULTS:BEGIN (auto-generated; do not edit) -->")
    lines.append("")
    lines.append("## Results — skill-definition variants (chunk 9)")
    lines.append("")
    lines.append(f"Total runs: {len(records)}")
    lines.append(
        f"Final graph: {len(graph)} nodes, "
        f"{sum(n.visits for n in graph.nodes.values())} total visits"
    )
    lines.append("")
    lines.append("### Per-epoch trajectory")
    lines.append("")
    lines.append(
        "| epoch | n | errors | tokens/run | RULER avg | reward avg | "
        "retries avg | skip% | solvers/run |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['epoch']} | {r['n']} | {r['n_errors']} | "
            f"{r['tokens_avg']:,} | {r['ruler_avg']:.2f} | "
            f"{r['reward_avg']:+.2f} | {r['retries_avg']:.2f} | "
            f"{r['skip_pct']:.0%} | {r['solvers_per_run']:.1f} |"
        )
    lines.append("")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        token_delta_pct = (
            (first["tokens_avg"] - last["tokens_avg"])
            / max(1, first["tokens_avg"]) * 100
        )
        ruler_delta = last["ruler_avg"] - first["ruler_avg"]
        skip_delta = last["skip_pct"] - first["skip_pct"]
        solvers_delta = last["solvers_per_run"] - first["solvers_per_run"]
        lines.append(
            f"**Δ epoch {first['epoch']} → {last['epoch']}:** "
            f"tokens {token_delta_pct:+.0f}% (positive = cheaper), "
            f"RULER {ruler_delta:+.2f} (positive = better quality), "
            f"skip {skip_delta:+.0%} (positive = more committed to skip), "
            f"solvers/run {solvers_delta:+.1f} (negative = pool converging)."
        )
        lines.append("")

    # Per-(skill_card × model) reliability profile from the final graph.
    lines.append("### Top per-(skill, model) edges in final graph")
    lines.append("")
    lines.append(
        "| signature | action (skill_card × model) | visits | mean reward | "
        "reward σ | mean tokens | token σ | failure rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    edges: list[tuple[str, str, int, float, float, float, float, float]] = []
    for sig, node in graph.nodes.items():
        for action in node.action_visits:
            n = node.action_visits[action]
            if n < 3:
                continue  # skip thin samples
            mean_r = node.action_value(action)
            var_r = node.action_reward_variance(action) ** 0.5
            mean_t = node.action_token_mean(action)
            var_t = node.action_token_variance(action) ** 0.5
            fail = node.action_failure_rate(action)
            regime = sig[0] if isinstance(sig, tuple) and sig else "?"
            edges.append((str(regime), action, n, mean_r, var_r, mean_t, var_t, fail))
    # Sort by visits desc, then by mean_tokens asc (cheapest popular edges first).
    edges.sort(key=lambda e: (-e[2], e[5]))
    for regime, action, n, mean_r, var_r, mean_t, var_t, fail in edges[:20]:
        lines.append(
            f"| {regime} | `{action}` | {n} | {mean_r:.2f} | "
            f"{var_r:.2f} | {int(mean_t)} | {int(var_t)} | {fail:.0%} |"
        )
    if not edges:
        lines.append("| (no edges with ≥3 visits yet) |  |  |  |  |  |  |  |")
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
    return """# Experiment 07 — Skill-definition variants (chunk 9)

Pre-registered predictions, written before the first run. Predictions are
preserved across runs; only the auto-generated results section is overwritten.

## Pre-registered predictions

**Date written:** 2026-05-07
**Author:** Nicole Königstein
**Framework version:** AgensFlow chunk 9 — declarative SKILL.md cards bound
to the policy graph as (skill_card × model) action cells, on top of the
chunk-8 substrate (sustained learning + topology skip-X).

### Setup

- Same 59-task pool as chunks 6/7/8 (minus C7.1 — recursion edge case).
- 8 epochs, shuffled per epoch with deterministic seed.
- Warm-start: chunk-8 final graph (richest substrate state available).
- Solver action space expanded from chunk-8's 3 model variants to chunk-9's
  9 (skill × model) cells = 3 SKILL.md cards × 3 model bindings (haiku /
  fast / mini).
- skip-X stays enabled (chunk-8 default).
- Same hybrid reward, λ=0.5 reliability weight (chunk-7 default).

### Solver SKILL.md cards under test

- `solver_concise` — minimum-viable answer; single-paragraph; no reasoning trace
- `solver_chain_of_thought` — explicit step-by-step inference, structured
  setup→reasoning→conclusion
- `solver_evidence_first` — citation-driven; cited evidence enumerated
  before any conclusion

The chunk-7/8 hardcoded solver was closest to chain_of_thought style;
chunk-9 adds two genuinely different behavioral envelopes.

### The systems-perspective hypothesis (4 parts)

**(a) Per-class differentiation.** At least 4 of 8 scenario classes converge
to a (skill, model) combination that is non-trivially different from the
"default skill + most-capable model" ground truth — a cheaper or
differently-constrained combination wins on RULER × cost.

**(b) Cost optimization through skill-as-constraint.** At least one class
converges to a (cheaper model, tighter skill) pair that produces ≥20% lower
tokens than the same model paired with the default skill spec, at preserved
RULER head-to-head. Direct test that SKILL.md acts as a *runtime constraint*
on model behavior — not as "better prompting."

**(c) Reliability differentiation.** Per-edge retry rate, reward variance,
and token variance differ meaningfully across (skill, model) pairs at the
same signature. The systems-level reliability profile is observable, not
noise. Specifically: at least 2 (skill, model) pairs at confident
signatures show retry-rate gaps ≥10 percentage points despite comparable
mean reward.

**(d) Stable interaction surface.** Re-runs of the same task pool against
the chunk-9 final graph (frozen, no learning) produce similar per-class
winning combinations — discovery is reproducible learning, not lucky
exploration. Tested via a stability-replay run after the main sweep
(`--frozen --epochs 2`). Predicted: ≥6 of 8 classes pick the same winning
(skill, model) combination in both replay epochs.

### What would falsify

- *(a) fails*: per-class winners are dominated by "default skill +
  most-capable model" — the framework didn't find better combinations.
  The systems claim doesn't survive: cards add nothing the substrate
  couldn't already discover from model variants alone.
- *(b) fails*: cost-saving combinations don't appear; the framework's
  "skill spec as constraint" framing is decorative.
- *(c) fails*: reliability metrics are noise — no robust per-edge
  differentiation. Either the substrate's tracking is too coarse or
  the reliability profile genuinely doesn't exist for this corpus.
- *(d) fails*: stability replay produces wildly different per-class
  winners. The substrate is over-fitting to traffic noise rather than
  discovering domain structure.

### Acknowledged constraints

- One corpus, one variant pool, one judge family. Cross-domain validation
  is a separate experiment (chunk 10+).
- Three SKILL.md cards is a small palette. The OSS user-story is "users
  ship as many SKILL.md alternatives as they want"; this experiment tests
  whether the substrate handles that surface, not how it scales to 20+
  cards.
- Same-family RULER judge bias is a known issue from chunk 6.5/7. Chunk-9
  reuses the chunk-8.5 cross-eval methodology to verify quality
  preservation independent of the in-condition RULER ranks.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run chunk 9 skill-definition variants experiment."
    )
    # ----- experiment-shape flags (no config equivalent) ----- #
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task ids to run.")
    parser.add_argument("--cold-start", action="store_true",
                        help="Don't warm from chunk 8; start with empty graph.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from this experiment's saved graph.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete saved graph + results before starting.")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--frozen", action="store_true",
                        help="Stability-replay mode: load graph but don't update it. "
                             "For testing prediction (d). Implies --resume semantics "
                             "but with a no-op backup.")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Subdirectory under e07_skill_variants/ for outputs "
                             "(useful for stability replay or ablations).")
    parser.add_argument("--no-write", action="store_true")

    # ----- YAML-driven config (the canonical source for runtime knobs) ----- #
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. Knobs from this file override "
             "the framework defaults. CLI flags below override the YAML "
             "for individual values when explicitly supplied.",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Load + print the merged AgensflowConfig and exit. "
             "Useful for inspecting what the runner WOULD use without "
             "spending any LLM tokens.",
    )

    # ----- per-knob CLI overrides (default=None → fall through to config) ----- #
    # Setting any of these on the CLI overrides the corresponding YAML
    # value; leaving them unset uses cfg.<section>.<knob>. Behavioral
    # parity with the pre-config-flag interface: passing
    # `--judge-model X` keeps working exactly as before.
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=int, default=None)
    parser.add_argument("--reliability-weight", type=float, default=None)

    # Tri-state booleans: None (use config) / True (explicit) / False (explicit).
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--no-skip", dest="enable_skip", action="store_false", default=None,
        help="Disable inline skip-X mechanism. Overrides cfg.router.enable_skip.",
    )
    skip_group.add_argument(
        "--enable-skip", dest="enable_skip", action="store_true", default=None,
        help="Enable inline skip-X mechanism. Overrides cfg.router.enable_skip.",
    )

    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--router-log", dest="enable_router_logging",
        action="store_true", default=None,
        help="Enable per-iteration router-loop logging. On exception, "
             "the in-flight router log is dumped to "
             "{output_dir}/router_logs/router_log_epNN_runNNNN_TASK.json. "
             "Useful for diagnosing GraphRecursionError without "
             "re-running the entire experiment. Adds small per-loop "
             "overhead but no LLM cost. Overrides cfg.router.enable_router_logging.",
    )
    log_group.add_argument(
        "--no-router-log", dest="enable_router_logging",
        action="store_false", default=None,
    )

    args = parser.parse_args(argv)

    # ----- Load config (defaults + optional user YAML) ----- #
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        cfg = load_config()

    # Resolve effective values: CLI overrides config when explicitly set.
    judge_model = (
        args.judge_model if args.judge_model is not None
        else (cfg.relative_judge.judge_model or DEFAULT_JUDGE_MODEL)
    )
    max_steps = (
        args.max_steps if args.max_steps is not None
        else max(cfg.router.max_steps, 18)  # chunk-9 needs ≥18 for variant pool
    )
    confidence_threshold = (
        args.confidence_threshold if args.confidence_threshold is not None
        else cfg.policy_graph.confidence_threshold
    )
    reliability_weight = (
        args.reliability_weight if args.reliability_weight is not None
        else cfg.policy_graph.reliability_weight
    )
    enable_skip = (
        args.enable_skip if args.enable_skip is not None
        else cfg.router.enable_skip
    )
    enable_router_logging = (
        args.enable_router_logging if args.enable_router_logging is not None
        else cfg.router.enable_router_logging
    )

    # --print-config: dump everything that WOULD be used and exit.
    # Cheap diagnostic — no I/O besides stdout, no LLM tokens, no
    # filesystem writes. Useful for verifying YAML overrides land
    # where you expect before launching a real sweep.
    if args.print_config:
        print()
        print("=== Effective configuration for this run ===")
        for f in cfg.__dataclass_fields__:
            print(f"\n[{f}]")
            section = getattr(cfg, f)
            for sf in section.__dataclass_fields__:
                print(f"  {sf} = {getattr(section, sf)!r}")
        print()
        print("=== Effective CLI-resolved values ===")
        print(f"  judge_model            = {judge_model!r}")
        print(f"  max_steps              = {max_steps!r}")
        print(f"  confidence_threshold   = {confidence_threshold!r}")
        print(f"  reliability_weight     = {reliability_weight!r}")
        print(f"  enable_skip            = {enable_skip!r}")
        print(f"  enable_router_logging  = {enable_router_logging!r}")
        return 0

    # Re-route output paths if --output-suffix is set.
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

    # Register the SKILL.md cards from the repo's `skills/` directory.
    # This is what makes the (skill_card × model) variants resolve to
    # the right system prompts in `make_solver`.
    n_cards = default_registry.register_cards_from_directory(
        SKILLS_DIR, overwrite=True
    )
    print(f"Loaded {n_cards} SKILL.md cards from {SKILLS_DIR}")
    for name in sorted(default_registry.card_names()):
        card = default_registry.get_card(name)
        print(f"  {name:<30s}  role={card.role}  ({len(card.instructions)} chars)")
    print()

    # Pick tasks.
    tasks = [t for t in ALL_TASKS if t.id not in EXCLUDE_TASK_IDS]
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"No tasks match: {sorted(wanted)}", file=sys.stderr)
            return 2

    # Load / start graph.
    if args.frozen:
        # Stability-replay mode: load the prior graph but use it without
        # updates. Implementation: load the graph, then back the
        # backup() function with a no-op via a thin wrapper. We do this
        # via state.policy_graph being a snapshot the harness still
        # writes to — but the snapshot doesn't get persisted at the
        # output dir. Simpler: just pass `--no-write` in stability runs.
        # For now, frozen mode just resumes from the graph and the user
        # is expected to inspect epoch records, not the graph diff.
        if not GRAPH_PATH.exists():
            print(f"ERROR: --frozen requires existing graph at {GRAPH_PATH}",
                  file=sys.stderr)
            return 1
        graph = load_policy_graph(GRAPH_PATH)
        print(f"Frozen replay from {GRAPH_PATH} ({len(graph)} nodes)")
        start_epoch = 1
        prior_runs = 0
    elif args.resume and GRAPH_PATH.exists():
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
        # Warm start. Prefer chunk-8 graph (richer substrate); fall back
        # to chunk-6.5 if chunk-8 hasn't been run yet.
        if CHUNK8_GRAPH_PATH.exists():
            graph = load_policy_graph(CHUNK8_GRAPH_PATH)
            print(f"Warm-started from chunk-8 graph at {CHUNK8_GRAPH_PATH} "
                  f"({len(graph)} nodes)")
        elif CHUNK6_GRAPH_PATH.exists():
            graph = load_policy_graph(CHUNK6_GRAPH_PATH)
            print(f"Warm-started from chunk-6.5 graph at {CHUNK6_GRAPH_PATH} "
                  f"(chunk-8 graph missing) ({len(graph)} nodes)")
        else:
            print(
                f"ERROR: no warm-start graph found at "
                f"{CHUNK8_GRAPH_PATH} or {CHUNK6_GRAPH_PATH}. "
                f"Pass --cold-start to start fresh.",
                file=sys.stderr,
            )
            return 1
        if RESULTS_AGENSFLOW.exists() and not args.resume:
            RESULTS_AGENSFLOW.unlink()
        start_epoch = 1
        prior_runs = 0

    print()
    print(f"Running chunk-9 skill-variants experiment.")
    print(f"  tasks per epoch:       {len(tasks)}")
    print(f"  epochs (this session): {start_epoch} .. {args.epochs}")
    print(f"  total runs estimated:  {len(tasks) * (args.epochs - start_epoch + 1)}")
    print(f"  judge model:           {judge_model}")
    print(f"  max_steps:             {max_steps}")
    print(f"  confidence_threshold:  {confidence_threshold}")
    print(f"  reliability_weight:    {reliability_weight}")
    print(f"  enable_skip:           {enable_skip}")
    print(f"  enable_router_logging: {enable_router_logging}")
    print(f"  frozen replay:         {args.frozen}")
    print(f"  output dir:            {RESULTS_DIR}")
    print()

    load_dotenv()

    # ----- Pre-flight: validate external dependencies before LLM cost ----- #
    # The chunk-9 disaster (~$5+ wasted on a throttled EXA endpoint) is
    # exactly what this catches. Costs ~$0.02 total; aborts the sweep with
    # a clear diagnosis if any dependency is misconfigured / out of quota.
    print("Running pre-flight checks...")
    preflight = run_preflight_checks(config=cfg.preflight)
    print(preflight.format_report())
    if not preflight.all_passed:
        print(
            "\nAborting before any LLM tokens are spent. "
            "Fix the failures above and re-run."
        )
        return 1
    print()

    client = OpenRouterClient()
    # Reward config from YAML — same instance used by the harness for
    # compute_hybrid_reward across all runs in the sweep.
    reward_config = cfg.reward

    # ----- Governance: per-run policy from YAML config ----- #
    # Each run gets its own GovernanceState (constructed fresh inside
    # run_one_task from this policy). Halts the affected run and skips
    # graph backup if any agent breaches the policy — substrate stays
    # clean, other runs in the epoch continue.
    governance_policy = cfg.governance
    print(
        f"Governance: max_consecutive_failures="
        f"{governance_policy.max_consecutive_failures_per_agent}, "
        f"max_calls_per_agent={governance_policy.max_calls_per_agent}, "
        f"halt_on_terminal_errors={governance_policy.halt_on_terminal_errors}"
    )
    REPORT_DIR = RESULTS_DIR / "run_reports"
    print(f"Per-task reports: {REPORT_DIR}/")
    print()

    # Rubric: prefer YAML override, else built-in DEFAULT_RUBRIC.
    rubric = cfg.relative_judge.rubric or DEFAULT_RUBRIC

    state = HarnessState(
        policy_graph=graph,
        client=client,
        reward_config=reward_config,
        judge_model=judge_model,
        rubric=rubric,
        # Chunk 11.A1: thread the full RelativeJudgeConfig so evidence_mode +
        # budget knobs land in the harness. When evidence_mode is "off"
        # (chunk-2..10 reproduction) the harness skips the
        # build_trajectory_evidence call entirely.
        ruler_config=cfg.ruler,
        # Chunk 11.C1: discounted backup. 1.0 by default → chunk-2..10
        # behavior; <1.0 enabled via cfg.policy_graph.gamma in YAML.
        backup_gamma=cfg.policy_graph.gamma,
    )

    cumulative_index = prior_runs
    for epoch in range(start_epoch, args.epochs + 1):
        rng = random.Random(args.shuffle_seed + epoch)
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)

        print(f"\n========== EPOCH {epoch}/{args.epochs} ==========")
        epoch_records = run_full_benchmark(
            epoch_tasks,
            state=state,
            max_steps=max_steps,
            confidence_threshold=confidence_threshold,
            epoch=epoch,
            run_index_offset=cumulative_index,
            reliability_weight=reliability_weight,
            enable_skip=enable_skip,
            plan_builder=build_chunk9_activation_plan,
            on_record=lambda rec: _append_record(rec, RESULTS_AGENSFLOW),
            enable_router_logging=enable_router_logging,
            router_log_dir=(RESULTS_DIR / "router_logs") if enable_router_logging else None,
            governance_policy=governance_policy,
            report_dir=REPORT_DIR,
        )
        cumulative_index += len(epoch_records)

        # Per-epoch checkpoint (skipped under --frozen so the input
        # graph remains untouched).
        if not args.frozen:
            snapshot_path = SNAPSHOTS_DIR / f"policy_graph_epoch_{epoch:02d}.pkl"
            save_policy_graph(graph, snapshot_path)
            save_policy_graph(graph, GRAPH_PATH)
            print(f"\nEpoch {epoch} complete. Graph: {len(graph)} nodes.")
        else:
            print(f"\nEpoch {epoch} complete (frozen replay — graph not saved).")

    # Final aggregation. Re-load all records (including any prior --resume).
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

    # ----- Session-level aggregation across per-task RunReports ----- #
    if REPORT_DIR.exists():
        report_files = sorted(REPORT_DIR.glob("run_report_*.json"))
        runs: list[RunReport] = []
        for p in report_files:
            try:
                d = json.loads(p.read_text())
                # Reconstruct RunReport (preserve enough for aggregation;
                # GovernanceViolation rebuild not strictly needed since
                # SessionReport.format_human only reads aggregates).
                from agensflow.runtime.report import AgentActivitySummary
                d["agents"] = [
                    AgentActivitySummary(**a) for a in d.get("agents", [])
                ]
                d.pop("governance_violations", None)
                runs.append(RunReport(governance_violations=[], **d))
            except Exception as exc:  # noqa: BLE001
                # Tolerate unparseable reports — don't let one bad file
                # block the session-level summary.
                print(f"  (skipping malformed report {p.name}: {exc})")
        if runs:
            session = SessionReport(
                runs=runs, label="chunk-9 sustained traffic + governance",
            )
            session_path = RESULTS_DIR / "session_report.json"
            session_path.write_text(
                json.dumps(session.to_dict(), indent=2, default=str)
            )
            print()
            print(session.format_human())
            print()
            print(f"Session report saved to {session_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
