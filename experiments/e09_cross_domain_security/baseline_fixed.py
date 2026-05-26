"""
Fixed-full-pipeline baseline for e09 cross-domain validation.

Runs each of the 60 tasks ONCE through a hardcoded topology:

    planner → memory → web_search_exa → web_search_tavily
            → solver_cot_haiku → verifier_haiku → evaluator

The activation plan exposes ONLY these 7 cells (no variant pool, no
skip-X, no learning). This is what an engineer would put together
without the substrate — the comparator for prediction #4:

  > Main run achieves ≥10% cost reduction at RULER within −0.03 of
  > the fixed-full-pipeline baseline's mean RULER.

Single epoch, single pass, no policy-graph persistence. The e03 harness
is reused as-is; we just hand it a different activation plan and an
empty graph that we don't save.

Usage:

    python -m experiments.e09_cross_domain_security.baseline_fixed \\
        --config experiments/e09_cross_domain_security/example_config.yaml

Output: `baseline_fixed/results_baseline.jsonl` plus a small summary
to stdout. The main run's RESULTS.md analysis script can load this
file and compute the prediction-#4 comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from agensflow import (
    ActivationPlan,
    BranchRule,
    DEFAULT_RUBRIC,
    PolicyGraph,
    RegimeEstimate,
    TaskFeatures,
    detect_regime,
)
from agensflow.config import load_config
from agensflow.registry import default_registry
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.governance import GovernancePolicy
from agensflow.runtime.preflight import run_preflight_checks

from experiments.e03_production_traffic.harness import (
    DEFAULT_JUDGE_MODEL,
    HarnessState,
    TrajectoryRecord,
    record_to_dict,
    run_full_benchmark,
)
from experiments.e09_cross_domain_security.tasks import ALL_TASKS


THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

OUTPUT_DIR = THIS_DIR / "baseline_fixed"
RESULTS_PATH = OUTPUT_DIR / "results_baseline.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# --------------------------------------------------------------------------- #
# Fixed activation plan — single canonical solver, full pipeline, no skip.
# --------------------------------------------------------------------------- #

BASELINE_SELECTED_SKILLS: list[str] = [
    "planner",
    "memory",
    "web_search_exa",
    "web_search_tavily",
    "solver_cot_haiku",   # canonical: chain-of-thought + haiku (closest to chunk-7/8's hardcoded solver)
    "verifier_haiku",
    "evaluator",
]


def build_baseline_activation_plan(
    features: TaskFeatures,
    *,
    regime: RegimeEstimate | None = None,
) -> ActivationPlan:
    """Activation plan exposing only the fixed-full-pipeline cells.

    Same branch_rule=disabled discipline as e07/e09 to defuse the
    ambiguous-regime branching landmine. Same merge_strategy + eval
    criteria so the harness behaves identically aside from the action
    surface.
    """
    estimate = regime if regime is not None else detect_regime(features)
    return ActivationPlan(
        regime=estimate,
        selected_skills=list(BASELINE_SELECTED_SKILLS),
        branch_rule=BranchRule(enabled=False, max_branches=1),
        merge_strategy="verifier_gate",
        evaluation_criteria=[
            "evidence_coverage",
            "verification_strength",
            "coherence",
        ],
    )


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def _append_record(record: TrajectoryRecord, path: Path) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record_to_dict(record), default=str) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-full-pipeline baseline for e09 cross-domain validation. "
            "Single canonical solver, no skip, no learning. Provides the "
            "comparator for prediction #4 (≥10% cost reduction at RULER "
            "within -0.03 of this baseline's mean RULER)."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (defaults to example_config.yaml).",
    )
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task ids (default: all 60).",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    # ----- Load config ----- #
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        cfg = load_config()

    judge_model = (
        args.judge_model if args.judge_model is not None
        else (cfg.relative_judge.judge_model or DEFAULT_JUDGE_MODEL)
    )
    max_steps = (
        args.max_steps if args.max_steps is not None
        else max(cfg.router.max_steps, 18)
    )
    confidence_threshold = cfg.policy_graph.confidence_threshold
    reliability_weight = cfg.policy_graph.reliability_weight

    # Register SKILL.md cards so solver_cot_haiku resolves.
    n_cards = default_registry.register_cards_from_directory(
        SKILLS_DIR, overwrite=True
    )
    print(f"Loaded {n_cards} SKILL.md cards from {SKILLS_DIR}")
    print()

    # Pick tasks.
    tasks = list(ALL_TASKS)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"No tasks match: {sorted(wanted)}", file=sys.stderr)
            return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print(f"Running fixed-full-pipeline baseline.")
    print(f"  tasks:                 {len(tasks)}")
    print(f"  pipeline cells:        {BASELINE_SELECTED_SKILLS}")
    print(f"  judge model:           {judge_model}")
    print(f"  max_steps:             {max_steps}")
    print(f"  enable_skip:           False (forced)")
    print(f"  output dir:            {OUTPUT_DIR}")
    print()

    load_dotenv()

    # Pre-flight (cheap; aborts if external deps misconfigured).
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
    reward_config = cfg.reward
    governance_policy = cfg.governance
    rubric = cfg.relative_judge.rubric or DEFAULT_RUBRIC

    # Fresh empty graph. We pass it to the harness but never save it —
    # "no learning" means the substrate's UCB explores uniformly at each
    # step, but since the activation plan only exposes one solver +
    # one verifier, the topology is effectively fixed.
    graph = PolicyGraph()

    state = HarnessState(
        policy_graph=graph,
        client=client,
        reward_config=reward_config,
        judge_model=judge_model,
        rubric=rubric,
        ruler_config=cfg.ruler,
        backup_gamma=cfg.policy_graph.gamma,
    )

    print(f"\n========== BASELINE PASS (1 epoch, no learning) ==========")
    records = run_full_benchmark(
        tasks,
        state=state,
        max_steps=max_steps,
        confidence_threshold=confidence_threshold,
        epoch=1,
        run_index_offset=0,
        reliability_weight=reliability_weight,
        enable_skip=False,  # forced: no skip-X
        plan_builder=build_baseline_activation_plan,
        on_record=lambda rec: _append_record(rec, RESULTS_PATH),
        enable_router_logging=False,
        router_log_dir=None,
        governance_policy=governance_policy,
        report_dir=OUTPUT_DIR / "run_reports",
    )

    # ----- Summary stats ----- #
    ok = [r for r in records if r.error is None]
    summary: dict[str, Any] = {
        "n_runs": len(records),
        "n_ok": len(ok),
        "n_errors": len(records) - len(ok),
        "tokens_mean": int(mean(r.total_tokens for r in ok)) if ok else 0,
        "ruler_mean": (
            mean(r.ruler_score for r in ok if r.ruler_score is not None)
            if ok else 0.0
        ),
        "reward_mean": (
            mean(r.hybrid_reward for r in ok if r.hybrid_reward is not None)
            if ok else 0.0
        ),
        "retries_mean": mean(r.validation_retries for r in ok) if ok else 0.0,
        "pipeline": BASELINE_SELECTED_SKILLS,
        "per_class": {},
    }

    # Per-class breakdown (for the analysis script that compares this
    # to the main run's per-class results).
    by_class: dict[str, list[TrajectoryRecord]] = {}
    for r in ok:
        by_class.setdefault(r.scenario_class, []).append(r)
    for c in sorted(by_class.keys()):
        grp = by_class[c]
        rulers = [r.ruler_score for r in grp if r.ruler_score is not None]
        summary["per_class"][c] = {
            "n": len(grp),
            "tokens_mean": int(mean(r.total_tokens for r in grp)),
            "ruler_mean": mean(rulers) if rulers else 0.0,
        }

    if args.no_write:
        print(json.dumps(summary, indent=2, default=str))
    else:
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
        print()
        print(f"Baseline summary:")
        print(f"  runs:          {summary['n_ok']}/{summary['n_runs']} ok "
              f"({summary['n_errors']} errors)")
        print(f"  tokens mean:   {summary['tokens_mean']:,}")
        print(f"  RULER mean:    {summary['ruler_mean']:.3f}")
        print(f"  reward mean:   {summary['reward_mean']:+.3f}")
        print(f"  retries mean:  {summary['retries_mean']:.2f}")
        print()
        print(f"  per-class mean tokens / RULER:")
        for c, s in summary["per_class"].items():
            print(f"    {c} (n={s['n']}): {s['tokens_mean']:,} tok, "
                  f"RULER={s['ruler_mean']:.2f}")
        print()
        print(f"Records: {RESULTS_PATH}")
        print(f"Summary: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
