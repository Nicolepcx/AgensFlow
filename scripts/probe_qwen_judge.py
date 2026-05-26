"""
Cross-provider judge compatibility probe.

Tests whether each candidate judge model can produce a valid
`_StrictRelativeJudgement` output through Instructor's TOOLS mode AND
JSON mode on OpenRouter.

Three failure layers from chunk-11 investigation:
  1. OpenRouter routing 404 (qwen-all, gemini-flash on TOOLS) —
     "no endpoints support tool_choice value." JSON mode bypasses
     tool_choice entirely; may unblock these.
  2. Model-side validation timeout (gemini-pro) — call returns
     malformed/null output. JSON mode has different output handling.
  3. Schema compliance refusal (grok-4.3) — model won't populate
     required fields even when schema demands. Mode-independent.

We probe each candidate under TOOLS first (fastest, primary mode),
then under JSON for those that fail TOOLS. If any model works under
JSON, that's our third family for chunk-11+ cross-judge.

Usage:

    python -m scripts.probe_qwen_judge

Cost: ~$0.05-0.10 per model × ~14 candidates × up to 2 modes.
Total: ~$1-2. Time: ~5 min.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from instructor import Mode

from agensflow.learning.relative_judge import TrajectoryToScore
from agensflow.learning.relative_judge.core import _StrictRelativeJudgement
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.trace import TraceCollector


# Candidates spanning the 4 known-issue families + 3 untested families.
CANDIDATES = [
    # Anthropic — control (known-good).
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    # OpenAI — control + bigger tier.
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
    # Qwen — all routing-failed under TOOLS in chunk-11; test JSON mode.
    "qwen/qwen3.6-flash",
    "qwen/qwen3.6-max-preview",
    "qwen/qwen3.6-235b-a22b",
    # Google Gemini — pro-preview validation-failed, flash routing-failed.
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash",
    # xAI — schema-compliance refused on TOOLS+strict.
    "x-ai/grok-4.3",
    # Untested families — chunk-11 doesn't have data on these:
    "mistralai/mistral-large-2511",
    "deepseek/deepseek-v3.5",
    "meta-llama/llama-3.4-405b-instruct",
    "cohere/command-a-2025-09",
]

# Minimal probe payload: 2 trajectories so the call doesn't short-circuit
# to the single-trajectory neutral path.
PROBE_PROMPT = """\
Score these two trajectories from 0.0 to 1.0 on these axes:
goal_achievement, grounding, coordination, recovery.

Output JSON with one entry per trajectory.

Trajectory 1:
  trajectory_id: t1
  final_answer: Paris is the capital of France.

Trajectory 2:
  trajectory_id: t2
  final_answer: I don't know.

The first trajectory is correct and grounded; the second is uncertain.
Score each trajectory's per-axis and a holistic score. axis_scores
MUST contain all four axis names with [0,1] floats.
"""


def _probe_one(model: str, mode: Mode) -> dict[str, Any]:
    """Probe a single (model, mode) combination using the documented
    Instructor+OpenRouter pattern: pass `extra_body` with
    `provider: {require_parameters: True}` so OpenRouter only routes
    to providers that fully support the requested parameters.
    Without this, the router can fall back to providers that don't
    support `tool_choice`, producing the 404 we hit in chunk-6 + chunk-11.
    """
    import instructor
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or ""
    raw = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                 timeout=60.0, max_retries=2)
    client_inst = instructor.from_openai(raw, mode=mode)

    started = time.monotonic()
    try:
        parsed, _ = client_inst.chat.completions.create_with_completion(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are an evaluator. Output STRICT JSON."},
                {"role": "user", "content": PROBE_PROMPT},
            ],
            response_model=_StrictRelativeJudgement,
            max_retries=2,
            temperature=0.0,
            max_tokens=600,
            # Instructor+OpenRouter integration pattern (per Instructor
            # docs): forces OpenRouter to route only to providers that
            # support all requested parameters. Critical for tool-mode
            # calls — without it, the router can pick a provider that
            # doesn't accept `tool_choice` and return the 404 we
            # mistakenly attributed to "qwen doesn't work" in chunk-6.
            extra_body={"provider": {"require_parameters": True}},
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
            "ok": False,
            "elapsed_s": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:300],
        }
    axes_populated = all(
        bool(s.axis_scores) and len(s.axis_scores) >= 1
        for s in parsed.scores
    )
    return {
        "model": model,
        "mode": mode.value if hasattr(mode, "value") else str(mode),
        "ok": True,
        "elapsed_s": time.monotonic() - started,
        "n_scores": len(parsed.scores),
        "axes_populated": axes_populated,
        "sample_axis_keys": (
            sorted(parsed.scores[0].axis_scores.keys())
            if parsed.scores else []
        ),
        "sample_score": parsed.scores[0].score if parsed.scores else None,
    }


def probe_with_fallback(model: str) -> list[dict[str, Any]]:
    """Probe a model under TOOLS first; if that fails, try JSON.
    Returns a list of result dicts (one per mode tried)."""
    results = []
    tools_result = _probe_one(model, Mode.TOOLS)
    results.append(tools_result)
    if not tools_result["ok"] or not tools_result.get("axes_populated"):
        # TOOLS failed or didn't populate axes — try JSON.
        json_result = _probe_one(model, Mode.JSON)
        results.append(json_result)
    return results


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set; cannot probe.", file=sys.stderr)
        return 1

    print(f"Probing {len(CANDIDATES)} candidate judges under TOOLS, "
          f"falling back to JSON mode if TOOLS fails.")
    print()

    all_results: list[dict[str, Any]] = []
    for model in CANDIDATES:
        for r in probe_with_fallback(model):
            all_results.append(r)
            mode_tag = r["mode"][:10].ljust(10)
            if r["ok"]:
                tag = "✓" if r.get("axes_populated") else "⚠ no-axes"
                print(f"  {model:<40s}  {mode_tag}  {tag}  "
                      f"{r['elapsed_s']:.1f}s  axes={r.get('axes_populated')}")
            else:
                print(f"  {model:<40s}  {mode_tag}  ✗ FAILED  "
                      f"{r['elapsed_s']:.1f}s  "
                      f"{r['error_type']}: {r['error_msg'][:60]}")

    # Summary: which (model, mode) combos work end-to-end with axes?
    print()
    print("=" * 80)
    print("Summary — validated working judges (axes populated)")
    print("=" * 80)
    working: list[tuple[str, str]] = []
    for r in all_results:
        if r["ok"] and r.get("axes_populated"):
            working.append((r["model"], r["mode"]))
    if working:
        # Group by family (first segment before '/').
        by_family: dict[str, list[tuple[str, str]]] = {}
        for model, mode in working:
            family = model.split("/")[0]
            by_family.setdefault(family, []).append((model, mode))
        for family in sorted(by_family.keys()):
            print(f"  {family}:")
            for model, mode in by_family[family]:
                print(f"     {model}  ({mode})")
        print()
        n_families = len(by_family)
        print(f"Distinct families with at least one working judge: {n_families}")
        if n_families >= 3:
            print(
                "✓ Cross-judge with ≥3 families is achievable — "
                "n=3 enables tie-breaking + outlier detection."
            )
        else:
            print(
                f"⚠ Only {n_families} families compliant — "
                "cross-judge bias mitigation is limited."
            )
    else:
        print("  (none)")

    print()
    print("Update `experiments/e06_cross_eval/run.py:DEFAULT_JUDGES` and "
          "`learning/ruler/README.md` with the new validated set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
