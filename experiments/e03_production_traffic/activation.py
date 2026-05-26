"""
Chunk 6 activation plan — exposes the full variant pool as legal options.

The default rule-based activation plan for `evidence_heavy` regime selects
a fixed coalition (`planner → memory → solver → verifier → evaluator`).
For chunk 6 we want the policy graph to *learn* the routing across:

  - 5 solver model variants (cross-family pool)
  - 2 verifier variants (fast / haiku)
  - 3 retrieval options (corpus memory + 2 web search providers)

So we override the activation plan to make all of them legal at the planner's
post-state. The router then consults the policy graph at each step and picks
the UCB-best legal action — the framework's actual coordination-learning
mechanism in action.

The original rule-based regime detector still runs (we use its label to seed
the policy graph signatures), but its *plan* is replaced by this richer one.
"""

from __future__ import annotations

from agensflow import (
    ActivationPlan,
    BranchRule,
    RegimeEstimate,
    TaskFeatures,
    detect_regime,
)


# All skills available to the policy in chunk 6.
#
# NOTE on the Qwen variants: `solver_qwen_flash` and `solver_qwen_max` are
# registered in the registry and the model bindings, but excluded from the
# chunk-6 activation plan because the Qwen endpoints OpenRouter routes to
# don't support the `tool_choice` parameter that Instructor's TOOLS mode
# requires (returns 404). Adding them back requires per-variant Instructor
# mode configuration (e.g. Mode.OPENROUTER_STRUCTURED_OUTPUTS for Qwen,
# Mode.TOOLS for OpenAI/Anthropic) — that's chunk 7 work. Chunk 6 still
# tests cross-family routing across the three OpenAI tiers and Anthropic.
CHUNK6_SELECTED_SKILLS: list[str] = [
    "planner",
    # Retrieval alternatives — corpus + 2 web search providers.
    "memory",
    "web_search_exa",
    "web_search_tavily",
    # Solver variants — 3 OpenAI tiers + 1 Anthropic.
    "solver_fast",
    "solver_mini",
    "solver_haiku",
    # Verifier variants.
    "verifier_fast",
    "verifier_haiku",
    # Termination.
    "evaluator",
]


def build_chunk6_activation_plan(
    features: TaskFeatures,
    *,
    regime: RegimeEstimate | None = None,
) -> ActivationPlan:
    """
    Build the chunk-6 activation plan.

    Uses the rule-based regime detector for the regime label (so signatures
    are still meaningfully classified) but overrides the coalition with the
    full variant pool, leaving the actual routing to the policy graph.
    """
    estimate = regime if regime is not None else detect_regime(features)
    return ActivationPlan(
        regime=estimate,
        selected_skills=list(CHUNK6_SELECTED_SKILLS),
        branch_rule=BranchRule(enabled=False, max_branches=1),
        merge_strategy="verifier_gate",
        evaluation_criteria=[
            "evidence_coverage",
            "verification_strength",
            "coherence",
        ],
    )
