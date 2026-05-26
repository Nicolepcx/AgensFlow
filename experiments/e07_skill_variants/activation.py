"""
Chunk 9 activation plan — exposes (skill card × model) solver variants.

Replaces chunk-6's three model-binding-only solver actions
(`solver_fast`, `solver_mini`, `solver_haiku`) with the chunk-9 cross
product of three skill cards × three model bindings (9 solver actions
total). The planner / memory / web search / verifier / evaluator
sections of the action space are unchanged from chunk 6.

The policy graph treats each `(skill_card, model)` pair as a distinct
action with its own value/visit/variance/failure-rate state, exactly the
same way it treated `solver_haiku` vs `solver_fast` in chunks 7/8.
The richer action space is what makes the chunk-9 systems claim
("(skill, model, signature, domain) reliability surface is observable
and learnable") empirically testable.
"""

from __future__ import annotations

from agensflow import (
    ActivationPlan,
    BranchRule,
    RegimeEstimate,
    TaskFeatures,
    detect_regime,
)


# Chunk-9 skill list. Solver position now exposes 9 (card × model) cells
# instead of chunk-6's 3 model-only variants. Other skill positions
# inherit chunk-6's choices.
CHUNK9_SELECTED_SKILLS: list[str] = [
    "planner",
    # Retrieval alternatives — corpus + 2 web search providers.
    "memory",
    "web_search_exa",
    "web_search_tavily",
    # Chunk-9 solver action space: 3 cards × 3 models = 9 cells.
    # The card defines the *behavior* (concise / chain-of-thought /
    # evidence-first); the model defines the *substrate* (haiku / fast /
    # mini). The framework learns the (card × model × signature) winners.
    "solver_concise_haiku", "solver_concise_fast", "solver_concise_mini",
    "solver_cot_haiku",     "solver_cot_fast",     "solver_cot_mini",
    "solver_evidence_haiku", "solver_evidence_fast", "solver_evidence_mini",
    # Verifier variants (unchanged).
    "verifier_fast",
    "verifier_haiku",
    # Termination.
    "evaluator",
]


def build_chunk9_activation_plan(
    features: TaskFeatures,
    *,
    regime: RegimeEstimate | None = None,
) -> ActivationPlan:
    """Build the chunk-9 activation plan with the (skill × model) solver
    cross product."""
    estimate = regime if regime is not None else detect_regime(features)
    return ActivationPlan(
        regime=estimate,
        selected_skills=list(CHUNK9_SELECTED_SKILLS),
        branch_rule=BranchRule(enabled=False, max_branches=1),
        merge_strategy="verifier_gate",
        evaluation_criteria=[
            "evidence_coverage",
            "verification_strength",
            "coherence",
        ],
    )
