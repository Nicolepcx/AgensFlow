"""
Activation plan for e09 cross-domain security validation.

This file is intentionally identical to `e07_skill_variants/activation.py`:
the cross-domain claim under test is that the SAME variant pool +
SAME branch-rule discipline learns useful routing on a different
corpus. Diverging from e07 here would conflate "the substrate transfers"
with "we changed the action space" — keep them locked together.

Key carry-overs from e07 (do not change without a reason):

- 9 (skill_card × model) solver cells — the chunk-9 cross product.
- `branch_rule = BranchRule(enabled=False, max_branches=1)` —
  unconditionally disables branching regardless of the regime label.
  This is what defuses the C5 / ambiguous-regime landmine: even when
  `detect_regime(C5_features)` returns `ambiguous`, no branching path
  is exercised, so no `NotImplementedError` from the unfinished
  branching runtime.
- `merge_strategy = "verifier_gate"`, identical evaluation criteria.

Re-exports `CHUNK9_SELECTED_SKILLS` and `build_chunk9_activation_plan`
so e09's run.py mirrors e07's import shape exactly.
"""

from __future__ import annotations

from agensflow import (
    ActivationPlan,
    BranchRule,
    RegimeEstimate,
    TaskFeatures,
    detect_regime,
)


# Identical to e07's CHUNK9_SELECTED_SKILLS.
CHUNK9_SELECTED_SKILLS: list[str] = [
    "planner",
    # Retrieval alternatives — corpus + 2 web search providers.
    "memory",
    "web_search_exa",
    "web_search_tavily",
    # Solver action space: 3 cards × 3 models = 9 cells.
    "solver_concise_haiku", "solver_concise_fast", "solver_concise_mini",
    "solver_cot_haiku",     "solver_cot_fast",     "solver_cot_mini",
    "solver_evidence_haiku", "solver_evidence_fast", "solver_evidence_mini",
    # Verifier variants.
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
    """Build the chunk-9 activation plan for the e09 cross-domain run.

    Identical body to `e07_skill_variants/activation.py`'s function of
    the same name. Kept here so e09 is self-contained and can be run
    without e07 imports — important because e09 is, in principle,
    a separable validation experiment.
    """
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
