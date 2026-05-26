"""
Model assignments and skill-variant bindings.

Two layers of indirection:

  1. DEFAULT_MODEL_ASSIGNMENT — maps each *base* skill (planner, memory,
     solver, critic, verifier, evaluator, synthesizer) to a default model.
     Used when the activation plan invokes the base skill name directly.

  2. SKILL_VARIANT_BINDINGS — maps *variant* skill names (e.g.
     `solver_fast`, `solver_capable`, `solver_qwen36_flash`) to a
     (base_skill, model_id) tuple. Variants exist so the policy graph can
     learn per-signature which model fits which query class. The registry
     entry for each variant has the *base skill's* preconditions, outputs,
     and handoff requirements — only the model differs.

The variant pattern is what enables AgensFlow's online model-routing
capability. Drop a new variant in the registry and the bindings map; the
policy explores it on a fraction of traffic, the reward signal judges it,
and the policy converges to using it on signatures where it wins.
"""

from __future__ import annotations

# Default model assignment per *base* skill.
DEFAULT_MODEL_ASSIGNMENT: dict[str, str] = {
    "planner": "openai/gpt-5.4-nano",
    "memory": "openai/gpt-5.4-nano",
    "solver": "anthropic/claude-haiku-4.5",
    "critic": "anthropic/claude-haiku-4.5",
    "verifier": "anthropic/claude-haiku-4.5",
    "evaluator": "openai/gpt-5.4-nano",
    "synthesizer": "anthropic/claude-haiku-4.5",
}


# Skill-variant bindings: variant_name -> (base_skill, model_id).
# Variants share the underlying agent factory of the base skill but route to
# a specific model. The policy graph learns per-signature which variant wins.
#
# The solver pool spans three families (OpenAI, Anthropic, Qwen) across the
# cost-capability spectrum. This lets the policy learn cross-family routing —
# e.g. "for this signature class, Qwen Flash matches Haiku at lower cost" or
# "this signature class needs Qwen Max's frontier reasoning." That cross-
# family decision space is what makes the framework's online-model-routing
# claim a real thing rather than a single-vendor optimization.
SKILL_VARIANT_BINDINGS: dict[str, tuple[str, str]] = {
    # Solver variants — same role, different (family, tier) cells.
    "solver_fast": ("solver", "openai/gpt-5.4-nano"),
    "solver_mini": ("solver", "openai/gpt-5.4-mini"),
    "solver_haiku": ("solver", "anthropic/claude-haiku-4.5"),
    "solver_qwen_flash": ("solver", "qwen/qwen3.6-flash"),
    "solver_qwen_max": ("solver", "qwen/qwen3.6-max-preview"),
    # Verifier variants — verification quality vs. cost.
    "verifier_fast": ("verifier", "openai/gpt-5.4-nano"),
    "verifier_haiku": ("verifier", "anthropic/claude-haiku-4.5"),
    # Chunk-9: solver-card × model variants — same skill *card* paired
    # with three model bindings each. The `(card, model)` action space
    # lets the policy graph learn per-signature which behavioral
    # specification × which model is the cost/quality/reliability winner.
    "solver_concise_haiku": ("solver", "anthropic/claude-haiku-4.5"),
    "solver_concise_fast":  ("solver", "openai/gpt-5.4-nano"),
    "solver_concise_mini":  ("solver", "openai/gpt-5.4-mini"),
    "solver_cot_haiku":     ("solver", "anthropic/claude-haiku-4.5"),
    "solver_cot_fast":      ("solver", "openai/gpt-5.4-nano"),
    "solver_cot_mini":      ("solver", "openai/gpt-5.4-mini"),
    "solver_evidence_haiku": ("solver", "anthropic/claude-haiku-4.5"),
    "solver_evidence_fast":  ("solver", "openai/gpt-5.4-nano"),
    "solver_evidence_mini":  ("solver", "openai/gpt-5.4-mini"),
}


# Maps a variant skill name to its skill-card name (parsed from a
# SKILL.md file). Variants without an entry here fall back to the
# hardcoded prompt in `runtime/agents.py` (backward compat for chunks
# 6/7/8). Chunk-9's solver card variants live here.
SKILL_VARIANT_CARDS: dict[str, str] = {
    "solver_concise_haiku":   "solver_concise",
    "solver_concise_fast":    "solver_concise",
    "solver_concise_mini":    "solver_concise",
    "solver_cot_haiku":       "solver_chain_of_thought",
    "solver_cot_fast":        "solver_chain_of_thought",
    "solver_cot_mini":        "solver_chain_of_thought",
    "solver_evidence_haiku":  "solver_evidence_first",
    "solver_evidence_fast":   "solver_evidence_first",
    "solver_evidence_mini":   "solver_evidence_first",
}


def get_card_for_skill(skill: str) -> str | None:
    """Return the skill-card name a variant binds to, or None if the
    variant has no card binding (falls back to hardcoded prompt)."""
    return SKILL_VARIANT_CARDS.get(skill)


def get_model_for_skill(
    skill: str,
    override: dict[str, str] | None = None,
) -> str:
    """
    Resolve the model identifier to use for a given skill.

    Lookup order: per-call override → SKILL_VARIANT_BINDINGS →
    DEFAULT_MODEL_ASSIGNMENT.
    """
    if override and skill in override:
        return override[skill]
    if skill in SKILL_VARIANT_BINDINGS:
        return SKILL_VARIANT_BINDINGS[skill][1]
    if skill in DEFAULT_MODEL_ASSIGNMENT:
        return DEFAULT_MODEL_ASSIGNMENT[skill]
    raise KeyError(
        f"No model assignment for skill {skill!r}. "
        f"Either add it to DEFAULT_MODEL_ASSIGNMENT or pass an override."
    )


def get_base_skill(skill: str) -> str:
    """
    Return the base skill name for a variant, or the skill itself if it's a
    base skill. Used by the runtime to look up the right agent factory.
    """
    if skill in SKILL_VARIANT_BINDINGS:
        return SKILL_VARIANT_BINDINGS[skill][0]
    return skill


def is_variant(skill: str) -> bool:
    """True if `skill` is a variant binding (vs. a base skill)."""
    return skill in SKILL_VARIANT_BINDINGS
