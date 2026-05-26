"""
ModelsConfig — typed configuration for skill→model bindings.

The default tables (`DEFAULT_MODEL_ASSIGNMENT`, `SKILL_VARIANT_BINDINGS`,
`SKILL_VARIANT_CARDS`) live in `core.py` and remain importable as
module-level constants for backward compat. This config dataclass mirrors
them as YAML-overridable knobs so users can:

  - swap their own provider/model identifiers per skill without forking
  - add experiment-specific variants without editing library code
  - bind variants to skill-cards via YAML

The runtime resolution functions (`get_model_for_skill`,
`get_card_for_skill`, etc.) operate against the merged tables — by
default they read the in-code constants; when a `ModelsConfig` is in
play, they read the merged result.

See `README.md` for per-knob explanation and the variant-pattern
rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_model_assignment() -> dict[str, str]:
    """Default base-skill → model mapping (matches in-code constants)."""
    return {
        "planner": "openai/gpt-5.4-nano",
        "memory": "openai/gpt-5.4-nano",
        "solver": "anthropic/claude-haiku-4.5",
        "critic": "anthropic/claude-haiku-4.5",
        "verifier": "anthropic/claude-haiku-4.5",
        "evaluator": "openai/gpt-5.4-nano",
        "synthesizer": "anthropic/claude-haiku-4.5",
    }


def _default_variant_bindings() -> dict[str, list[str]]:
    """Default variant_name → [base_skill, model_id] mapping.

    NOTE: stored as `list[str]` of length 2 instead of `tuple[str, str]`
    because OmegaConf's structured-config schema doesn't handle
    fixed-length tuples cleanly. The runtime accessor adapts to a
    tuple at read time.
    """
    return {
        # Solver variants — same role, different (family, tier) cells.
        "solver_fast": ["solver", "openai/gpt-5.4-nano"],
        "solver_mini": ["solver", "openai/gpt-5.4-mini"],
        "solver_haiku": ["solver", "anthropic/claude-haiku-4.5"],
        "solver_qwen_flash": ["solver", "qwen/qwen3.6-flash"],
        "solver_qwen_max": ["solver", "qwen/qwen3.6-max-preview"],
        # Verifier variants — verification quality vs. cost.
        "verifier_fast": ["verifier", "openai/gpt-5.4-nano"],
        "verifier_haiku": ["verifier", "anthropic/claude-haiku-4.5"],
        # Chunk-9: solver-card × model variants.
        "solver_concise_haiku":  ["solver", "anthropic/claude-haiku-4.5"],
        "solver_concise_fast":   ["solver", "openai/gpt-5.4-nano"],
        "solver_concise_mini":   ["solver", "openai/gpt-5.4-mini"],
        "solver_cot_haiku":      ["solver", "anthropic/claude-haiku-4.5"],
        "solver_cot_fast":       ["solver", "openai/gpt-5.4-nano"],
        "solver_cot_mini":       ["solver", "openai/gpt-5.4-mini"],
        "solver_evidence_haiku": ["solver", "anthropic/claude-haiku-4.5"],
        "solver_evidence_fast":  ["solver", "openai/gpt-5.4-nano"],
        "solver_evidence_mini":  ["solver", "openai/gpt-5.4-mini"],
    }


def _default_variant_cards() -> dict[str, str]:
    """Default variant_name → skill_card_name mapping."""
    return {
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


@dataclass
class ModelsConfig:
    """Configuration for skill→model bindings.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, never mutate.

    YAML override semantics: OmegaConf's dict-merge ADDS keys + REPLACES
    values for matching keys. So a YAML like
        models:
          variant_bindings:
            my_solver: ["solver", "openai/gpt-5.4-nano"]
    ADDS `my_solver` to the variant table. To REMOVE a default variant,
    build the dict explicitly in your YAML — OmegaConf doesn't have a
    "delete this key" operator.
    """

    # base skill → model id
    default_assignment: dict[str, str] = field(
        default_factory=_default_model_assignment
    )

    # variant name → [base_skill, model_id]
    # (list-of-2-strings instead of tuple — OmegaConf limitation)
    variant_bindings: dict[str, list[str]] = field(
        default_factory=_default_variant_bindings
    )

    # variant name → skill_card name
    variant_cards: dict[str, str] = field(
        default_factory=_default_variant_cards
    )
