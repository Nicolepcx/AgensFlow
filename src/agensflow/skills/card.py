"""
SkillCard — the behavioral specification for a skill.

A SkillCard is what gets parsed out of a SKILL.md file: structured
metadata (frontmatter) plus the system-prompt body (markdown). Together
with the existing `SkillSpec` (structural — preconditions, outputs,
role) it defines a complete skill.

Why two objects:

  - SkillSpec answers "what does this skill *consume and produce*?"
    (preconditions, outputs, handoff requirements, regime affinity).
    Used by the activation planner and router to decide *whether* a
    skill is legal at a given state.

  - SkillCard answers "*how* does this skill behave?" (system prompt,
    when-to-use guidance, optional model hint). Used by the agent
    factory at construction time to set the LLM's system prompt and
    optional preferred model.

A skill registered in the registry has a SkillSpec (always) and may
have a SkillCard. When both are present, the agent factory uses the
card's instructions; when only the spec is present, it falls back to
the hardcoded prompt in `runtime/agents.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillCard:
    """Behavioral specification for a skill, parsed from a SKILL.md file.

    Fields map directly to the YAML frontmatter (metadata) plus the
    markdown body (instructions). The `role` field, when present, lets
    the runtime group cards by role-category (e.g. "solver") so a single
    card can underpin multiple model-binding variants — chunk-9's
    pattern is one solver card paired with three model bindings.

    Attributes:
        name:           Unique identifier matching what the registry uses.
                        For chunk-9 cards this is the *card name*
                        (e.g. "solver_concise"), not a per-binding variant
                        name (e.g. "solver_concise_haiku"). The variant
                        binding maps cards to model IDs separately.
        description:    One-line summary of when this card is appropriate.
                        Lifted from frontmatter; used in routing-tooltip
                        UI and as a hint to the policy graph viz.
        instructions:   The full system-prompt body. The agent factory
                        passes this to the underlying LLM as the
                        system_prompt at construction time.
        role:           Optional role grouping (e.g. "solver", "verifier").
                        When set, the runtime knows this card can be paired
                        with model bindings of that role category.
        model_hint:     Optional preferred model id (OpenRouter slug).
                        The runtime treats this as a default that can be
                        overridden by per-call model_overrides or by the
                        SKILL_VARIANT_BINDINGS table.
        tools:          Optional list of tool names this card requires.
                        Reserved for future skill definitions that bind
                        to tools (web search, code exec, etc.). For now
                        cards that need tools list them as identifiers
                        the runtime resolves at agent-factory time.
        license:        Optional license declaration from the SKILL.md
                        frontmatter (e.g. for distributable skill cards).
        source_path:    Where the card was loaded from. Useful for
                        diagnostics and for displaying the card source
                        in the viz.
    """

    name: str
    description: str
    instructions: str
    role: str | None = None
    model_hint: str | None = None
    tools: list[str] = field(default_factory=list)
    license: str | None = None
    source_path: str | None = None
