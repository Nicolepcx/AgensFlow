"""
Declarative skill definitions (SKILL.md cards).

This package adds a *behavioral* layer alongside the existing structural
SkillSpec. The structural spec (preconditions, outputs, role) lives in
`agensflow.schema.SkillSpec`. The behavioral content (system prompt,
when-to-use guidance) lives here as `SkillCard`, parsed from a SKILL.md
file with YAML frontmatter + markdown body.

The two compose: a registered skill has a SkillSpec (always) and may
also have a SkillCard (when behavior is specified declaratively rather
than hardcoded in `runtime/agents.py`). When both are present, the
agent factory uses the SkillCard's instructions as the system prompt.
When only the SkillSpec is present, the factory falls back to the
hardcoded prompt (backward compat with chunks 6/7/8).

Format follows Anthropic's Claude Code skills convention:

    ---
    name: solver_concise
    description: Minimum-viable answer; single-paragraph responses.
    role: solver
    model_hint: anthropic/claude-haiku-4.5  # optional
    ---

    # Solver — Concise

    You answer the user's subproblem directly, in the fewest words that
    still convey a complete answer. ...

The framework loads SKILL.md files from a configurable directory at
startup; users can drop new files in to add new behavioral variants
without touching code.
"""

from agensflow.skills.card import SkillCard
from agensflow.skills.loader import (
    load_skill_card,
    load_skills_from_directory,
    parse_skill_md,
)

__all__ = [
    "SkillCard",
    "load_skill_card",
    "load_skills_from_directory",
    "parse_skill_md",
]
