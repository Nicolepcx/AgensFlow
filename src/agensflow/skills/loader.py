"""
SKILL.md parser and directory loader.

Parses files in the Anthropic Claude Code skills format:

    ---
    name: solver_concise
    description: ...
    role: solver
    ---

    # Body
    Markdown system-prompt instructions...

The frontmatter is YAML; the body is everything after the closing `---`.
Both are required.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agensflow.skills.card import SkillCard


_FRONTMATTER_DELIM = "---"


def parse_skill_md(text: str, *, source_path: str | None = None) -> SkillCard:
    """
    Parse a SKILL.md document into a SkillCard.

    The document must start with a YAML frontmatter block delimited by
    `---` lines, followed by markdown body content. `name`, `description`
    are required frontmatter fields. `role`, `model_hint`, `tools`, and
    `license` are optional.

    Raises ValueError if the document is malformed or required fields
    are missing — fail loud rather than silently registering a broken
    card.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        raise ValueError(
            f"SKILL.md missing opening frontmatter delimiter '---'"
            f"{f' at {source_path}' if source_path else ''}."
        )

    # Find the closing ---.
    closing_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            closing_idx = i
            break
    if closing_idx is None:
        raise ValueError(
            f"SKILL.md missing closing frontmatter delimiter '---'"
            f"{f' at {source_path}' if source_path else ''}."
        )

    frontmatter_text = "\n".join(lines[1:closing_idx])
    body_text = "\n".join(lines[closing_idx + 1 :]).strip()

    try:
        meta = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"SKILL.md frontmatter is not valid YAML"
            f"{f' at {source_path}' if source_path else ''}: {exc}"
        ) from exc

    if not isinstance(meta, dict):
        raise ValueError(
            f"SKILL.md frontmatter must be a YAML mapping, got "
            f"{type(meta).__name__}{f' at {source_path}' if source_path else ''}."
        )

    name = meta.get("name")
    description = meta.get("description")
    if not name or not isinstance(name, str):
        raise ValueError(
            f"SKILL.md missing required `name` field (string)"
            f"{f' at {source_path}' if source_path else ''}."
        )
    if not description or not isinstance(description, str):
        raise ValueError(
            f"SKILL.md missing required `description` field (string)"
            f"{f' at {source_path}' if source_path else ''}."
        )
    if not body_text:
        raise ValueError(
            f"SKILL.md has empty body — instructions are required"
            f"{f' at {source_path}' if source_path else ''}."
        )

    tools_raw = meta.get("tools") or []
    if isinstance(tools_raw, str):
        tools = [tools_raw]
    elif isinstance(tools_raw, list):
        tools = [str(t) for t in tools_raw]
    else:
        raise ValueError(
            f"SKILL.md `tools` must be a list or a string"
            f"{f' at {source_path}' if source_path else ''}."
        )

    return SkillCard(
        name=name,
        description=description,
        instructions=body_text,
        role=meta.get("role"),
        model_hint=meta.get("model_hint"),
        tools=tools,
        license=meta.get("license"),
        source_path=source_path,
    )


def load_skill_card(path: str | Path) -> SkillCard:
    """Load a single SKILL.md file from disk and parse it."""
    p = Path(path)
    text = p.read_text()
    return parse_skill_md(text, source_path=str(p))


def load_skills_from_directory(
    directory: str | Path,
    *,
    pattern: str = "*.md",
    recursive: bool = False,
) -> dict[str, SkillCard]:
    """
    Load every SKILL.md file in a directory into a `{name: SkillCard}` dict.

    Naming collisions raise — duplicate skill names indicate a setup bug
    (two SKILL.md files declaring the same `name` in their frontmatter).
    The framework would otherwise silently route on whichever was loaded
    last, which is a footgun.

    Args:
        directory: path to the skills directory (e.g. `skills/` at the
                   repo root, or wherever the user has organized theirs).
        pattern:   glob pattern for files to load. Defaults to `*.md`.
        recursive: when True, walks subdirectories. Default False keeps
                   the loader honest about what's being registered.
    """
    d = Path(directory)
    if not d.exists():
        return {}
    if not d.is_dir():
        raise NotADirectoryError(
            f"Skills directory {d} exists but is not a directory."
        )

    iterator = d.rglob(pattern) if recursive else d.glob(pattern)
    out: dict[str, SkillCard] = {}
    for p in sorted(iterator):
        if not p.is_file():
            continue
        card = load_skill_card(p)
        if card.name in out:
            existing = out[card.name].source_path
            raise ValueError(
                f"Duplicate skill name {card.name!r}: declared in "
                f"both {existing} and {p}. Skill names must be unique "
                f"across the directory; rename one or remove the duplicate."
            )
        out[card.name] = card
    return out
