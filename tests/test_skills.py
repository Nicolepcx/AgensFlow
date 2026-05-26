"""
Tests for the SKILL.md parser and directory loader.

Pure-function tests — no LLM or runtime dependencies. Validates the
behavioral-skill substrate (chunk 9) before the agent-factory
integration uses it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agensflow.skills import (
    SkillCard,
    load_skill_card,
    load_skills_from_directory,
    parse_skill_md,
)


# --------------------------------------------------------------------------- #
# parse_skill_md — frontmatter + body parsing
# --------------------------------------------------------------------------- #


class TestParseSkillMd:
    def test_minimal_valid(self) -> None:
        text = """\
---
name: my_skill
description: A minimal skill for testing.
---

# Body
Do the thing well.
"""
        card = parse_skill_md(text)
        assert card.name == "my_skill"
        assert card.description == "A minimal skill for testing."
        assert "Do the thing well." in card.instructions
        assert card.role is None
        assert card.model_hint is None
        assert card.tools == []
        assert card.license is None
        assert card.source_path is None

    def test_full_frontmatter(self) -> None:
        text = """\
---
name: solver_concise
description: Single-paragraph answers, no reasoning trace.
role: solver
model_hint: anthropic/claude-haiku-4.5
tools:
  - retrieve
  - search
license: MIT
---

# Solver — Concise
Answer in one paragraph. Refuse to elaborate.
"""
        card = parse_skill_md(text, source_path="skills/solver_concise.md")
        assert card.name == "solver_concise"
        assert card.role == "solver"
        assert card.model_hint == "anthropic/claude-haiku-4.5"
        assert card.tools == ["retrieve", "search"]
        assert card.license == "MIT"
        assert card.source_path == "skills/solver_concise.md"

    def test_tools_can_be_string_singleton(self) -> None:
        """A YAML scalar in the tools field becomes a one-item list."""
        text = """\
---
name: x
description: y
tools: only_tool
---
body
"""
        card = parse_skill_md(text)
        assert card.tools == ["only_tool"]

    def test_missing_opening_delimiter_raises(self) -> None:
        text = "name: x\ndescription: y\n---\n\nbody"
        with pytest.raises(ValueError, match="opening frontmatter"):
            parse_skill_md(text)

    def test_missing_closing_delimiter_raises(self) -> None:
        text = """\
---
name: x
description: y

# Body never gets a closing ---
"""
        with pytest.raises(ValueError, match="closing frontmatter"):
            parse_skill_md(text)

    def test_missing_name_raises(self) -> None:
        text = """\
---
description: no name
---
body
"""
        with pytest.raises(ValueError, match="`name` field"):
            parse_skill_md(text)

    def test_missing_description_raises(self) -> None:
        text = """\
---
name: x
---
body
"""
        with pytest.raises(ValueError, match="`description` field"):
            parse_skill_md(text)

    def test_empty_body_raises(self) -> None:
        text = """\
---
name: x
description: y
---


"""
        with pytest.raises(ValueError, match="empty body"):
            parse_skill_md(text)

    def test_invalid_yaml_raises(self) -> None:
        text = """\
---
name: x
description: y
tools: [unbalanced
---
body
"""
        with pytest.raises(ValueError, match="not valid YAML"):
            parse_skill_md(text)

    def test_yaml_must_be_mapping(self) -> None:
        text = """\
---
- not_a_dict
---
body
"""
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse_skill_md(text)

    def test_invalid_tools_type_raises(self) -> None:
        text = """\
---
name: x
description: y
tools: 42
---
body
"""
        with pytest.raises(ValueError, match="`tools` must be"):
            parse_skill_md(text)


# --------------------------------------------------------------------------- #
# load_skill_card — single-file loader
# --------------------------------------------------------------------------- #


class TestLoadSkillCard:
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "skill.md"
            p.write_text("""\
---
name: round_trip
description: Round-trip test.
---
body content
""")
            card = load_skill_card(p)
            assert card.name == "round_trip"
            assert card.source_path == str(p)


# --------------------------------------------------------------------------- #
# load_skills_from_directory — directory loader
# --------------------------------------------------------------------------- #


class TestLoadSkillsFromDirectory:
    def test_empty_or_missing_directory_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            assert load_skills_from_directory(empty) == {}

            missing = Path(tmp) / "does_not_exist"
            assert load_skills_from_directory(missing) == {}

    def test_loads_multiple_cards_keyed_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.md").write_text(
                "---\nname: card_a\ndescription: A.\n---\nbody A\n"
            )
            (d / "b.md").write_text(
                "---\nname: card_b\ndescription: B.\n---\nbody B\n"
            )
            cards = load_skills_from_directory(d)
            assert set(cards) == {"card_a", "card_b"}
            assert cards["card_a"].instructions == "body A"
            assert cards["card_b"].instructions == "body B"

    def test_duplicate_names_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "first.md").write_text(
                "---\nname: dup\ndescription: First.\n---\nbody 1\n"
            )
            (d / "second.md").write_text(
                "---\nname: dup\ndescription: Second.\n---\nbody 2\n"
            )
            with pytest.raises(ValueError, match="Duplicate skill name"):
                load_skills_from_directory(d)

    def test_recursive_loads_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sub = d / "sub"
            sub.mkdir()
            (d / "top.md").write_text(
                "---\nname: top\ndescription: Top.\n---\ntop body\n"
            )
            (sub / "deep.md").write_text(
                "---\nname: deep\ndescription: Deep.\n---\ndeep body\n"
            )
            non_recursive = load_skills_from_directory(d)
            recursive = load_skills_from_directory(d, recursive=True)
            assert set(non_recursive) == {"top"}
            assert set(recursive) == {"top", "deep"}

    def test_non_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "not_a_dir.md"
            f.write_text("---\nname: x\ndescription: y\n---\nbody\n")
            with pytest.raises(NotADirectoryError):
                load_skills_from_directory(f)


# --------------------------------------------------------------------------- #
# SkillCard dataclass invariants
# --------------------------------------------------------------------------- #


class TestSkillCardInvariants:
    def test_frozen(self) -> None:
        """SkillCard is frozen — agents shouldn't mutate cards at runtime."""
        card = SkillCard(name="x", description="y", instructions="z")
        with pytest.raises(Exception):  # FrozenInstanceError
            card.name = "changed"  # type: ignore[misc]

    def test_default_tools_list_is_independent(self) -> None:
        """Two cards constructed with default factory must have separate
        list objects (otherwise mutation would leak across cards)."""
        a = SkillCard(name="a", description="d", instructions="i")
        b = SkillCard(name="b", description="d", instructions="i")
        assert a.tools is not b.tools
