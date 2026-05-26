"""
Tests for `agensflow.config` central loader.

Verified behaviors:
  - Loading library defaults (currently empty schema; expanded as
    modules are converted to OmegaConf flow)
  - Merging user YAML on top
  - Strict mode raises UnknownKeyError on typos
  - Permissive mode logs a warning + drops unknown keys
  - Multiple user files merge in order (later overrides earlier)
  - `extra` dict overrides on top of files
  - File-not-found is a clear error

Tests use a temporary structured schema injected via fixture so they
don't depend on which modules have been converted yet — they verify
the LOADER behavior, not specific module configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from agensflow.config import (
    ConfigError,
    UnknownKeyError,
    load_config,
)
from agensflow.config import loader as loader_mod


# --------------------------------------------------------------------------- #
# Fixture: a minimal schema we substitute in for AgensflowConfig
# --------------------------------------------------------------------------- #


@dataclass
class _SubA:
    knob: int = 4
    factor: float = 1.0


@dataclass
class _SubB:
    name: str = "default"
    enabled: bool = True


@dataclass
class _TestSchema:
    """Stand-in for AgensflowConfig used by the loader tests."""
    sub_a: _SubA = field(default_factory=_SubA)
    sub_b: _SubB = field(default_factory=_SubB)


@pytest.fixture
def patched_schema(monkeypatch):
    """Swap AgensflowConfig for our test schema for the duration of one
    test. Lets us verify loader behavior independent of which real
    module configs are wired in."""
    monkeypatch.setattr(loader_mod, "AgensflowConfig", _TestSchema)
    # Also point the defaults discovery at an empty list so the test
    # isn't influenced by any YAMLs shipped under configs/defaults/.
    monkeypatch.setattr(
        loader_mod, "_list_default_yaml_files", lambda: []
    )
    yield


# --------------------------------------------------------------------------- #
# Default-only load
# --------------------------------------------------------------------------- #


class TestLoadDefaults:
    def test_no_user_files_returns_defaults(self, patched_schema) -> None:
        cfg = load_config()
        assert cfg.sub_a.knob == 4
        assert cfg.sub_a.factor == 1.0
        assert cfg.sub_b.name == "default"
        assert cfg.sub_b.enabled is True


# --------------------------------------------------------------------------- #
# User YAML overrides
# --------------------------------------------------------------------------- #


class TestUserYamlOverrides:
    def test_single_yaml_overrides_one_field(self, patched_schema, tmp_path) -> None:
        user = tmp_path / "user.yaml"
        user.write_text("sub_a:\n  knob: 99\n")
        cfg = load_config(user)
        assert cfg.sub_a.knob == 99
        # Untouched defaults stay
        assert cfg.sub_a.factor == 1.0
        assert cfg.sub_b.name == "default"

    def test_multiple_yamls_merge_in_order(self, patched_schema, tmp_path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text("sub_a:\n  knob: 10\n")
        b = tmp_path / "b.yaml"
        b.write_text("sub_a:\n  knob: 20\nsub_b:\n  name: 'overridden'\n")
        cfg = load_config(a, b)
        # b wins on sub_a.knob
        assert cfg.sub_a.knob == 20
        # b also sets sub_b.name
        assert cfg.sub_b.name == "overridden"

    def test_partial_override_preserves_unset_fields(self, patched_schema, tmp_path) -> None:
        """Setting one field in a sub-config doesn't reset siblings to
        defaults — partial overrides really merge."""
        user = tmp_path / "user.yaml"
        user.write_text("sub_a:\n  knob: 7\n")  # factor unspecified
        cfg = load_config(user)
        assert cfg.sub_a.knob == 7
        assert cfg.sub_a.factor == 1.0  # default preserved


# --------------------------------------------------------------------------- #
# Programmatic extra dict
# --------------------------------------------------------------------------- #


class TestExtraOverrides:
    def test_extra_dict_overrides_after_files(self, patched_schema, tmp_path) -> None:
        user = tmp_path / "user.yaml"
        user.write_text("sub_a:\n  knob: 10\n")
        cfg = load_config(user, extra={"sub_a": {"knob": 999}})
        assert cfg.sub_a.knob == 999

    def test_no_extra_no_files_no_changes(self, patched_schema) -> None:
        cfg = load_config()
        assert cfg.sub_a.knob == 4


# --------------------------------------------------------------------------- #
# Strict vs permissive
# --------------------------------------------------------------------------- #


class TestStrictMode:
    def test_unknown_top_level_key_raises(self, patched_schema, tmp_path) -> None:
        user = tmp_path / "user.yaml"
        user.write_text("typo_at_top: 1\n")
        with pytest.raises(UnknownKeyError, match="typo_at_top"):
            load_config(user, strict=True)

    def test_unknown_nested_key_raises_with_dotted_path(
        self, patched_schema, tmp_path
    ) -> None:
        user = tmp_path / "user.yaml"
        user.write_text("sub_a:\n  knobb: 99\n")  # typo: knobb
        with pytest.raises(UnknownKeyError, match="sub_a.knobb"):
            load_config(user, strict=True)

    def test_unknown_in_extra_raises(self, patched_schema) -> None:
        with pytest.raises(UnknownKeyError, match="bogus"):
            load_config(extra={"bogus": True}, strict=True)


class TestPermissiveMode:
    def test_unknown_key_warns_and_drops(self, patched_schema, tmp_path, caplog) -> None:
        import logging
        user = tmp_path / "user.yaml"
        user.write_text("sub_a:\n  knobb: 99\n")  # typo
        with caplog.at_level(logging.WARNING, logger="agensflow.config"):
            cfg = load_config(user, strict=False)
        # Default preserved (typo dropped)
        assert cfg.sub_a.knob == 4
        # Warning emitted
        warnings = [r for r in caplog.records if r.name == "agensflow.config"]
        assert any("knobb" in r.message for r in warnings)


# --------------------------------------------------------------------------- #
# File-not-found
# --------------------------------------------------------------------------- #


class TestFileNotFound:
    def test_missing_user_file_raises_clear_error(self, patched_schema) -> None:
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            load_config("/tmp/does_not_exist_xxxxx.yaml")
