"""
Smoke tests for the chunk-9 runner's `--config` flag and the
defaults / YAML / CLI override merge.

These tests exercise the runner's `main()` entry point with
`--print-config`, which loads the merged config + dumps it + exits.
No LLM tokens are spent, no filesystem writes happen, no preflight
probes fire — it's a pure config-resolution smoke test.

What we verify:

  1. `main(["--print-config"])` exits 0 with framework defaults.
  2. A user YAML overlays cleanly (CLI value < YAML value < default).
  3. CLI overrides win over YAML for individual knobs.
  4. The mutually-exclusive boolean flags (`--no-skip` / `--enable-skip`,
     `--router-log` / `--no-router-log`) exit 2 when both are passed.

The smoke test catches the regression class that motivated this
refactor: someone tweaks the runner's flag parsing and the YAML
override silently stops working.
"""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import pytest

from experiments.e07_skill_variants.run import main


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run(argv: list[str]) -> tuple[int, str]:
    """Invoke main(argv) capturing stdout. Returns (exit_code, captured)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def _grep_section(captured: str, section: str) -> str:
    """Return just the lines under one [section] header in --print-config output."""
    lines = captured.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == f"[{section}]":
            in_section = True
            continue
        if in_section:
            if line.startswith("[") or line.startswith("==="):
                break
            if line.strip():
                out.append(line.strip())
    return "\n".join(out)


def _grep_resolved(captured: str, knob: str) -> str:
    """Return the value of one [Effective CLI-resolved values] entry.

    Scoped to lines AFTER the "Effective CLI-resolved values" header
    so we don't accidentally match the per-section dump above (which
    also has e.g. `judge_model` under `[ruler]`).
    """
    in_resolved_block = False
    for line in captured.splitlines():
        if "Effective CLI-resolved values" in line:
            in_resolved_block = True
            continue
        if not in_resolved_block:
            continue
        s = line.strip()
        if s.startswith(f"{knob}"):
            return s.split("=", 1)[1].strip()
    raise AssertionError(f"Effective CLI-resolved value {knob!r} not in output")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestPrintConfigDefaults:
    def test_exits_zero(self) -> None:
        rc, _ = _run(["--print-config"])
        assert rc == 0

    def test_prints_every_section(self) -> None:
        _, out = _run(["--print-config"])
        # All 12 AgensflowConfig sections should appear.
        for section in (
            "web_search", "governance", "preflight", "report",
            "client", "models", "graph", "persistence",
            "policy_graph", "router", "reward", "ruler",
        ):
            assert f"[{section}]" in out, f"missing section [{section}]"

    def test_default_resolved_values(self) -> None:
        _, out = _run(["--print-config"])
        # judge_model defaults to ruler config's value (haiku).
        assert "claude-haiku-4.5" in _grep_resolved(out, "judge_model")
        # max_steps gets bumped to chunk-9's floor of 18 even when
        # cfg.router.max_steps is the framework default of 12.
        assert _grep_resolved(out, "max_steps") == "18"
        # Confidence threshold + reliability weight = policy_graph defaults.
        assert _grep_resolved(out, "confidence_threshold") == "5"
        assert _grep_resolved(out, "reliability_weight") == "0.5"
        # Booleans default to False.
        assert _grep_resolved(out, "enable_skip") == "False"
        assert _grep_resolved(out, "enable_router_logging") == "False"


class TestUserYamlOverride:
    @pytest.fixture
    def user_yaml(self, tmp_path: Path) -> Path:
        path = tmp_path / "test_run.yaml"
        path.write_text(
            "router:\n"
            "  enable_skip: true\n"
            "  enable_router_logging: true\n"
            "policy_graph:\n"
            "  confidence_threshold: 8\n"
            "  reliability_weight: 0.75\n"
            "ruler:\n"
            "  judge_model: \"openai/gpt-5.4-mini\"\n"
            "reward:\n"
            "  cost_weight: 0.5\n"
        )
        return path

    def test_yaml_values_take_effect(self, user_yaml: Path) -> None:
        rc, out = _run(["--print-config", "--config", str(user_yaml)])
        assert rc == 0
        assert "Loaded config from" in out
        # YAML overrides flow through to "Effective CLI-resolved values":
        assert _grep_resolved(out, "enable_skip") == "True"
        assert _grep_resolved(out, "enable_router_logging") == "True"
        assert _grep_resolved(out, "confidence_threshold") == "8"
        assert _grep_resolved(out, "reliability_weight") == "0.75"
        assert "gpt-5.4-mini" in _grep_resolved(out, "judge_model")

    def test_yaml_section_dump_matches_overrides(self, user_yaml: Path) -> None:
        _, out = _run(["--print-config", "--config", str(user_yaml)])
        router_section = _grep_section(out, "router")
        assert "enable_skip = True" in router_section
        assert "enable_router_logging = True" in router_section

        reward_section = _grep_section(out, "reward")
        assert "cost_weight = 0.5" in reward_section
        # Other reward knobs stay at defaults — proves the merge didn't
        # nuke them.
        assert "ruler_weight = 1.0" in reward_section

    def test_unknown_key_strict_raises(self, tmp_path: Path) -> None:
        # A typo'd knob should fail loudly via the loader, not silently
        # be ignored. Catches the regression class where YAML-driven
        # config silently does nothing.
        bad = tmp_path / "typo.yaml"
        bad.write_text("policy_graph:\n  confidence_treshold: 8\n")  # typo
        from agensflow.config import UnknownKeyError
        with pytest.raises(UnknownKeyError, match="confidence_treshold"):
            _run(["--print-config", "--config", str(bad)])


class TestCliOverridesYaml:
    @pytest.fixture
    def user_yaml(self, tmp_path: Path) -> Path:
        path = tmp_path / "yaml_says_5.yaml"
        path.write_text(
            "router:\n  enable_skip: true\n"
            "policy_graph:\n  confidence_threshold: 5\n"
        )
        return path

    def test_cli_int_override_wins(self, user_yaml: Path) -> None:
        rc, out = _run([
            "--print-config", "--config", str(user_yaml),
            "--confidence-threshold", "20",
        ])
        assert rc == 0
        # YAML had 5; CLI passed 20; effective should be 20.
        assert _grep_resolved(out, "confidence_threshold") == "20"

    def test_no_skip_override_wins(self, user_yaml: Path) -> None:
        # YAML has enable_skip: true; --no-skip flips it back to False.
        _, out = _run([
            "--print-config", "--config", str(user_yaml), "--no-skip",
        ])
        assert _grep_resolved(out, "enable_skip") == "False"

    def test_enable_skip_override_wins(self, tmp_path: Path) -> None:
        # YAML has enable_skip: false (or omits it); --enable-skip turns it on.
        yaml = tmp_path / "skip_off.yaml"
        yaml.write_text("router:\n  enable_skip: false\n")
        _, out = _run([
            "--print-config", "--config", str(yaml), "--enable-skip",
        ])
        assert _grep_resolved(out, "enable_skip") == "True"

    def test_judge_model_override_wins(self, user_yaml: Path) -> None:
        _, out = _run([
            "--print-config", "--config", str(user_yaml),
            "--judge-model", "openai/gpt-5.4-pro",
        ])
        assert "gpt-5.4-pro" in _grep_resolved(out, "judge_model")


class TestMutuallyExclusiveFlags:
    def test_skip_flags_are_exclusive(self) -> None:
        # argparse exits with SystemExit(2) on mutually-exclusive collision.
        with pytest.raises(SystemExit) as exc_info:
            _run(["--print-config", "--no-skip", "--enable-skip"])
        assert exc_info.value.code == 2

    def test_router_log_flags_are_exclusive(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _run(["--print-config", "--router-log", "--no-router-log"])
        assert exc_info.value.code == 2
