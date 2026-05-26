"""
Tests for Layer 5 — pre-flight check module.

Mocks all HTTP and env-var access so no real network calls or real
secrets are needed. Verifies:
  - `check_openrouter` / `check_exa` / `check_tavily` correctly classify
    success, auth failure, quota, rate-limit, missing env var
  - `PreflightResult.all_passed` semantics (skipped checks don't fail)
  - `format_report` produces a readable summary
  - `run_preflight_checks` aggregates correctly + supports subset
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agensflow.runtime.governance import AgentErrorReason
from agensflow.runtime.preflight import (
    DEFAULT_CHECKS,
    CheckResult,
    PreflightResult,
    check_exa,
    check_openrouter,
    check_tavily,
    run_preflight_checks,
)


# --------------------------------------------------------------------------- #
# CheckResult / PreflightResult
# --------------------------------------------------------------------------- #


class TestPreflightResult:
    def test_all_passed_with_all_passing(self) -> None:
        r = PreflightResult(checks=[
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=True, detail="ok"),
        ])
        assert r.all_passed

    def test_all_passed_with_one_failing(self) -> None:
        r = PreflightResult(checks=[
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=False, detail="fail"),
        ])
        assert not r.all_passed

    def test_skipped_check_does_not_fail_aggregate(self) -> None:
        """A check marked `not_configured=True` (env var missing) should
        NOT count as a failure — the user opted out of that dependency."""
        r = PreflightResult(checks=[
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=False, detail="not set",
                        not_configured=True),
        ])
        assert r.all_passed

    def test_format_report_lists_each_check(self) -> None:
        r = PreflightResult(checks=[
            CheckResult(name="alpha", passed=True, detail="auth ok",
                        elapsed_seconds=0.4),
            CheckResult(name="beta", passed=False, detail="HTTP 401",
                        error_reason=AgentErrorReason.AUTH,
                        suggested_fix="Check your beta API key.",
                        elapsed_seconds=0.7),
        ])
        out = r.format_report()
        assert "alpha" in out
        assert "auth ok" in out
        assert "beta" in out
        assert "HTTP 401" in out
        assert "Check your beta API key." in out
        assert "Pre-flight check report" in out

    def test_format_report_includes_skip_marker(self) -> None:
        r = PreflightResult(checks=[
            CheckResult(name="alpha", passed=False, detail="X_KEY not set",
                        not_configured=True),
        ])
        out = r.format_report()
        assert "skipped" in out


# --------------------------------------------------------------------------- #
# check_openrouter
# --------------------------------------------------------------------------- #


def _mock_resp(status: int, body: object = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = "" if body is None else (
        body if isinstance(body, str) else __import__("json").dumps(body)
    )
    r.json.return_value = body if isinstance(body, dict) else {}
    return r


class TestCheckOpenrouter:
    def test_missing_env_var_returns_not_configured(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            r = check_openrouter()
        assert not r.passed
        assert r.not_configured is True

    def test_success(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("agensflow.runtime.preflight.core.httpx.get") as m:
            m.return_value = _mock_resp(200, {"data": [{"id": "x"}, {"id": "y"}]})
            r = check_openrouter()
        assert r.passed
        assert "auth ok" in r.detail
        assert "2 models" in r.detail

    def test_401_classified_as_auth(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-bad"}), \
             patch("agensflow.runtime.preflight.core.httpx.get") as m:
            m.return_value = _mock_resp(401, "Unauthorized")
            r = check_openrouter()
        assert not r.passed
        assert r.error_reason == AgentErrorReason.AUTH
        assert "API key" in r.suggested_fix or "OPENROUTER_API_KEY" in r.suggested_fix

    def test_network_error_classified(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("agensflow.runtime.preflight.core.httpx.get") as m:
            m.side_effect = ConnectionError("Connection refused")
            r = check_openrouter()
        assert not r.passed
        assert r.error_reason == AgentErrorReason.NETWORK


# --------------------------------------------------------------------------- #
# check_exa
# --------------------------------------------------------------------------- #


class TestCheckExa:
    def test_missing_env_var_returns_not_configured(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            r = check_exa()
        assert r.not_configured is True

    def test_success(self) -> None:
        with patch.dict("os.environ", {"EXA_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.post") as m:
            m.return_value = _mock_resp(200, {"results": [{"title": "x"}]})
            r = check_exa()
        assert r.passed
        assert "1 results" in r.detail

    def test_402_with_credits_classified_as_rate_limited(self) -> None:
        """The chunk-9 finding: Exa's 402-with-credits-language is a
        throttle signal, not real exhaustion. Pre-flight must classify
        it correctly so the user gets the right diagnosis."""
        with patch.dict("os.environ", {"EXA_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.post") as m:
            m.return_value = _mock_resp(
                402, "You have exceeded your credits limit"
            )
            r = check_exa()
        assert not r.passed
        assert r.error_reason == AgentErrorReason.RATE_LIMITED
        # Suggested fix should mention throttling, not credits
        assert "throttl" in r.suggested_fix.lower() or "lower" in r.suggested_fix.lower()

    def test_402_unsupported_payment_classified_as_quota(self) -> None:
        """Bare 402 without credits-limit/rate language IS quota
        exhaustion — distinct from the rate-limited-as-402 case."""
        with patch.dict("os.environ", {"EXA_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.post") as m:
            m.return_value = _mock_resp(402, "unsupported payment method")
            r = check_exa()
        assert not r.passed
        assert r.error_reason == AgentErrorReason.QUOTA
        assert "dashboard.exa.ai" in r.suggested_fix


# --------------------------------------------------------------------------- #
# check_tavily
# --------------------------------------------------------------------------- #


class TestCheckTavily:
    def test_success(self) -> None:
        with patch.dict("os.environ", {"TAVILY_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.post") as m:
            m.return_value = _mock_resp(200, {"results": [{"title": "x"}]})
            r = check_tavily()
        assert r.passed

    def test_432_classified_as_rate_limited(self) -> None:
        """Tavily's 432 plan-overage code is treated as throttle —
        sometimes clears between requests."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.post") as m:
            m.return_value = _mock_resp(432, "plan exceeded")
            r = check_tavily()
        assert not r.passed
        assert r.error_reason == AgentErrorReason.RATE_LIMITED


# --------------------------------------------------------------------------- #
# run_preflight_checks aggregator
# --------------------------------------------------------------------------- #


class TestRunPreflightChecks:
    def test_runs_all_default_checks(self) -> None:
        # Mock every dependency to "skipped" by clearing env vars.
        with patch.dict("os.environ", {}, clear=True):
            r = run_preflight_checks()
        assert {c.name for c in r.checks} == {"openrouter", "exa", "tavily"}
        # All skipped → all_passed should be True (skipped doesn't fail).
        assert r.all_passed

    def test_subset_of_checks(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
             patch("agensflow.runtime.preflight.core.httpx.get") as m:
            m.return_value = _mock_resp(200, {"data": []})
            r = run_preflight_checks(checks=["openrouter"])
        assert len(r.checks) == 1
        assert r.checks[0].name == "openrouter"
        assert r.all_passed

    def test_unknown_check_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown pre-flight check"):
            run_preflight_checks(checks=["nope_does_not_exist"])

    def test_custom_registry(self) -> None:
        custom = {
            "always_pass": lambda: CheckResult(
                name="always_pass", passed=True, detail="forced",
            ),
            "always_fail": lambda: CheckResult(
                name="always_fail", passed=False, detail="forced fail",
            ),
        }
        r = run_preflight_checks(registry=custom)
        assert {c.name for c in r.checks} == {"always_pass", "always_fail"}
        assert not r.all_passed

    def test_default_registry_has_three_checks(self) -> None:
        assert set(DEFAULT_CHECKS.keys()) == {"openrouter", "exa", "tavily"}
