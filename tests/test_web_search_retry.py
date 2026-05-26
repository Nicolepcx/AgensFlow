"""
Tests for Layer 0 — web-search retry/backoff/clamp wrapper.

Covers:
  - `_is_rate_limited`: detects 429, 432, 402-with-credits, throttle keywords
  - `_clamp_exa_args`: bounds expensive params to safe defaults
  - `_backoff_seconds`: exponential backoff math, capped
  - `_exa_request_with_retry`: retries on rate-limit, raises on terminal
  - `_tavily_request_with_retry`: same semantics for Tavily

All tests mock httpx.post so no real network calls are made. The
`sleep_fn` parameter is overridden to avoid actually sleeping during
backoff — tests record sleep durations and assert on them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agensflow.runtime.web_search import (
    EXA_BACKOFF_BASE_S,
    EXA_BACKOFF_CAP_S,
    EXA_MAX_RETRIES,
    _backoff_seconds,
    _clamp_exa_args,
    _exa_request_with_retry,
    _is_rate_limited,
    _tavily_request_with_retry,
)


# --------------------------------------------------------------------------- #
# _is_rate_limited
# --------------------------------------------------------------------------- #


class TestIsRateLimited:
    def test_429_detected(self) -> None:
        assert _is_rate_limited("HTTP 429 Too Many Requests")

    def test_432_detected(self) -> None:
        assert _is_rate_limited("HTTP 432 plan exceeded")

    def test_402_with_credits_keyword(self) -> None:
        # Exa's quirk: 402 with credits-limit verbiage on burst traffic
        assert _is_rate_limited(
            "HTTP 402 You have exceeded your credits limit"
        )

    def test_402_with_rate_keyword(self) -> None:
        assert _is_rate_limited("HTTP 402 rate exceeded")

    def test_throttle_keyword(self) -> None:
        assert _is_rate_limited("upstream provider throttling requests")

    def test_rate_limit_phrase(self) -> None:
        assert _is_rate_limited("rate limit reached")

    def test_clean_500_not_rate_limited(self) -> None:
        # 500 is server-side, terminal — should NOT be retried as rate-limit
        assert not _is_rate_limited("HTTP 500 Internal Server Error")

    def test_auth_401_not_rate_limited(self) -> None:
        # Auth errors are terminal — should NOT be retried
        assert not _is_rate_limited("HTTP 401 invalid api key")

    def test_schema_error_not_rate_limited(self) -> None:
        assert not _is_rate_limited("ValidationError: missing field 'query'")

    def test_402_without_credits_or_rate_keyword(self) -> None:
        # Bare 402 without credits/rate language — could be other 402 cases
        # we don't recognize. Be conservative: don't classify as rate-limit.
        assert not _is_rate_limited("HTTP 402 unsupported payment method")


# --------------------------------------------------------------------------- #
# _backoff_seconds
# --------------------------------------------------------------------------- #


class TestBackoffSeconds:
    def test_first_attempt(self) -> None:
        assert _backoff_seconds(1, base=1.0, cap=30.0) == 1.0

    def test_exponential_growth(self) -> None:
        assert _backoff_seconds(2, base=1.0, cap=30.0) == 2.0
        assert _backoff_seconds(3, base=1.0, cap=30.0) == 4.0
        assert _backoff_seconds(4, base=1.0, cap=30.0) == 8.0

    def test_capped(self) -> None:
        # 1 * 2^9 = 512, but cap is 30 → 30
        assert _backoff_seconds(10, base=1.0, cap=30.0) == 30.0


# --------------------------------------------------------------------------- #
# _clamp_exa_args
# --------------------------------------------------------------------------- #


class TestClampExaArgs:
    def test_caps_num_results(self) -> None:
        out = _clamp_exa_args({"numResults": 100})
        assert out["numResults"] == 3

    def test_min_num_results(self) -> None:
        out = _clamp_exa_args({"numResults": 0})
        assert out["numResults"] == 1

    def test_default_num_results(self) -> None:
        out = _clamp_exa_args({})
        assert out["numResults"] == 3

    def test_default_type_auto(self) -> None:
        out = _clamp_exa_args({})
        assert out["type"] == "auto"

    def test_user_type_preserved(self) -> None:
        # Clamp uses setdefault — user-supplied "neural" stays
        out = _clamp_exa_args({"type": "neural"})
        assert out["type"] == "neural"

    def test_caps_context_max_chars(self) -> None:
        out = _clamp_exa_args({"contextMaxCharacters": 50000})
        assert out["contextMaxCharacters"] == 6000

    def test_default_context_max_chars(self) -> None:
        out = _clamp_exa_args({})
        assert out["contextMaxCharacters"] == 6000

    def test_does_not_mutate_input(self) -> None:
        original = {"numResults": 100, "extra_field": "preserved"}
        out = _clamp_exa_args(original)
        assert original["numResults"] == 100  # unchanged
        assert out["extra_field"] == "preserved"  # passed through


# --------------------------------------------------------------------------- #
# _exa_request_with_retry
# --------------------------------------------------------------------------- #


def _make_mock_response(status: int, body: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.json.return_value = {"results": [{"title": "ok", "text": "..."}]} if status == 200 else {}
    resp.raise_for_status.side_effect = (
        None if status < 400 else httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp,
        )
    )
    resp.request = MagicMock()
    return resp


class TestExaRequestWithRetry:
    def test_success_on_first_attempt(self) -> None:
        sleeps: list[float] = []
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(200)
            data = _exa_request_with_retry(
                api_key="x", query="q", max_results=3,
                sleep_fn=sleeps.append,
            )
        assert data == {"results": [{"title": "ok", "text": "..."}]}
        assert mock_post.call_count == 1
        assert sleeps == []  # no retries → no sleeps

    def test_retries_on_429_then_succeeds(self) -> None:
        sleeps: list[float] = []
        responses = [
            _make_mock_response(429, "Too Many Requests"),
            _make_mock_response(429, "Too Many Requests"),
            _make_mock_response(200),
        ]
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.side_effect = responses
            data = _exa_request_with_retry(
                api_key="x", query="q", max_results=3,
                max_retries=4, backoff_base=1.0, backoff_cap=30.0,
                sleep_fn=sleeps.append,
            )
        assert data["results"][0]["title"] == "ok"
        assert mock_post.call_count == 3
        # 2 backoffs: attempt 1 → 1s, attempt 2 → 2s
        assert sleeps == [1.0, 2.0]

    def test_retries_on_402_with_credits_then_succeeds(self) -> None:
        """Exa's quirk: 402 with 'credits limit' on burst traffic should
        be treated as rate-limit and retried, not as terminal."""
        sleeps: list[float] = []
        responses = [
            _make_mock_response(402, "You have exceeded your credits limit"),
            _make_mock_response(200),
        ]
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.side_effect = responses
            data = _exa_request_with_retry(
                api_key="x", query="q", max_results=3,
                max_retries=4, sleep_fn=sleeps.append,
            )
        assert data["results"][0]["title"] == "ok"
        assert mock_post.call_count == 2
        assert sleeps == [1.0]

    def test_terminal_error_raises_without_retry(self) -> None:
        """Auth errors (401/403) should fail-fast, not be retried."""
        sleeps: list[float] = []
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(401, "invalid api key")
            with pytest.raises(httpx.HTTPStatusError):
                _exa_request_with_retry(
                    api_key="bad", query="q", max_results=3,
                    sleep_fn=sleeps.append,
                )
        assert mock_post.call_count == 1
        assert sleeps == []  # no retries for terminal errors

    def test_exhausts_retries_on_persistent_rate_limit(self) -> None:
        sleeps: list[float] = []
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(429, "Too Many Requests")
            with pytest.raises(httpx.HTTPStatusError, match="rate-limited after"):
                _exa_request_with_retry(
                    api_key="x", query="q", max_results=3,
                    max_retries=3, backoff_base=1.0, backoff_cap=30.0,
                    sleep_fn=sleeps.append,
                )
        # 3 attempts → 2 sleeps in between (no sleep before final raise)
        assert mock_post.call_count == 3
        assert sleeps == [1.0, 2.0]

    def test_clamps_args_in_request_body(self) -> None:
        """The retry wrapper applies _clamp_exa_args internally — even if
        a caller passed numResults=100 (unrealistic), the request body
        sent to Exa caps it at 3."""
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(200)
            _exa_request_with_retry(
                api_key="x", query="q", max_results=100,
                sleep_fn=lambda s: None,
            )
        body = mock_post.call_args.kwargs["json"]
        assert body["numResults"] == 3  # clamped from 100
        assert body["contextMaxCharacters"] == 6000  # default-clamped


# --------------------------------------------------------------------------- #
# _tavily_request_with_retry
# --------------------------------------------------------------------------- #


class TestTavilyRequestWithRetry:
    def test_success_on_first_attempt(self) -> None:
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(200)
            data = _tavily_request_with_retry(
                api_key="x", query="q", max_results=4,
                sleep_fn=lambda s: None,
            )
        assert data["results"][0]["title"] == "ok"

    def test_retries_on_432_then_succeeds(self) -> None:
        """Tavily's plan-overage signal (432) is retried as a throttle."""
        sleeps: list[float] = []
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.side_effect = [
                _make_mock_response(432, "plan exceeded"),
                _make_mock_response(200),
            ]
            data = _tavily_request_with_retry(
                api_key="x", query="q", max_results=4,
                max_retries=4, sleep_fn=sleeps.append,
            )
        assert data["results"][0]["title"] == "ok"
        assert sleeps == [1.0]

    def test_clamps_max_results(self) -> None:
        with patch("agensflow.runtime.web_search.core.httpx.post") as mock_post:
            mock_post.return_value = _make_mock_response(200)
            _tavily_request_with_retry(
                api_key="x", query="q", max_results=50,
                sleep_fn=lambda s: None,
            )
        body = mock_post.call_args.kwargs["json"]
        assert body["max_results"] == 5  # clamped from 50
