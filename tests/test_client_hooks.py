"""
Tests for Instructor hook integration in OpenRouterClient.

These tests verify the load-bearing claim of the chunk-2 client: that
validation retries are visible to the trace as discrete events, with token
counts and the validation error captured. This visibility is required for
honest cost accounting in Layer 2 metrics — without it, the framework would
silently undercount its own recovery overhead.

We mock the underlying OpenAI client so the tests are fast, deterministic,
and don't burn API budget. The mock is deliberately minimal: it returns
pre-canned ChatCompletion-shaped objects in sequence, simulating "first
attempt is malformed, second attempt is well-formed."
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agensflow.runtime.agent_outputs import PlannerOutput
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.errors import InvalidAgentOutputError
from agensflow.runtime.trace import TraceCollector


# --------------------------------------------------------------------------- #
# Mock OpenAI completion shape
# --------------------------------------------------------------------------- #


def _make_tool_call_message(arguments: str) -> SimpleNamespace:
    """Build a ChatCompletionMessage-shaped object with a tool call.

    Instructor in TOOLS mode parses the tool_calls[0].function.arguments
    field as the JSON to validate against the response_model.
    """
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(
            name="PlannerOutput",
            arguments=arguments,
        ),
    )
    return SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
        refusal=None,
    )


def _make_completion(arguments: str, prompt_tok: int = 100, completion_tok: int = 50) -> SimpleNamespace:
    """Build a ChatCompletion-shaped mock with one choice carrying a tool call."""
    return SimpleNamespace(
        id="chatcmpl-mock",
        model="mock/model",
        choices=[
            SimpleNamespace(
                index=0,
                message=_make_tool_call_message(arguments),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
        ),
    )


# --------------------------------------------------------------------------- #
# Real OpenAI client wired to a sequenced fake `create`
# --------------------------------------------------------------------------- #
#
# Instructor's `from_openai` does an isinstance(client, openai.OpenAI) check
# and returns None for non-OpenAI clients. So duck-typed mocks won't work —
# we need a real openai.OpenAI instance. We construct one with a fake api_key
# and never let it make a network call, then monkey-patch
# `chat.completions.create` to return our pre-canned responses.
#
# The patch must happen BEFORE `instructor.from_openai`, because Instructor
# captures the `create` callable at wrap-time (see the from_openai source).


def _build_client_with_responses(
    responses: list[Any],
) -> tuple[OpenRouterClient, list[dict[str, Any]]]:
    """
    Build an OpenRouterClient backed by a real openai.OpenAI whose
    `chat.completions.create` returns pre-canned responses in sequence.

    Returns the client and a list that will be appended to with each call's
    kwargs (for assertions about what was sent).
    """
    import instructor
    import openai
    from instructor import Mode

    real_openai = openai.OpenAI(api_key="fake-test-key-not-used")

    response_queue = list(responses)
    captured_calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        if not response_queue:
            raise RuntimeError("Mock transport exhausted")
        return response_queue.pop(0)

    # Patch BEFORE wrapping with Instructor — order matters.
    real_openai.chat.completions.create = fake_create  # type: ignore[method-assign]

    client = OpenRouterClient.__new__(OpenRouterClient)
    client._raw = real_openai  # type: ignore[attr-defined]
    client._default_mode = Mode.TOOLS  # type: ignore[attr-defined]
    client._instructor = instructor.from_openai(real_openai, mode=Mode.TOOLS)
    # Chunk 11.A2: per-call mode selection requires both wrappers.
    client._instructor_json = instructor.from_openai(real_openai, mode=Mode.JSON)  # type: ignore[attr-defined]
    client._register_hooks()
    return client, captured_calls


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestSuccessfulFirstAttempt:
    def test_no_failed_attempts_recorded(self) -> None:
        good_args = json.dumps(
            {"goal": "g", "subproblem": "s", "constraints": ["c1"]}
        )
        client, _ = _build_client_with_responses([_make_completion(good_args)])
        trace = TraceCollector()

        result = client.complete_typed(
            model="mock/model",
            system_prompt="sys",
            user_prompt="user",
            output_model=PlannerOutput,
            agent_name="planner",
            trace=trace,
            state_snapshot={"x": 1},
        )

        assert isinstance(result.parsed_output, PlannerOutput)
        assert result.parsed_output.goal == "g"
        assert result.failed_attempts == 0
        # No failed-attempt trace events should have been recorded.
        assert all(e.error is None for e in trace.events)


class TestValidationRetryIsRecorded:
    def test_failed_attempt_appears_in_trace_with_error(self) -> None:
        # First call: malformed (missing required `subproblem`).
        bad_args = json.dumps({"goal": "g"})
        # Second call: well-formed.
        good_args = json.dumps(
            {"goal": "g", "subproblem": "s", "constraints": []}
        )
        client, _ = _build_client_with_responses(
            [
                _make_completion(bad_args, prompt_tok=80, completion_tok=20),
                _make_completion(good_args, prompt_tok=120, completion_tok=30),
            ]
        )
        trace = TraceCollector()

        result = client.complete_typed(
            model="mock/model",
            system_prompt="sys",
            user_prompt="user",
            output_model=PlannerOutput,
            agent_name="planner",
            trace=trace,
            state_snapshot={"x": 1},
            max_retries=2,
        )

        # Final attempt succeeded.
        assert isinstance(result.parsed_output, PlannerOutput)
        assert result.failed_attempts == 1
        # Failed attempt is in the trace, marked with an error.
        failed_events = [e for e in trace.events if e.error is not None]
        assert len(failed_events) == 1
        assert failed_events[0].agent == "planner"
        # Failed-attempt tokens were captured (load-bearing for cost accounting).
        assert failed_events[0].prompt_tokens == 80
        assert failed_events[0].completion_tokens == 20
        assert failed_events[0].total_tokens == 100

    def test_two_failures_then_success(self) -> None:
        bad_1 = json.dumps({"goal": "g"})
        bad_2 = json.dumps({"goal": ""})  # empty string violates min_length
        good = json.dumps({"goal": "g", "subproblem": "s", "constraints": []})
        client, _ = _build_client_with_responses(
            [
                _make_completion(bad_1, prompt_tok=70, completion_tok=10),
                _make_completion(bad_2, prompt_tok=85, completion_tok=15),
                _make_completion(good, prompt_tok=100, completion_tok=25),
            ]
        )
        trace = TraceCollector()

        result = client.complete_typed(
            model="mock/model",
            system_prompt="sys",
            user_prompt="user",
            output_model=PlannerOutput,
            agent_name="planner",
            trace=trace,
            state_snapshot={},
            max_retries=3,  # allow 2 retries
        )

        assert result.failed_attempts == 2
        failed_events = [e for e in trace.events if e.error is not None]
        assert len(failed_events) == 2
        assert all(e.agent == "planner" for e in failed_events)


class TestRetryExhaustionRaises:
    def test_exhausted_retries_raise_invalid_agent_output(self) -> None:
        # Every call returns malformed output.
        bad = json.dumps({"goal": "g"})  # missing subproblem
        client, _ = _build_client_with_responses(
            [_make_completion(bad) for _ in range(3)]
        )
        trace = TraceCollector()

        with pytest.raises(InvalidAgentOutputError) as exc_info:
            client.complete_typed(
                model="mock/model",
                system_prompt="sys",
                user_prompt="user",
                output_model=PlannerOutput,
                agent_name="planner",
                trace=trace,
                state_snapshot={},
                max_retries=2,  # initial + 1 retry, both fail
            )

        assert exc_info.value.agent_name == "planner"
        # Both failed attempts were recorded to the trace before the raise.
        assert len([e for e in trace.events if e.error is not None]) == 2


class TestAttributionAcrossAgents:
    def test_consecutive_calls_attribute_correctly(self) -> None:
        good_planner = json.dumps(
            {"goal": "g1", "subproblem": "s1", "constraints": []}
        )
        # Second sequence: failure then success for a different "agent".
        bad = json.dumps({"goal": "g2"})
        good_planner_2 = json.dumps(
            {"goal": "g2", "subproblem": "s2", "constraints": []}
        )
        client, _ = _build_client_with_responses(
            [
                _make_completion(good_planner),
                _make_completion(bad, prompt_tok=60, completion_tok=10),
                _make_completion(good_planner_2),
            ]
        )
        trace = TraceCollector()

        client.complete_typed(
            model="mock/model",
            system_prompt="sys1",
            user_prompt="u1",
            output_model=PlannerOutput,
            agent_name="planner_call_1",
            trace=trace,
            state_snapshot={},
            max_retries=2,
        )
        client.complete_typed(
            model="mock/model",
            system_prompt="sys2",
            user_prompt="u2",
            output_model=PlannerOutput,
            agent_name="planner_call_2",
            trace=trace,
            state_snapshot={},
            max_retries=2,
        )

        failed = [e for e in trace.events if e.error is not None]
        assert len(failed) == 1
        # The failure must be attributed to the second logical call, not the first.
        assert failed[0].agent == "planner_call_2"
