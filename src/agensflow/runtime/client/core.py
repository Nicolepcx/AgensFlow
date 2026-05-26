"""
OpenRouter client, Instructor-wrapped.

This is the validation+retry transport layer for AgensFlow agents. It does
three things:

  1. Wraps the OpenAI Python client pointed at OpenRouter, so any provider on
     OpenRouter (OpenAI, Anthropic, Google, etc.) is reachable through one
     interface and one key.

  2. Wraps that with Instructor for typed I/O. Each agent passes a Pydantic
     `response_model`; Instructor enforces the schema via tools mode and does
     bounded corrective retries on validation failure (the discipline the
     framework's "no retry stacks" thesis requires).

  3. Registers Instructor hooks that record every failed attempt as a
     TraceEvent. This is load-bearing: the framework's headline cost claim
     ("fewer tokens per successful task than retry stacks") requires honest
     accounting of internal recovery events, which means failed retries must
     be visible to the metric layer.

Per-call agent context (which agent is calling, which trace to record into,
which input state to snapshot) is threaded to the hooks via a ContextVar so
the hook handlers can attribute completion events to the correct logical
agent invocation.
"""

from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import instructor
from instructor import Mode
from instructor.core.exceptions import InstructorRetryException
from openai import OpenAI
from pydantic import BaseModel

from agensflow.runtime.errors import InvalidAgentOutputError
from agensflow.runtime.trace import TraceCollector, TraceEvent

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Per-call context (carried through Instructor hooks via ContextVar)
# --------------------------------------------------------------------------- #


@dataclass
class _CallContext:
    """
    Carries per-call attribution and timing state through Instructor hooks.

    Set by `complete_typed` before each call, read by the hook handlers when
    they fire. Lives in a ContextVar so concurrent callers (eventually) don't
    interfere.
    """

    agent_name: str
    trace: TraceCollector
    state_snapshot: dict[str, Any]
    timer_start: float | None = None
    last_response: Any = None
    last_latency_seconds: float = 0.0
    failed_attempts: list[TraceEvent] = field(default_factory=list)


_current_call: contextvars.ContextVar[_CallContext | None] = contextvars.ContextVar(
    "agensflow_current_call", default=None
)


# --------------------------------------------------------------------------- #
# Result types (kept for compatibility and convenience)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompletionResult:
    """Structured result of a single typed LLM call.

    Returned by `complete_typed` alongside the parsed Pydantic object. Carries
    the final-attempt's transport-level statistics (tokens, latency) so the
    caller can record a TraceEvent for the successful attempt. Failed-attempt
    events are recorded automatically by the client's hooks.
    """

    parsed_output: BaseModel
    raw_completion: Any
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    failed_attempts: int


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class OpenRouterClient:
    """
    Synchronous OpenRouter client with Instructor-enforced typed I/O.

    Configuration:
      api_key:  OPENROUTER_API_KEY env var by default. Pass to override.
      base_url: OpenRouter's OpenAI-compatible endpoint by default.
      app_name, site_url: optional headers OpenRouter uses for attribution.
      timeout_seconds: per-request timeout passed to the underlying SDK.
      mode: Instructor mode. Tools mode is the most provider-portable choice
            and gives the strongest schema enforcement.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        app_name: str = "AgensFlow",
        site_url: str = "https://agensflow.ai",
        timeout_seconds: float = 60.0,
        mode: Mode = Mode.TOOLS,
        max_transport_retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Either pass api_key= explicitly, "
                "or export OPENROUTER_API_KEY in your environment (a .env file at "
                "the repo root with python-dotenv loaded works too)."
            )

        # The OpenAI SDK has built-in transport retries (transient network /
        # rate-limit errors). We rely on those and don't add another layer —
        # stacking retry layers would create the very pattern this framework
        # criticizes.
        self._raw = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_transport_retries,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )
        # The default mode + a JSON-mode wrapper. Per-call `mode=` in
        # `complete_typed` selects which wrapper handles a given call —
        # critical for cross-judge runs where some models route via
        # tool-calling (TOOLS) and others require JSON-mode fallback
        # (chunk 11.A2 finding: qwen + grok routes on OpenRouter).
        self._default_mode = mode
        self._instructor = instructor.from_openai(self._raw, mode=mode)
        self._instructor_json = instructor.from_openai(self._raw, mode=Mode.JSON)
        self._register_hooks()

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #

    def complete_typed(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_model: type[T],
        agent_name: str,
        trace: TraceCollector,
        state_snapshot: dict[str, Any],
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        mode: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """
        Run a typed completion with bounded corrective retries.

        Behavior:
          - On success (first attempt or after corrective retry): returns a
            CompletionResult containing the parsed Pydantic object and the
            final-attempt statistics. The caller records a TraceEvent for the
            successful attempt.
          - On any failed attempt before success: a TraceEvent with
            `error="..."` is recorded automatically by the parse-error hook,
            so AR / cost metrics see the recovery event.
          - On exhausted retries: raises InvalidAgentOutputError, mapped from
            Instructor's InstructorRetryException, with the last failure's
            details preserved.

        max_retries semantics: Instructor counts the *initial* attempt plus
        retries. So max_retries=2 means "first attempt + at most one corrective
        retry" — exactly the bounded-recovery discipline the framework
        requires.

        `mode`: per-call override of the Instructor mode. None = use the
        client's default (TOOLS). "json" or "JSON" = use JSON mode (no
        tool_choice in the request, structured-output enforced via
        prompt + Pydantic parse). Required for OpenRouter routes that
        don't support tool_choice (qwen/* family, grok-4.3 — see
        chunk-11 probe results in `learning/ruler/README.md`).

        `extra_body`: dict passed through to the OpenAI SDK's
        `extra_body` kwarg. The Instructor+OpenRouter integration
        pattern uses `{"provider": {"require_parameters": True}}` to
        force OpenRouter to route to providers that fully support all
        parameters. Required for qwen/grok JSON-mode calls; BREAKS
        OpenAI's primary route (paradoxically — see chunk-11 probe).
        Per-judge config in RelativeJudgeConfig handles this asymmetry.
        """
        ctx = _CallContext(
            agent_name=agent_name,
            trace=trace,
            state_snapshot=state_snapshot,
        )
        # Pick the Instructor wrapper based on the requested mode.
        if mode is None or mode.lower() in ("tools", "tool", "tool_call"):
            inst = self._instructor
        elif mode.lower() in ("json", "json_mode"):
            inst = self._instructor_json
        else:
            raise ValueError(
                f"Unsupported Instructor mode {mode!r}. "
                f"Supported: 'tools' (default), 'json'. "
                f"Other Instructor modes can be added by extending "
                f"OpenRouterClient.__init__ to build the corresponding "
                f"wrapper."
            )

        # Build the kwargs dict. Only pass extra_body when supplied —
        # passing None or {} can interact weirdly with OpenRouter's
        # routing on some models (chunk-11 probe finding: OpenAI route
        # fails when require_parameters is on, regardless of intent).
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_model": output_model,
            "max_retries": max_retries,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        token = _current_call.set(ctx)
        try:
            try:
                parsed, completion = (
                    inst.chat.completions.create_with_completion(**call_kwargs)
                )
            except InstructorRetryException as exc:
                # Map Instructor's retry-exhaustion into our typed error so
                # callers don't need to know about the underlying library.
                last = exc.failed_attempts[-1] if exc.failed_attempts else None
                reason = (
                    f"validation failed after {len(exc.failed_attempts)} attempts: "
                    f"{last!r}"
                    if last is not None
                    else "validation retries exhausted"
                )
                raise InvalidAgentOutputError(
                    agent_name=agent_name,
                    raw_content=str(last) if last is not None else "",
                    reason=reason,
                ) from exc

            usage = getattr(completion, "usage", None)
            return CompletionResult(
                parsed_output=parsed,
                raw_completion=completion,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency_seconds=ctx.last_latency_seconds,
                model=getattr(completion, "model", model),
                failed_attempts=len(ctx.failed_attempts),
            )
        finally:
            _current_call.reset(token)

    # ----------------------------------------------------------------------- #
    # Hooks (private)
    # ----------------------------------------------------------------------- #

    def _register_hooks(self) -> None:
        # Register hooks on BOTH wrappers (TOOLS + JSON) — failed-attempt
        # trace events must be recorded regardless of which mode the call
        # used. Otherwise JSON-mode judges' parse failures become invisible
        # to the metric layer.
        for wrapper in (self._instructor, self._instructor_json):
            wrapper.on("completion:kwargs", self._on_kwargs)
            wrapper.on("completion:response", self._on_response)
            wrapper.on("parse:error", self._on_parse_error)

    def _on_kwargs(self, *args: Any, **kwargs: Any) -> None:
        """Fires before each underlying completion call. Start the timer."""
        ctx = _current_call.get()
        if ctx is None:
            return
        ctx.timer_start = time.monotonic()

    def _on_response(self, response: Any, *args: Any, **kwargs: Any) -> None:
        """Fires after each underlying completion call. Capture timing + raw response."""
        ctx = _current_call.get()
        if ctx is None:
            return
        if ctx.timer_start is not None:
            ctx.last_latency_seconds = time.monotonic() - ctx.timer_start
        ctx.last_response = response

    def _on_parse_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        """
        Fires when Instructor fails to parse/validate an attempt.

        This is the failed-attempt boundary. We emit a TraceEvent with the
        validation error so the metric layer can count this as a recovery
        event. Tokens still count.
        """
        ctx = _current_call.get()
        if ctx is None:
            return

        usage = getattr(ctx.last_response, "usage", None)
        model = getattr(ctx.last_response, "model", "unknown")

        event = TraceEvent(
            agent=ctx.agent_name,
            model=model,
            input_state=ctx.state_snapshot,
            output_update={"_validation_error": str(error)[:200]},
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_seconds=ctx.last_latency_seconds,
            error=str(error)[:200],
        )
        ctx.trace.record(event)
        ctx.failed_attempts.append(event)
