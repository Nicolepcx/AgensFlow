"""
Naive baseline.

A single LLM call with the user task and any documents inlined into the user
message. No orchestration, no validation, no retries beyond the OpenAI SDK's
built-in transport layer. This is the apples-to-oranges comparison: what does
"just call the model" produce?

We use claude-haiku-4.5 because it is the strongest model in the agensflow
default mix — using a weaker model for the baseline would create an unfair
comparison that flatters the framework.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from agensflow import Document

NAIVE_MODEL = "anthropic/claude-haiku-4.5"


@dataclass(frozen=True)
class NaiveResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str


def _build_prompt(user_task: str, documents: list[Document]) -> str:
    if not documents:
        return user_task
    docs_block = "\n\n".join(f"[{d.id}]\n{d.text}" for d in documents)
    return f"{user_task}\n\nReference documents:\n{docs_block}"


def run_naive(
    user_task: str,
    documents: list[Document],
    *,
    api_key: str | None = None,
    model: str = NAIVE_MODEL,
    load_env: bool = True,
) -> NaiveResult:
    """Single LLM call. No orchestration."""
    if load_env:
        load_dotenv()

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://agensflow.ai",
            "X-Title": "AgensFlow Experiment 01 (naive baseline)",
        },
    )

    user_msg = _build_prompt(user_task, documents)
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=1024,
    )
    elapsed = time.monotonic() - start

    usage = response.usage
    return NaiveResult(
        answer=response.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        latency_seconds=elapsed,
        model=getattr(response, "model", model),
    )
