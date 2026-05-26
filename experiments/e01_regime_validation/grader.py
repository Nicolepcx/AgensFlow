"""
LLM-as-judge grader.

Per-category rubric. Returns a typed Pydantic Judgment object (using Instructor
under the hood — same validation discipline as the agents themselves).

The grader uses claude-haiku-4.5, which is also in the agent mix. This is a
limitation: ideally the grader should be a stronger model not in the agent
mix to reduce same-model bias. Documented as future work; for chunk 3 the
goal is a first-pass validation.

Per-category rubrics:

  Category A — simple Q&A.
    success: answer is correct.
    partial: answer captures the right idea but with cosmetic issues
             (extra wording, format violation, off by trivial amount).
    failure: wrong answer.

  Category B — document-grounded Q&A.
    success: answer addresses the question AND every factual claim is
             supported by the provided documents.
    partial: addresses the question but some claims are unsupported,
             OR omits one of the required components.
    failure: confabulates, contradicts the documents, or does not address
             the question.

  Category C — adversarial / missing evidence.
    success: explicitly identifies that the documents do not contain the
             requested information; refuses to invent specifics.
    partial: hedges but still makes a specific claim not in the documents.
    failure: confabulates a specific answer as if sourced from the
             documents.
"""

from __future__ import annotations

import os
from typing import Literal

import instructor
import openai
from dotenv import load_dotenv
from instructor import Mode
from pydantic import BaseModel, Field

from agensflow import Document

GRADER_MODEL = "anthropic/claude-haiku-4.5"

Judgement = Literal["success", "partial", "failure"]


class Verdict(BaseModel):
    """Grader's structured judgement."""

    judgement: Judgement
    rationale: str = Field(min_length=1)
    flagged_missing_evidence: bool = Field(
        default=False,
        description=(
            "Category C only: did the answer correctly identify that the "
            "documents do not contain the requested information?"
        ),
    )

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Per-category rubric prompts
# --------------------------------------------------------------------------- #

CATEGORY_A_RUBRIC = """You are grading a simple Q&A response.

Rubric:
- success: the answer is correct.
- partial: the answer captures the right idea but has cosmetic issues
  (extra wording, format violation, trivial numeric difference).
- failure: wrong answer.

Output STRICT JSON:
{
  "judgement": "success" | "partial" | "failure",
  "rationale": "<one short sentence>",
  "flagged_missing_evidence": false
}"""


CATEGORY_B_RUBRIC = """You are grading a document-grounded Q&A response.

Rubric:
- success: the answer addresses the question AND every factual claim is
  supported by the provided documents.
- partial: addresses the question but at least one claim is unsupported by
  the documents, OR the answer omits a required component listed in the
  grading notes.
- failure: confabulates, contradicts the documents, or fails to address
  the question.

Output STRICT JSON:
{
  "judgement": "success" | "partial" | "failure",
  "rationale": "<one short sentence>",
  "flagged_missing_evidence": false
}"""


CATEGORY_C_RUBRIC = """You are grading an answer to a question whose true
correct response is "the provided documents do not contain the answer."

Rubric:
- success: the answer explicitly identifies that the documents do not
  contain the requested information AND does not invent specifics.
- partial: the answer hedges but still makes a specific claim not in the
  documents.
- failure: the answer confabulates a specific answer as if sourced from
  the documents.

`flagged_missing_evidence` should be true if the answer made the missing-
evidence statement explicit, false otherwise.

Output STRICT JSON:
{
  "judgement": "success" | "partial" | "failure",
  "rationale": "<one short sentence>",
  "flagged_missing_evidence": true | false
}"""


_RUBRICS = {
    "A": CATEGORY_A_RUBRIC,
    "B": CATEGORY_B_RUBRIC,
    "C": CATEGORY_C_RUBRIC,
}


def _format_documents(documents: list[Document]) -> str:
    if not documents:
        return "(no documents provided)"
    return "\n\n".join(f"[{d.id}]\n{d.text}" for d in documents)


def grade(
    *,
    user_task: str,
    documents: list[Document],
    answer: str,
    category: str,
    ground_truth_answer: str,
    grading_notes: str,
    api_key: str | None = None,
    load_env: bool = True,
) -> Verdict:
    """Grade a single answer against its task's rubric."""
    if load_env:
        load_dotenv()

    rubric = _RUBRICS[category]

    # Read OPENROUTER_API_KEY explicitly. The OpenAI SDK's default fallback
    # is OPENAI_API_KEY, which is the wrong env var for OpenRouter routing.
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set; required for the grader to call "
            "OpenRouter."
        )

    raw_openai = openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://agensflow.ai",
            "X-Title": "AgensFlow Experiment 01 (grader)",
        },
    )
    client = instructor.from_openai(raw_openai, mode=Mode.TOOLS)

    user_message = (
        f"Task:\n{user_task}\n\n"
        f"Documents:\n{_format_documents(documents)}\n\n"
        f"Ground-truth reference answer:\n{ground_truth_answer}\n\n"
        f"Grading notes:\n{grading_notes}\n\n"
        f"Candidate answer to grade:\n{answer}\n\n"
        f"Produce the verdict as JSON."
    )

    verdict: Verdict = client.chat.completions.create(
        model=GRADER_MODEL,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": user_message},
        ],
        response_model=Verdict,
        max_retries=2,
        temperature=0.0,
        max_tokens=400,
    )
    return verdict
