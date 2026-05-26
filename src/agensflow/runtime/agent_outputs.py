"""
Per-agent output schemas.

Each agent returns JSON that maps to fields on the Handoff. We validate the
LLM's output against a strict Pydantic model BEFORE writing to the Handoff,
so a malformed response fails loudly at the validation boundary instead of
silently corrupting downstream state.

This is the framework's thesis applied to itself: structured handoffs all the
way down, validated at every edge, not free-form text trusted on faith.

The output schemas are intentionally separate from the Handoff schema. The
Handoff is the *shared state*; these are the *contracts* each agent fulfils
when it speaks. They map cleanly into Handoff fields but they are not the
same objects, because:
  - An agent only writes part of the Handoff per call.
  - An agent's output may need fields the Handoff doesn't carry (e.g., the
    verifier's structured verdict before it gets serialised into the
    Handoff.verification string).
  - The contracts evolve independently of the shared state schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class PlannerOutput(BaseModel):
    """What the planner agent must return."""

    goal: str = Field(min_length=1, description="One-sentence restatement of success.")
    subproblem: str = Field(
        min_length=1, description="The specific question the next agent should tackle."
    )
    constraints: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


class MemoryOutput(BaseModel):
    """What the memory agent must return."""

    evidence: list[str] = Field(
        default_factory=list,
        description="Factual statements grounded in retrieved_context documents.",
    )
    retrieved_context: list[str] = Field(
        default_factory=list,
        description="Document ids whose content supports the evidence entries.",
    )

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class SolverOutput(BaseModel):
    """What the solver agent must return."""

    draft_answer: str = Field(min_length=1)

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

VerifierVerdict = Literal["supported", "partially_supported", "unsupported"]


class VerifierOutput(BaseModel):
    """
    What the verifier agent must return.

    The verdict is a small enumeration so downstream code (and metrics) can
    branch on it without parsing free text. The schema is intentionally
    structured even though Handoff.verification is currently a string field —
    we serialise this object as JSON into that field, and downstream agents
    parse it back. The schema will eventually be promoted into the Handoff.
    """

    verdict: VerifierVerdict
    rationale: str = Field(min_length=1)
    uncertain_claims: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class EvaluatorOutput(BaseModel):
    """What the evaluator agent must return."""

    done: bool
    final_answer: str = Field(min_length=1)
    reasoning: str = ""

    model_config = {"extra": "ignore"}
