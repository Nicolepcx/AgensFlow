"""
Agent factories.

Each factory returns a node function suitable for use as a LangGraph node:
it takes the current Handoff (state) and returns a dict of field updates.

Agents are factories rather than plain functions so the OpenRouter client,
trace collector, document set, and per-skill model overrides can be captured
via closure. This keeps the LangGraph node signature clean (state -> dict)
while still threading runtime context through.

The validation+retry mechanic is delegated to OpenRouterClient (Instructor
under the hood). Each agent calls `client.complete_typed(response_model=...)`
and receives a typed Pydantic object back. Failed validation attempts are
recorded automatically by the client's hooks; the agent records the trace
event for the successful attempt itself, with the full output_update.

The framework's bounded-retry discipline (one corrective retry, not a stack)
is enforced by passing max_retries=2 to complete_typed (initial + at most one
corrective). The discipline lives in the policy, the mechanism lives in the
client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agensflow.runtime.agent_outputs import (
    EvaluatorOutput,
    MemoryOutput,
    PlannerOutput,
    SolverOutput,
    VerifierOutput,
)
from agensflow.runtime.client import CompletionResult, OpenRouterClient
from agensflow.runtime.models import get_model_for_skill
from agensflow.runtime.prompts import (
    EVALUATOR_SYSTEM,
    EVALUATOR_USER_TEMPLATE,
    MEMORY_SYSTEM,
    MEMORY_USER_TEMPLATE,
    PLANNER_SYSTEM,
    PLANNER_USER_TEMPLATE,
    SOLVER_SYSTEM,
    SOLVER_USER_TEMPLATE,
    VERIFIER_SYSTEM,
    VERIFIER_USER_TEMPLATE,
)
from agensflow.runtime.trace import TraceCollector, TraceEvent
from agensflow.runtime.types import Document
from agensflow.schema import Handoff

NodeFn = Callable[[Handoff], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _accumulate_refs(
    existing: dict[str, list[str]],
    additions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge upstream_refs without losing prior entries."""
    out = dict(existing)
    for k, v in additions.items():
        out[k] = list(v)
    return out


def _format_block(items: list[str], bullet: str = "- ") -> str:
    if not items:
        return "(none)"
    return "\n".join(f"{bullet}{x}" for x in items)


def _format_documents(docs: list[Document]) -> str:
    if not docs:
        return "(no documents provided)"
    return "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)


def _record_success(
    *,
    trace: TraceCollector,
    agent: str,
    state_snapshot: dict[str, Any],
    output_update: dict[str, Any],
    result: CompletionResult,
) -> None:
    """Record the trace event for a successful agent call."""
    trace.record(
        TraceEvent(
            agent=agent,
            model=result.model,
            input_state=state_snapshot,
            output_update=output_update,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_seconds=result.latency_seconds,
        )
    )


def _with_belief_update(
    state: Handoff,
    update: dict[str, Any],
    agent: str,
) -> dict[str, Any]:
    """
    Add a `belief` entry to the agent's update dict, computed by projecting
    the existing update onto the state and running the belief updater on the
    resulting Handoff. The belief delta corresponds to *what this specific
    agent contributed* (planner, memory, solver, critic, verifier,
    synthesizer, evaluator); the rules live in `agensflow.learning.belief`.

    Importing the updater here (not at module top) avoids a runtime ↔ learning
    circular import and keeps the runtime functional even if the learning
    package were absent.
    """
    from agensflow.learning.belief import update_belief

    projected = state.model_copy(update=update)
    new_belief = update_belief(state.belief, agent=agent, handoff=projected)
    update["belief"] = new_belief
    return update


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def make_planner(
    client: OpenRouterClient,
    user_task: str,
    trace: TraceCollector,
    model_override: dict[str, str] | None = None,
) -> NodeFn:
    model = get_model_for_skill("planner", model_override)

    def planner_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        result = client.complete_typed(
            model=model,
            system_prompt=PLANNER_SYSTEM,
            user_prompt=PLANNER_USER_TEMPLATE.format(user_task=user_task),
            output_model=PlannerOutput,
            agent_name="planner",
            trace=trace,
            state_snapshot=snapshot,
        )
        parsed: PlannerOutput = result.parsed_output  # type: ignore[assignment]

        update: dict[str, Any] = {
            "goal": parsed.goal,
            "subproblem": parsed.subproblem,
            "constraints": list(parsed.constraints),
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {
                    "goal": ["user_task"],
                    "subproblem": ["user_task"],
                    "constraints": ["user_task"],
                },
            ),
        }
        update = _with_belief_update(state, update, "planner")
        _record_success(
            trace=trace,
            agent="planner",
            state_snapshot=snapshot,
            output_update=update,
            result=result,
        )
        return update

    return planner_node


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def make_memory(
    client: OpenRouterClient,
    documents: list[Document],
    trace: TraceCollector,
    model_override: dict[str, str] | None = None,
) -> NodeFn:
    """
    Memory agent backed by a fixed document set.

    For the demo, the memory agent does retrieval against documents provided
    at run() time. Production deployments plug in a vector store, BM25 index,
    or whatever retrieval system fits. The contract — "produce evidence
    grounded in retrievable sources" — is what matters, not the mechanism.
    """
    model = get_model_for_skill("memory", model_override)

    def memory_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        result = client.complete_typed(
            model=model,
            system_prompt=MEMORY_SYSTEM,
            user_prompt=MEMORY_USER_TEMPLATE.format(
                goal=state.goal or "(unset)",
                subproblem=state.subproblem or "(unset)",
                documents_block=_format_documents(documents),
            ),
            output_model=MemoryOutput,
            agent_name="memory",
            trace=trace,
            state_snapshot=snapshot,
        )
        parsed: MemoryOutput = result.parsed_output  # type: ignore[assignment]

        update: dict[str, Any] = {
            "evidence": list(parsed.evidence),
            "retrieved_context": list(parsed.retrieved_context),
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {
                    "evidence": ["subproblem", "available_documents"],
                    "retrieved_context": ["available_documents"],
                },
            ),
        }
        update = _with_belief_update(state, update, "memory")
        _record_success(
            trace=trace,
            agent="memory",
            state_snapshot=snapshot,
            output_update=update,
            result=result,
        )
        return update

    return memory_node


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def make_solver(
    client: OpenRouterClient,
    trace: TraceCollector,
    model_override: dict[str, str] | None = None,
    skill_name: str = "solver",
) -> NodeFn:
    """
    Build a solver node bound to a specific model.

    `skill_name` allows the same factory to back model variants (solver_fast,
    solver_capable, etc.). The model is resolved from skill_name via
    `get_model_for_skill`, which checks SKILL_VARIANT_BINDINGS first. The
    trace records the actual variant name so per-variant value backup is
    correctly attributed at the policy graph level.

    Chunk-9: when `skill_name` is mapped to a registered SkillCard via
    `runtime.models.SKILL_VARIANT_CARDS`, the card's `instructions` are
    used as the system prompt instead of the hardcoded `SOLVER_SYSTEM`.
    This is what makes (skill_card × model) the unit of action-space
    variation — same skill name → same model → different system prompt
    if a different card is bound. Backward-compatible: variants without
    a card binding (e.g. `solver_fast` from chunk 8) use SOLVER_SYSTEM.
    """
    from agensflow.registry import default_registry
    from agensflow.runtime.models import get_card_for_skill

    model = get_model_for_skill(skill_name, model_override)

    # Resolve the system prompt: SkillCard if a card is bound for this
    # variant *and* registered, otherwise fall back to SOLVER_SYSTEM.
    card_name = get_card_for_skill(skill_name)
    if card_name is not None and default_registry.has_card(card_name):
        system_prompt = default_registry.get_card(card_name).instructions
    else:
        system_prompt = SOLVER_SYSTEM

    def solver_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        result = client.complete_typed(
            model=model,
            system_prompt=system_prompt,
            user_prompt=SOLVER_USER_TEMPLATE.format(
                subproblem=state.subproblem or "(unset)",
                constraints_block=_format_block(state.constraints),
                evidence_block=_format_block(state.evidence),
            ),
            output_model=SolverOutput,
            agent_name=skill_name,
            trace=trace,
            state_snapshot=snapshot,
        )
        parsed: SolverOutput = result.parsed_output  # type: ignore[assignment]

        update: dict[str, Any] = {
            "draft_answer": parsed.draft_answer,
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {"draft_answer": ["subproblem", "constraints", "evidence"]},
            ),
        }
        update = _with_belief_update(state, update, "solver")
        _record_success(
            trace=trace,
            agent=skill_name,
            state_snapshot=snapshot,
            output_update=update,
            result=result,
        )
        return update

    return solver_node


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


def make_verifier(
    client: OpenRouterClient,
    trace: TraceCollector,
    model_override: dict[str, str] | None = None,
    skill_name: str = "verifier",
) -> NodeFn:
    """
    Build a verifier node bound to a specific model.

    `skill_name` allows the same factory to back model variants
    (verifier_fast, verifier_capable). See `make_solver` for the pattern.
    """
    model = get_model_for_skill(skill_name, model_override)

    def verifier_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        result = client.complete_typed(
            model=model,
            system_prompt=VERIFIER_SYSTEM,
            user_prompt=VERIFIER_USER_TEMPLATE.format(
                subproblem=state.subproblem or "(unset)",
                constraints_block=_format_block(state.constraints),
                evidence_block=_format_block(state.evidence),
                draft_answer=state.draft_answer or "(unset)",
            ),
            output_model=VerifierOutput,
            agent_name=skill_name,
            trace=trace,
            state_snapshot=snapshot,
        )
        parsed: VerifierOutput = result.parsed_output  # type: ignore[assignment]

        verification_str = parsed.model_dump_json()

        update: dict[str, Any] = {
            "verification": verification_str,
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {"verification": ["draft_answer", "evidence", "constraints"]},
            ),
        }
        update = _with_belief_update(state, update, "verifier")
        _record_success(
            trace=trace,
            agent=skill_name,
            state_snapshot=snapshot,
            output_update=update,
            result=result,
        )
        return update

    return verifier_node


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


def make_evaluator(
    client: OpenRouterClient,
    trace: TraceCollector,
    model_override: dict[str, str] | None = None,
) -> NodeFn:
    model = get_model_for_skill("evaluator", model_override)

    def evaluator_node(state: Handoff) -> dict[str, Any]:
        snapshot = state.model_dump()
        result = client.complete_typed(
            model=model,
            system_prompt=EVALUATOR_SYSTEM,
            user_prompt=EVALUATOR_USER_TEMPLATE.format(
                goal=state.goal or "(unset)",
                subproblem=state.subproblem or "(unset)",
                draft_answer=state.draft_answer or "(unset)",
                verification=state.verification or "(unset)",
            ),
            output_model=EvaluatorOutput,
            agent_name="evaluator",
            trace=trace,
            state_snapshot=snapshot,
        )
        parsed: EvaluatorOutput = result.parsed_output  # type: ignore[assignment]

        # Evaluator's structured output goes into metadata so the schema stays
        # narrow. The final answer is exposed both via metadata and via the
        # RunResult.
        metadata = dict(state.metadata)
        metadata["evaluator"] = {
            "done": parsed.done,
            "final_answer": parsed.final_answer,
            "reasoning": parsed.reasoning,
        }

        update: dict[str, Any] = {
            "metadata": metadata,
            "upstream_refs": _accumulate_refs(
                state.upstream_refs,
                {"metadata.evaluator": ["draft_answer", "verification", "goal"]},
            ),
        }
        update = _with_belief_update(state, update, "evaluator")
        _record_success(
            trace=trace,
            agent="evaluator",
            state_snapshot=snapshot,
            output_update=update,
            result=result,
        )
        return update

    return evaluator_node


# --------------------------------------------------------------------------- #
# Registry of agent factories — used by the graph builder.
# --------------------------------------------------------------------------- #


AGENT_FACTORIES: dict[str, str] = {
    "planner": "make_planner",
    "memory": "make_memory",
    "solver": "make_solver",
    "verifier": "make_verifier",
    "evaluator": "make_evaluator",
}
"""
Mapping from skill name to factory function name. Used for documentation and
for the graph builder to validate that every selected skill has a factory.

Critic and synthesizer factories are intentionally not implemented in chunk 2
because the evidence_heavy demo regime does not use them. They will be added
in subsequent chunks alongside the regimes that need them (ambiguous,
contradictory, high_risk).
"""
