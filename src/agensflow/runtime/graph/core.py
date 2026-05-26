"""
LangGraph integration.

Two builders:

  - `build_graph` — linear, rule-based execution. Edges go agent₁ → agent₂ →
    ... → END in plan order. Backward-compatible with chunks 2/3; used when
    no policy graph is provided.

  - `build_learning_graph` — dynamic, policy-driven execution. A single
    router node consults the policy graph at each step and emits
    `Command(goto=next_agent)`. The router is the place where AgensFlow's
    distinguishing claim — *learnable orchestration policy* — becomes
    actionable.

The dynamic builder is the framework's intended runtime path. The linear
builder is preserved for the no-learning case and for backward compat with
existing tests/examples that don't pass a policy graph.

Branching runtime (parallel coalitions + merge strategies) is a separate
later chunk and not implemented in either builder yet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from agensflow.learning.policy_graph import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RELIABILITY_WEIGHT,
    PolicyGraph,
)
from agensflow.learning.router import select_next_action
from agensflow.runtime.trace import TraceCollector, TraceEvent
from agensflow.schema import ActivationPlan, Handoff

NodeFn = Callable[[Handoff], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Linear builder (chunks 2/3 — backward-compat path).
# --------------------------------------------------------------------------- #


def build_graph(
    plan: ActivationPlan,
    nodes: dict[str, NodeFn],
) -> Any:  # langgraph.graph.state.CompiledStateGraph
    """
    Build and compile a *linear* LangGraph from an ActivationPlan.

    Edges are wired in the plan's `selected_skills` order. No policy
    consultation, no dynamic routing. This is the backward-compatible builder
    used when `run()` is called without a `policy_graph`.

    Branching plans are rejected with NotImplementedError — branching runtime
    is a later chunk regardless of which builder is used.
    """
    if plan.branch_rule.enabled:
        raise NotImplementedError(
            "Branching activation plans are not yet supported by the runtime. "
            "Use a non-branching regime (straightforward, evidence_heavy) for now."
        )

    selected = list(plan.selected_skills)
    if not selected:
        raise ValueError("ActivationPlan has no selected_skills; nothing to build.")

    missing = [s for s in selected if s not in nodes]
    if missing:
        raise KeyError(
            f"No node function provided for skill(s): {missing}. "
            f"Provided nodes: {sorted(nodes.keys())}."
        )

    graph: StateGraph = StateGraph(Handoff)

    for skill in selected:
        graph.add_node(skill, nodes[skill])

    graph.set_entry_point(selected[0])
    for src, dst in zip(selected, selected[1:]):
        graph.add_edge(src, dst)
    graph.add_edge(selected[-1], END)

    return graph.compile()


# --------------------------------------------------------------------------- #
# Dynamic-routing builder
# --------------------------------------------------------------------------- #


# Name of the synthetic router node. Kept as a constant so the runner and
# the trace can refer to it consistently.
ROUTER_NODE_NAME = "router"


def build_learning_graph(
    plan: ActivationPlan,
    nodes: dict[str, NodeFn],
    *,
    policy_graph: PolicyGraph,
    trace: TraceCollector,
    max_steps: int = 12,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    reliability_weight: float = DEFAULT_RELIABILITY_WEIGHT,
    enable_skip: bool = False,
    enable_router_logging: bool = False,
) -> Any:
    """
    Build a LangGraph with policy-driven dynamic routing.

    Topology:

        START → router → <agent picked by router> → router → ... → END

    The router is a single node that:
      - Reads the current state's belief signature.
      - Computes legal actions (plan-allowed, preconditions met, not yet taken).
      - Consults the policy graph at this signature for a confident
        recommendation; falls back to the rule-based prior if the graph isn't
        confident.
      - Emits `Command(goto=...)` to dispatch to the chosen agent, or
        `Command(goto=END)` if a termination condition fires.

    Each agent node finishes by returning a normal dict update (LangGraph's
    standard contract). Control then flows back to the router via a static
    edge — the agent doesn't itself decide where to go.

    `actions_taken` is read from the trace collector at each routing
    decision: it's the list of successful agent calls so far this run
    (failed validation retries are excluded from routing-decision history,
    though they remain in the trace for cost accounting).
    """
    if plan.branch_rule.enabled:
        raise NotImplementedError(
            "Branching activation plans are not yet supported by the runtime."
        )

    selected = list(plan.selected_skills)
    if not selected:
        raise ValueError("ActivationPlan has no selected_skills; nothing to build.")

    missing = [s for s in selected if s not in nodes]
    if missing:
        raise KeyError(
            f"No node function provided for skill(s): {missing}. "
            f"Provided nodes: {sorted(nodes.keys())}."
        )

    # Counter for router-node entries — each new entry is a fresh trip
    # through the while-True loop after an agent invocation returned
    # control. Used by router-logging to give each iteration a unique
    # `(entry, sub_iter)` coordinate so the forensic log has structure.
    _router_entries = [0]

    def router_node(state: Handoff) -> Command:
        # Inline skip-loop: when chunk-8's `enable_skip` is on, the router
        # may choose `skip:X` actions, which don't dispatch to an agent
        # node — they record a synthetic trace event and re-route. The
        # loop here keeps the LangGraph topology unchanged (router still
        # only emits Command(goto=<real-skill>) or Command(goto=END))
        # while letting the policy compose chains of skips inline.
        _router_entries[0] += 1
        entry_idx = _router_entries[0]
        sub_iter = 0
        while True:
            sub_iter += 1
            # Deduplicate by agent name. An action is "taken" if it has
            # ANY trace event (success or error) for that agent. This
            # was originally `[e.agent for e in trace.events if e.error
            # is None]`, which filtered out tool-agent failure events
            # and caused the chunk-9 epoch-8 recursion-loop bug: when
            # web_search_exa's API failed, the failure event was filtered
            # out, the router thought web_search_exa hadn't been called,
            # picked it again, and looped until LangGraph's recursion
            # ceiling fired.
            #
            # The fix: any event for agent X means X was attempted.
            # Counts toward actions_taken regardless of error status.
            # Mechanism A+C still tracks per-edge failure rates via the
            # error field separately — so tool-failure data is preserved
            # for the substrate's reliability learning, the router just
            # doesn't infinite-loop on it.
            actions_taken = list(dict.fromkeys(
                e.agent for e in trace.events
            ))
            # Compute legal_actions BEFORE select_next_action so we can
            # log it for diagnostics (select_next_action recomputes it
            # internally; this duplication is cheap and only happens
            # when enable_router_logging is on).
            from agensflow.learning.router import (
                _legal_actions, SKIP_PREFIX,
            )
            if enable_router_logging:
                legal_for_log = _legal_actions(plan, state, actions_taken)
                candidates_for_log = list(legal_for_log)
                if enable_skip and len(legal_for_log) > 1:
                    candidates_for_log += [
                        f"{SKIP_PREFIX}{x}" for x in legal_for_log
                    ]
            decision = select_next_action(
                current_state=state,
                plan=plan,
                policy_graph=policy_graph,
                actions_taken=actions_taken,
                max_steps=max_steps,
                confidence_threshold=confidence_threshold,
                reliability_weight=reliability_weight,
                enable_skip=enable_skip,
            )

            if enable_router_logging:
                trace.record_router_iteration({
                    "entry": entry_idx,
                    "sub_iter": sub_iter,
                    "actions_taken_count": len(actions_taken),
                    "actions_taken": list(actions_taken),
                    "legal": legal_for_log,
                    "candidates": candidates_for_log,
                    "decision_action": decision.action,
                    "decision_reason": decision.reason,
                })

            if decision.action is None:
                return Command(goto=END)
            if decision.action.startswith("skip:"):
                # Record a zero-cost synthetic event so the policy graph
                # backup credits this signature/action like any other
                # routing decision. No agent invocation, no LLM call.
                trace.record(TraceEvent(
                    agent=decision.action,
                    model="(skip — no model invoked)",
                    input_state=(
                        state.model_dump()
                        if hasattr(state, "model_dump") else dict(state)
                    ),
                    output_update={},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_seconds=0.0,
                ))
                # Loop continues: actions_taken now reflects this skip,
                # legal_actions on the next iteration will exclude this
                # skill, and the policy picks again from the remaining
                # candidates.
                continue
            return Command(goto=decision.action)

    graph: StateGraph = StateGraph(Handoff)
    graph.add_node(ROUTER_NODE_NAME, router_node)

    for skill in selected:
        graph.add_node(skill, nodes[skill])
        # Every agent returns to the router for the next decision.
        graph.add_edge(skill, ROUTER_NODE_NAME)

    graph.set_entry_point(ROUTER_NODE_NAME)

    # Compile with a recursion_limit that accommodates the worst case.
    # Each "step" in the policy traversal triggers multiple LangGraph
    # internal accounting increments (router invocation + Command emission
    # + agent node + Instructor's per-attempt internal retries which
    # touch the LangGraph dispatcher). Empirically with the chunk-6
    # variant pool (10+ skills), C7-style multi-step ambiguous-regime
    # tasks need >80 limit; 8x max_steps + buffer covers observed worst
    # cases with margin.
    return graph.compile().with_config(
        recursion_limit=max(200, 12 * max_steps + 32)
    )
