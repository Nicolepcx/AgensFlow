"""
The `run()` entry point.

Composes everything: load env, instantiate the OpenRouter client, derive the
activation plan, build agent factories, build the graph, invoke it, and
return a structured RunResult.

This is the function users will call. Keep its signature stable; everything
else is implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from agensflow.activation.planner import make_activation_plan
from agensflow.learning.policy_graph import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RELIABILITY_WEIGHT,
    PolicyGraph,
)
from agensflow.learning.reward import RewardInputs, compute_reward
from agensflow.learning.signature import Signature, belief_signature
from agensflow.runtime.agents import (
    make_evaluator,
    make_memory,
    make_planner,
    make_solver,
    make_verifier,
)
from agensflow.runtime.client import OpenRouterClient
from agensflow.runtime.graph import build_graph, build_learning_graph
from agensflow.runtime.trace import TraceCollector, TraceEvent
from agensflow.runtime.types import Document
from agensflow.schema import ActivationPlan, Handoff, RegimeEstimate, TaskFeatures


@dataclass
class RunResult:
    """
    Result of one end-to-end run.

    Carries the activation plan that was used (for inspection), the final
    Handoff state, the in-memory trace of every agent call, the final
    user-facing answer extracted from the evaluator, and a `done` flag.

    When a policy_graph was passed to `run()`, the post-run fields
    `policy_path`, `reward`, and `policy_graph_size` summarise what the
    learning step did.
    """

    plan: ActivationPlan
    final_state: Handoff
    trace: TraceCollector
    final_answer: str
    done: bool
    evaluator_reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Layer 1 learning artifacts (None when no policy_graph was passed).
    policy_path: list[tuple[Signature, str]] | None = None
    reward: float | None = None
    policy_graph_size: int | None = None
    # Governance state (None when no policy was passed). Populated even on
    # successful runs so the harness can attach the per-agent activity
    # counts and any non-fatal violations to the RunReport.
    governance_state: Any | None = None

    @property
    def total_tokens(self) -> int:
        return self.trace.total_tokens

    @property
    def total_latency_seconds(self) -> float:
        return self.trace.total_latency_seconds

    @property
    def agent_call_sequence(self) -> list[str]:
        return self.trace.agent_call_sequence


def _reconstruct_policy_path(
    trace_events: list[TraceEvent],
    regime_label: str,
) -> list[tuple[Signature, str]]:
    """
    Replay the trace into a list of (signature, action) pairs.

    Each successful agent call contributes one entry: the signature of the
    state *before* the call (taken from the event's input_state snapshot)
    paired with the action (agent name) that was taken from there.

    Failed validation attempts are skipped — they are visible in the trace
    for cost accounting, but they don't represent orchestration choices, so
    they don't contribute to the policy graph's per-action statistics.
    """
    path: list[tuple[Signature, str]] = []
    for event in trace_events:
        if event.error is not None:
            continue
        try:
            input_handoff = Handoff.model_validate(event.input_state)
        except Exception:  # noqa: BLE001 — defensive against schema drift in old traces
            continue
        sig = belief_signature(input_handoff, regime_label)  # type: ignore[arg-type]
        path.append((sig, event.agent))
    return path


def _record_failures_to_graph(
    *,
    policy_graph: PolicyGraph,
    trace_events: list[TraceEvent],
    regime_label: str,
) -> int:
    """
    Iterate trace events with `error` set and tally each as a per-edge failure.

    These are typically Instructor validation retries that fired at a specific
    `(input_state, agent)` edge. Even when the retry recovers and the
    downstream reward looks fine, we want UCB's reliability term to know
    this edge tripped — that is the substrate for the framework's "models
    that are unreliable for your domain get downweighted over time" story.

    Returns the number of failures recorded (for trace/debug visibility).
    """
    n_recorded = 0
    for event in trace_events:
        if event.error is None:
            continue
        try:
            input_handoff = Handoff.model_validate(event.input_state)
        except Exception:  # noqa: BLE001 — defensive against schema drift
            continue
        sig = belief_signature(input_handoff, regime_label)  # type: ignore[arg-type]
        policy_graph.record_failure(sig, event.agent)
        n_recorded += 1
    return n_recorded


def _backup_to_policy_graph(
    *,
    policy_graph: PolicyGraph,
    trace: TraceCollector,
    regime_label: str,
    final_state: Handoff,
    done: bool,
) -> tuple[list[tuple[Signature, str]], float]:
    """
    Reconstruct the run's (signature, action) path, compute its reward, and
    backpropagate the reward through the policy graph.

    Returns the path and the reward so the caller can expose them on the
    RunResult for inspection and demos.
    """
    path = _reconstruct_policy_path(trace.events, regime_label)

    # Record edges (so the graph remembers the topology of what was tried).
    for i in range(len(path) - 1):
        from_sig, action = path[i]
        to_sig = path[i + 1][0]
        policy_graph.record_transition(from_sig, action, to_sig)

    # Tally per-edge validation failures so UCB can downweight unreliable
    # edges even when the recovered retry produced a normal reward.
    _record_failures_to_graph(
        policy_graph=policy_graph,
        trace_events=trace.events,
        regime_label=regime_label,
    )

    # Compute reward.
    n_validation_retries = sum(1 for e in trace.events if e.error is not None)
    reward_inputs = RewardInputs(
        done=done,
        verification_str=final_state.verification,
        total_tokens=trace.total_tokens,
        n_validation_retries=n_validation_retries,
    )
    reward = compute_reward(reward_inputs)

    # Backup reward through the visited (signature, action) pairs.
    # Chunk-9: feed per-action tokens into Welford variance tracking
    # for the (skill, model) reliability/cost-stability story.
    action_tokens: dict[str, int] = {}
    for ev in trace.events:
        if ev.error is not None:
            continue
        action_tokens[ev.agent] = action_tokens.get(ev.agent, 0) + ev.total_tokens
    policy_graph.backup(path, reward, action_tokens=action_tokens)

    return path, reward


def _build_node_table(
    selected_skills: list[str],
    *,
    client: OpenRouterClient,
    user_task: str,
    documents: list[Document],
    trace: TraceCollector,
    model_overrides: dict[str, str] | None,
) -> dict[str, Any]:
    """
    Instantiate node functions for exactly the skills the plan selected.

    Handles two cases:
      - Base skills (planner, memory, solver, verifier, evaluator) use their
        own factory directly.
      - Variant skills (solver_fast, solver_capable, verifier_fast, ...)
        delegate to the base factory but pass through the variant name. The
        factory then resolves the variant's model via SKILL_VARIANT_BINDINGS
        and the trace records the variant name for per-variant value backup.
    """
    from agensflow.runtime.models import get_base_skill, is_variant
    from agensflow.runtime.web_search import (
        make_web_search_exa,
        make_web_search_tavily,
    )

    base_factories = {
        "planner": lambda name: make_planner(client, user_task, trace, model_overrides),
        "memory": lambda name: make_memory(client, documents, trace, model_overrides),
        "solver": lambda name: make_solver(
            client, trace, model_overrides, skill_name=name
        ),
        "verifier": lambda name: make_verifier(
            client, trace, model_overrides, skill_name=name
        ),
        "evaluator": lambda name: make_evaluator(client, trace, model_overrides),
        # Tool-as-skill factories. Web search providers don't take a client
        # because they call their own provider APIs directly.
        "web_search_exa": lambda name: make_web_search_exa(trace),
        "web_search_tavily": lambda name: make_web_search_tavily(trace),
    }

    nodes: dict[str, Any] = {}
    for skill in selected_skills:
        if skill in base_factories:
            base = skill
        elif is_variant(skill):
            base = get_base_skill(skill)
        else:
            raise NotImplementedError(
                f"Skill {skill!r} has no runtime factory. "
                f"Implemented skills: {sorted(base_factories.keys())} "
                f"plus variants of solver/verifier."
            )
        if base not in base_factories:
            raise NotImplementedError(
                f"Skill {skill!r} (base={base!r}) has no runtime factory. "
                f"Implemented base skills: {sorted(base_factories.keys())}."
            )
        nodes[skill] = base_factories[base](skill)
    return nodes


def run(
    user_task: str,
    *,
    features: TaskFeatures,
    documents: list[Document] | None = None,
    regime: RegimeEstimate | None = None,
    client: OpenRouterClient | None = None,
    model_overrides: dict[str, str] | None = None,
    load_env: bool = True,
    policy_graph: PolicyGraph | None = None,
    max_steps: int = 12,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    reliability_weight: float = DEFAULT_RELIABILITY_WEIGHT,
    enable_skip: bool = False,
    enable_router_logging: bool = False,
    defer_backup: bool = False,
    plan: ActivationPlan | None = None,
    trace: TraceCollector | None = None,
    governance_policy: "Any | None" = None,
    governance_state: "Any | None" = None,
) -> RunResult:
    """
    Execute one end-to-end run.

    Arguments:
      user_task: the user's natural-language task.
      features:  TaskFeatures describing the task. Drives regime detection.
      documents: documents the memory agent can retrieve from. Required when
                 the plan selects the memory skill (most regimes).
      regime:    optional pre-computed RegimeEstimate. When None, the default
                 rule-based regime detector is invoked from features.
      client:    optional pre-built OpenRouterClient. When None, a default
                 client is constructed (reads OPENROUTER_API_KEY from env).
      model_overrides: optional mapping from skill name to model id, overriding
                 DEFAULT_MODEL_ASSIGNMENT for this run.
      load_env:  whether to call load_dotenv() before constructing the client.
                 Set False if you've already loaded env vars yourself.
      policy_graph: when provided, the runtime uses **policy-driven dynamic
                 routing** (chunk 4.5): a router node consults the graph at
                 each step, taking the UCB-best action when the current
                 signature has accumulated enough visits, falling back to
                 the rule-based prior otherwise. After the run completes,
                 the (signature, action) path is reconstructed, a reward
                 is computed, and the reward is backpropagated through the
                 graph. Subsequent runs sharing this graph see the
                 accumulated value estimates and may make different routing
                 choices.

                 When None, the runtime uses linear rule-based routing
                 (backward-compatible chunks 2/3 path). The dynamic-routing
                 path is the framework's intended runtime; the linear path
                 is preserved for tests and for the "no learning" baseline.
      max_steps: budget cap on routing decisions per run (default 12).
                 Prevents runaway loops if the policy makes pathological
                 choices. Only applied when `policy_graph` is provided.
      confidence_threshold: minimum visits to a signature before the policy
                 graph's recommendation is trusted over the rule-based prior
                 (default 3). Higher values make the policy more conservative.

    Returns: a RunResult with the final Handoff, the trace, and the answer.
             When `policy_graph` was provided, also includes `policy_path`,
             `reward`, and `policy_graph_size`.

    Limitations in chunk 4.5:
      - Only linear (non-branching) activation plans are supported. The
        router operates within `plan.selected_skills`.
      - Each skill is callable at most once per run; the router optimises
        the *order* of the coalition, not how many times each agent is
        invoked.
      - Critic and synthesizer agents are not yet implemented.
    """
    if load_env:
        load_dotenv()

    # Custom plan overrides the rule-based default. Chunk 6 uses this to pass
    # the variant-pool activation plan; default behavior (plan=None) builds
    # the rule-based plan from features+regime as before.
    if plan is None:
        plan = make_activation_plan(features, regime=regime)

    # Memory is allowed to run with an empty document set — it will simply
    # return empty `evidence` and `retrieved_context` lists, which the
    # downstream pipeline handles. This intentionally permits the
    # "force-the-wrong-regime" experimental scenario where evidence_heavy
    # is applied to a no-document task.

    client = client or OpenRouterClient()
    # Use the caller-supplied trace if given (lets the harness keep a
    # reference and dump the router-iteration log even if the run raises).
    # Otherwise construct a fresh one.
    if trace is None:
        trace = TraceCollector()

    # Wire governance into the trace if a policy or pre-built state was
    # passed. The state accumulates per-agent activity across this run;
    # if any constraint is breached, BrokenAgentError propagates up
    # through the next trace.record() and is caught by the harness for
    # skip-backup handling.
    if governance_state is None and governance_policy is not None:
        from agensflow.runtime.governance import GovernanceState
        governance_state = GovernanceState(policy=governance_policy)
    if governance_state is not None:
        from agensflow.runtime.governance import bind_governance_to_trace
        bind_governance_to_trace(trace, governance_state)

    nodes = _build_node_table(
        plan.selected_skills,
        client=client,
        user_task=user_task,
        documents=documents or [],
        trace=trace,
        model_overrides=model_overrides,
    )

    if policy_graph is not None:
        compiled = build_learning_graph(
            plan,
            nodes,
            policy_graph=policy_graph,
            trace=trace,
            max_steps=max_steps,
            confidence_threshold=confidence_threshold,
            reliability_weight=reliability_weight,
            enable_skip=enable_skip,
            enable_router_logging=enable_router_logging,
        )
    else:
        compiled = build_graph(plan, nodes)
    initial_state = Handoff()

    raw_final = compiled.invoke(initial_state)

    # LangGraph returns a dict in current versions; revalidate as a Handoff so
    # downstream code can rely on the typed surface.
    final_state = (
        raw_final
        if isinstance(raw_final, Handoff)
        else Handoff.model_validate(raw_final)
    )

    evaluator_output = final_state.metadata.get("evaluator", {})
    final_answer = evaluator_output.get("final_answer", "") or (
        final_state.draft_answer or ""
    )
    done = bool(evaluator_output.get("done", False))
    reasoning = evaluator_output.get("reasoning", "")

    # Layer 1 learning: if a policy graph was provided, reconstruct the
    # (signature, action) path from the trace. By default also compute the
    # v1 symbolic reward and backup. When `defer_backup=True`, only
    # reconstruct the path — the caller takes responsibility for computing
    # a reward (e.g. via RelativeJudge) and calling policy_graph.backup() manually.
    # This is the integration point for chunk 6's hybrid reward.
    policy_path: list[tuple[Signature, str]] | None = None
    reward: float | None = None
    policy_graph_size: int | None = None
    if policy_graph is not None:
        if defer_backup:
            policy_path = _reconstruct_policy_path(
                trace.events, plan.regime.label
            )
            policy_graph_size = len(policy_graph)
        else:
            policy_path, reward = _backup_to_policy_graph(
                policy_graph=policy_graph,
                trace=trace,
                regime_label=plan.regime.label,
                final_state=final_state,
                done=done,
            )
            policy_graph_size = len(policy_graph)

    return RunResult(
        plan=plan,
        final_state=final_state,
        trace=trace,
        final_answer=final_answer,
        done=done,
        evaluator_reasoning=reasoning,
        policy_path=policy_path,
        reward=reward,
        policy_graph_size=policy_graph_size,
        governance_state=governance_state,
    )
