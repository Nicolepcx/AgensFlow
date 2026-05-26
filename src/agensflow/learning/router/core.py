"""
The routing decision function — Layer 1's "real feature."

`select_next_action` is the place where the policy-learning substrate becomes
actionable. At each step of a run, the router consults the policy graph at
the current belief signature; if the graph has accumulated enough evidence
to be confident, the UCB-best legal action wins. Otherwise, the rule-based
prior (the order in `plan.selected_skills`) decides.

The router is a pure function: state in → (action_or_None, reason) out. No
LLM calls, no I/O. This makes routing decisions fully testable with synthetic
graphs and synthetic states, and makes the routing logic re-usable wherever
"what should the next action be?" needs to be answered (LangGraph runtime
today, programmatic planning tomorrow, MCTS rollouts later).

Termination conditions, in priority order:
  1. Budget exhausted (more than `max_steps` actions have been taken).
  2. Evaluator has marked the run done (state.metadata["evaluator"]["done"]).
  3. No legal actions remain (every skill in the plan that has its
     preconditions met has already been called).

If any of these fire, the router returns (None, reason). Otherwise it returns
(action_name, reason) and the caller proceeds.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from agensflow.learning.policy_graph import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RELIABILITY_WEIGHT,
    UCB_C,
    PolicyGraph,
)
from agensflow.learning.signature import belief_signature
from agensflow.registry import default_registry
from agensflow.schema import ActivationPlan, Handoff, RegimeLabel

RoutingReason = Literal[
    "budget_exhausted",
    "evaluator_done",
    "no_legal_actions",
    "graph_recommendation",
    "rule_based_prior",
]


class RoutingDecision(NamedTuple):
    """One routing decision: the chosen action plus the reason."""

    action: str | None  # None means "terminate"
    reason: RoutingReason


# --------------------------------------------------------------------------- #
# Precondition checks
# --------------------------------------------------------------------------- #


def _handoff_has_field(handoff: Handoff, field_name: str) -> bool:
    """
    True if the handoff has meaningful content in `field_name`.

    Treats empty strings, empty lists, and empty dicts as "not set" so that
    preconditions like "needs subproblem" don't pass when the field is just
    a default empty value.
    """
    val = getattr(handoff, field_name, None)
    if val is None:
        return False
    if isinstance(val, str | list | dict | tuple) and len(val) == 0:
        return False
    return True


def _preconditions_met(handoff: Handoff, skill_name: str) -> bool:
    """
    Check whether `skill_name`'s declared preconditions are satisfied by the
    current handoff. Falls through gracefully for skills not in the registry.
    """
    try:
        spec = default_registry.get(skill_name)
    except KeyError:
        # Unknown skill: assume always legal so user-registered skills work.
        return True
    return all(_handoff_has_field(handoff, p) for p in spec.preconditions)


SKIP_PREFIX = "skip:"


def _legal_actions(
    plan: ActivationPlan,
    handoff: Handoff,
    actions_taken: list[str],
) -> list[str]:
    """
    Compute the set of legal next actions:

      - In `plan.selected_skills` (the activation plan defines the coalition).
      - Preconditions met against the current handoff.
      - Not already invoked in this run (skills are called at most once).
      - Not already explicitly skipped this run. When chunk-8's `enable_skip`
        is on, a `skip:X` action recorded in the trace excludes X from any
        subsequent routing step in the same run — the policy committed to
        not invoking it, so legality must respect that.

    Returning an *ordered* list preserves the rule-based prior (the activation
    plan's order) for downstream callers that fall back to it.
    """
    taken = {a for a in actions_taken if not a.startswith(SKIP_PREFIX)}
    skipped = {a[len(SKIP_PREFIX):] for a in actions_taken if a.startswith(SKIP_PREFIX)}
    out: list[str] = []
    for skill in plan.selected_skills:
        if skill in taken:
            continue
        if skill in skipped:
            continue
        if not _preconditions_met(handoff, skill):
            continue
        out.append(skill)
    return out


# --------------------------------------------------------------------------- #
# Termination detection
# --------------------------------------------------------------------------- #


def _evaluator_marked_done(handoff: Handoff) -> bool:
    """
    True if the evaluator already ran *and* set done=True on its metadata.

    The evaluator's structured output is stored under handoff.metadata["evaluator"]
    by `make_evaluator`. Defensive parsing keeps us safe against missing fields.
    """
    ev = handoff.metadata.get("evaluator") if handoff.metadata else None
    if not isinstance(ev, dict):
        return False
    return bool(ev.get("done", False))


# --------------------------------------------------------------------------- #
# The routing decision
# --------------------------------------------------------------------------- #


def select_next_action(
    *,
    current_state: Handoff,
    plan: ActivationPlan,
    policy_graph: PolicyGraph,
    actions_taken: list[str],
    regime_label: RegimeLabel | None = None,
    max_steps: int = 12,
    exploration_c: float = UCB_C,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    reliability_weight: float = DEFAULT_RELIABILITY_WEIGHT,
    enable_skip: bool = False,
) -> RoutingDecision:
    """
    Pick the next action, or terminate.

    Decision flow:

      1. If actions_taken already exceeds the budget, terminate
         ("budget_exhausted").
      2. If the evaluator has marked done, terminate ("evaluator_done").
      3. Compute legal actions. If none, terminate ("no_legal_actions").
      4. Compute the current belief signature.
      5. Ask the policy graph for a confident recommendation. If the node
         at this signature has at least `confidence_threshold` visits, take
         the UCB-best legal action ("graph_recommendation").
      6. Otherwise, fall back to the rule-based prior — the first legal
         action in plan order ("rule_based_prior").

    Notes:
      - The signature is computed from `current_state` (which carries the
        evolved belief from prior agents). The (signature, action) pair
        recorded in the policy graph during backup matches this signature
        exactly — this is what makes the learning substrate consistent
        between online routing and offline backup.
      - `regime_label` defaults to `plan.regime.label` if not supplied. The
        explicit override exists for tests and for future "regime-shifted-
        mid-run" scenarios that we don't support yet.
    """
    if len(actions_taken) >= max_steps:
        return RoutingDecision(None, "budget_exhausted")

    if _evaluator_marked_done(current_state):
        return RoutingDecision(None, "evaluator_done")

    legal = _legal_actions(plan, current_state, actions_taken)
    if not legal:
        return RoutingDecision(None, "no_legal_actions")

    label = regime_label if regime_label is not None else plan.regime.label
    signature = belief_signature(current_state, label)

    # When chunk-8's skip-mechanism is enabled, the action space the policy
    # chooses among expands to include a `skip:X` candidate for every legal
    # X. The substrate doesn't change — `skip:X` is just another action in
    # the per-(signature, action) value table. UCB picks the best, whether
    # invoke or skip. Unvisited skip actions get the +inf cold-start
    # treatment from `ucb_score`, so exploration-by-construction is
    # preserved without any rule-based prior on skips.
    #
    # Skip-offers are *gated by the existence of an alternative*. We only
    # add `skip:X` candidates when `len(legal) > 1` — i.e. when skipping X
    # would still leave at least one other legal action for the router to
    # pick. This rules out the degenerate case where the only legal skill
    # gets skipped and the run terminates with no_legal_actions producing
    # nothing. Skip is a coordination *choice between alternatives*; if
    # there is no alternative, invocation is mandatory. Termination of
    # otherwise-progressing runs is handled by `evaluator_done`,
    # `budget_exhausted`, and `no_legal_actions` — not by skip.
    candidates = list(legal)
    if enable_skip and len(legal) > 1:
        candidates += [f"{SKIP_PREFIX}{x}" for x in legal]

    # Try the policy graph first — only override when confident.
    recommended = policy_graph.best_action(
        signature,
        candidates,
        c=exploration_c,
        confidence_threshold=confidence_threshold,
        reliability_weight=reliability_weight,
    )
    if recommended is not None:
        return RoutingDecision(recommended, "graph_recommendation")

    # Fall back to the rule-based prior. The prior never recommends a skip
    # action — it's the conservative default, and skipping is something
    # only the learned policy should do.
    return RoutingDecision(legal[0], "rule_based_prior")
