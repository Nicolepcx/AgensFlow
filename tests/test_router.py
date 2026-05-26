"""
Tests for the routing decision function.

These tests don't touch any LLM. They construct synthetic ActivationPlans,
PolicyGraphs, and Handoff states, then verify that select_next_action
returns the expected (action, reason) for each scenario.

The router is the place where "learnable orchestration" becomes a runtime
choice. Getting its behaviour right is load-bearing — these tests exist to
make sure changes to routing don't silently regress.
"""

from __future__ import annotations

import pytest

from agensflow.learning.policy_graph import PolicyGraph
from agensflow.learning.router import (
    RoutingDecision,
    _evaluator_marked_done,
    _handoff_has_field,
    _legal_actions,
    _preconditions_met,
    select_next_action,
)
from agensflow.learning.signature import belief_signature
from agensflow.schema import (
    ActivationPlan,
    BranchRule,
    Handoff,
    RegimeEstimate,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _evidence_heavy_plan() -> ActivationPlan:
    return ActivationPlan(
        regime=RegimeEstimate(label="evidence_heavy", confidence=0.9),
        selected_skills=["planner", "memory", "solver", "verifier", "evaluator"],
        branch_rule=BranchRule(enabled=False),
        merge_strategy="verifier_gate",
        evaluation_criteria=["evidence_coverage"],
    )


def _straightforward_plan() -> ActivationPlan:
    return ActivationPlan(
        regime=RegimeEstimate(label="straightforward", confidence=0.9),
        selected_skills=["planner", "solver", "evaluator"],
        branch_rule=BranchRule(enabled=False),
        merge_strategy="select_best",
    )


def _populate_confident(graph: PolicyGraph, sig, action: str, reward: float, n: int) -> None:
    """Backup the same (sig, action) pair `n` times to push it past confidence_threshold."""
    for _ in range(n):
        graph.backup([(sig, action)], reward=reward)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestHandoffHasField:
    def test_none_returns_false(self) -> None:
        assert _handoff_has_field(Handoff(), "goal") is False

    def test_set_string_returns_true(self) -> None:
        assert _handoff_has_field(Handoff(goal="x"), "goal") is True

    def test_empty_list_returns_false(self) -> None:
        assert _handoff_has_field(Handoff(evidence=[]), "evidence") is False

    def test_populated_list_returns_true(self) -> None:
        assert _handoff_has_field(Handoff(evidence=["fact"]), "evidence") is True


class TestPreconditionsMet:
    def test_planner_always_legal(self) -> None:
        assert _preconditions_met(Handoff(), "planner") is True

    def test_solver_needs_subproblem(self) -> None:
        assert _preconditions_met(Handoff(), "solver") is False
        assert _preconditions_met(Handoff(subproblem="x"), "solver") is True

    def test_verifier_needs_draft_and_evidence(self) -> None:
        assert _preconditions_met(Handoff(draft_answer="x"), "verifier") is False
        assert _preconditions_met(
            Handoff(draft_answer="x", evidence=["e"]), "verifier"
        ) is True

    def test_unknown_skill_assumed_legal(self) -> None:
        # Unknown skills (user-registered) default to legal so the router
        # doesn't accidentally exclude them.
        assert _preconditions_met(Handoff(), "user_custom_skill") is True


class TestLegalActions:
    def test_initial_state_only_planner_legal(self) -> None:
        legal = _legal_actions(_evidence_heavy_plan(), Handoff(), [])
        assert legal == ["planner"]

    def test_after_planner_memory_and_solver_become_legal(self) -> None:
        # After planner runs (goal + subproblem set), memory and solver have
        # their preconditions met. Evaluator/verifier still need draft_answer.
        state = Handoff(goal="g", subproblem="s")
        legal = _legal_actions(_evidence_heavy_plan(), state, ["planner"])
        assert "memory" in legal
        assert "solver" in legal
        assert "planner" not in legal
        assert "verifier" not in legal
        assert "evaluator" not in legal

    def test_actions_taken_excluded(self) -> None:
        state = Handoff(goal="g", subproblem="s", draft_answer="a", evidence=["e"])
        legal = _legal_actions(
            _evidence_heavy_plan(), state, ["planner", "memory", "solver"]
        )
        assert legal == ["verifier", "evaluator"]

    def test_legal_actions_preserve_plan_order(self) -> None:
        # plan order: planner, memory, solver, verifier, evaluator
        state = Handoff(goal="g", subproblem="s")
        legal = _legal_actions(_evidence_heavy_plan(), state, ["planner"])
        # memory comes before solver in the plan; legal ordering should reflect that.
        assert legal.index("memory") < legal.index("solver")

    def test_empty_when_all_taken(self) -> None:
        state = Handoff(
            goal="g", subproblem="s", draft_answer="a", evidence=["e"],
            verification="v", metadata={"evaluator": {"done": True}},
        )
        legal = _legal_actions(
            _evidence_heavy_plan(), state,
            ["planner", "memory", "solver", "verifier", "evaluator"],
        )
        assert legal == []


class TestEvaluatorMarkedDone:
    def test_unset_returns_false(self) -> None:
        assert _evaluator_marked_done(Handoff()) is False

    def test_done_false_returns_false(self) -> None:
        h = Handoff(metadata={"evaluator": {"done": False}})
        assert _evaluator_marked_done(h) is False

    def test_done_true_returns_true(self) -> None:
        h = Handoff(metadata={"evaluator": {"done": True}})
        assert _evaluator_marked_done(h) is True

    def test_malformed_metadata_returns_false(self) -> None:
        h = Handoff(metadata={"evaluator": "not a dict"})
        assert _evaluator_marked_done(h) is False


# --------------------------------------------------------------------------- #
# select_next_action — termination
# --------------------------------------------------------------------------- #


class TestRouterTermination:
    def test_budget_exhausted_terminates(self) -> None:
        decision = select_next_action(
            current_state=Handoff(),
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=["x"] * 12,
            max_steps=12,
        )
        assert decision == RoutingDecision(None, "budget_exhausted")

    def test_evaluator_done_terminates(self) -> None:
        state = Handoff(
            goal="g", subproblem="s", draft_answer="a",
            metadata={"evaluator": {"done": True}},
        )
        decision = select_next_action(
            current_state=state,
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=["planner", "solver", "evaluator"],
        )
        assert decision == RoutingDecision(None, "evaluator_done")

    def test_no_legal_actions_terminates(self) -> None:
        # All skills already taken.
        state = Handoff(goal="g", subproblem="s", draft_answer="a", evidence=["e"])
        decision = select_next_action(
            current_state=state,
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=["planner", "memory", "solver", "verifier", "evaluator"],
        )
        assert decision == RoutingDecision(None, "no_legal_actions")


# --------------------------------------------------------------------------- #
# select_next_action — rule-based prior
# --------------------------------------------------------------------------- #


class TestRouterRuleBasedPrior:
    def test_initial_state_picks_planner(self) -> None:
        decision = select_next_action(
            current_state=Handoff(),
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=[],
        )
        assert decision == RoutingDecision("planner", "rule_based_prior")

    def test_after_planner_picks_memory_per_plan_order(self) -> None:
        state = Handoff(goal="g", subproblem="s")
        decision = select_next_action(
            current_state=state,
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=["planner"],
        )
        assert decision == RoutingDecision("memory", "rule_based_prior")

    def test_low_confidence_falls_back_to_prior(self) -> None:
        # Graph has visits but below the default threshold of 3.
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # 2 visits < 3 threshold, so the graph's recommendation is not trusted.
        graph.backup([(sig, "solver")], reward=1.0)
        graph.backup([(sig, "solver")], reward=1.0)

        decision = select_next_action(
            current_state=state,
            plan=plan,
            policy_graph=graph,
            actions_taken=["planner"],
            confidence_threshold=3,
        )
        # Falls back to rule-based: memory comes before solver in plan order.
        assert decision == RoutingDecision("memory", "rule_based_prior")


# --------------------------------------------------------------------------- #
# select_next_action — graph recommendation
# --------------------------------------------------------------------------- #


class TestRouterGraphRecommendation:
    def test_confident_high_value_action_wins(self) -> None:
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # Push solver well past threshold with high reward.
        _populate_confident(graph, sig, "solver", reward=2.0, n=5)
        # Memory has been tried but with low reward.
        _populate_confident(graph, sig, "memory", reward=-0.5, n=3)

        decision = select_next_action(
            current_state=state,
            plan=plan,
            policy_graph=graph,
            actions_taken=["planner"],
            confidence_threshold=3,
        )
        # The graph should recommend solver despite plan order preferring memory.
        assert decision.reason == "graph_recommendation"
        assert decision.action == "solver"

    def test_graph_recommendation_only_among_legal_actions(self) -> None:
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s", draft_answer="a", evidence=["e"])
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # Push planner with very high reward — but planner is illegal (already taken).
        _populate_confident(graph, sig, "planner", reward=5.0, n=10)

        decision = select_next_action(
            current_state=state,
            plan=plan,
            policy_graph=graph,
            actions_taken=["planner", "memory", "solver"],
            confidence_threshold=3,
        )
        # Graph cannot recommend an illegal action; choice falls within legal set.
        assert decision.action in {"verifier", "evaluator"}

    def test_unvisited_action_explored_via_ucb_infinity(self) -> None:
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # memory has been explored a lot with mediocre reward.
        _populate_confident(graph, sig, "memory", reward=0.1, n=10)
        # solver has never been tried at this signature.
        # UCB +inf for unvisited should make solver win.

        decision = select_next_action(
            current_state=state,
            plan=plan,
            policy_graph=graph,
            actions_taken=["planner"],
            confidence_threshold=3,
        )
        assert decision.reason == "graph_recommendation"
        assert decision.action == "solver"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestRouterEdgeCases:
    def test_explicit_regime_label_overrides_plan(self) -> None:
        # Explicit regime_label produces a different signature from the plan's.
        # Verify the router uses the explicit one (no exception, returns a
        # routable decision).
        decision = select_next_action(
            current_state=Handoff(),
            plan=_evidence_heavy_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=[],
            regime_label="straightforward",
        )
        # Regime label only affects signature, not legal actions, so the
        # initial decision is still planner via rule-based prior.
        assert decision.action == "planner"

    def test_straightforward_plan_skips_memory_entirely(self) -> None:
        decision = select_next_action(
            current_state=Handoff(goal="g", subproblem="s", draft_answer="a"),
            plan=_straightforward_plan(),
            policy_graph=PolicyGraph(),
            actions_taken=["planner", "solver"],
        )
        # straightforward plan has no memory or verifier; only evaluator left.
        assert decision == RoutingDecision("evaluator", "rule_based_prior")


# --------------------------------------------------------------------------- #
# Chunk-8: inline `skip:X` action mechanism
# --------------------------------------------------------------------------- #


class TestSkipActionLegalActions:
    """`_legal_actions` must respect prior `skip:X` decisions in the same run.

    A `skip:X` event in the trace excludes X from any subsequent legal-action
    computation in the same run — the policy committed not to invoke X."""

    def test_skip_action_excludes_skill(self) -> None:
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s", draft_answer="a", evidence=["e"])
        # Already invoked planner+memory+solver, then explicitly skipped verifier.
        legal = _legal_actions(
            plan, state,
            actions_taken=["planner", "memory", "solver", "skip:verifier"],
        )
        assert "verifier" not in legal
        assert legal == ["evaluator"]

    def test_invoked_and_skipped_treated_as_disjoint(self) -> None:
        # Sanity: a prior skip:X doesn't accidentally also exclude X if we
        # later (somehow) recorded an invoke of X. They're tracked as two
        # different sets within `_legal_actions`. We expect the union to
        # exclude X regardless.
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s", draft_answer="a", evidence=["e"])
        legal_skip_only = _legal_actions(
            plan, state,
            actions_taken=["planner", "memory", "solver", "skip:verifier"],
        )
        legal_invoke_only = _legal_actions(
            plan, state,
            actions_taken=["planner", "memory", "solver", "verifier"],
        )
        assert legal_skip_only == legal_invoke_only == ["evaluator"]


class TestSkipActionRouting:
    """Router-level behaviour with and without `enable_skip`."""

    def test_skip_disabled_by_default(self) -> None:
        """When enable_skip=False, the router should never return a skip:*
        action even if the policy graph has rich data.

        Construction: at this signature, only solver is legal. Even if we
        deliberately gave skip:solver +inf via never visiting it, the router
        must not see it as a candidate.
        """
        plan = _straightforward_plan()
        state = Handoff(goal="g", subproblem="s")  # solver legal
        graph = PolicyGraph()
        decision = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["planner"],
        )
        # solver is the only legal action; default is enable_skip=False.
        assert decision.action == "solver"

    def test_skip_enabled_offers_both_invoke_and_skip(self) -> None:
        """With enable_skip=True at a state with multiple legal actions, an
        unvisited skip-X candidate gets the +inf cold-start score and wins
        on the first encounter. Verifies the action-space expansion fires
        when len(legal) > 1."""
        plan = _evidence_heavy_plan()
        # State after planner: memory and solver both legal (no draft yet,
        # so verifier and evaluator are still gated out by preconditions).
        state = Handoff(goal="g", subproblem="s")
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # Push both legal actions to finite mediocre values so the
        # corresponding unvisited skip-actions win on +inf.
        _populate_confident(graph, sig, "memory", reward=0.2, n=10)
        _populate_confident(graph, sig, "solver", reward=0.2, n=10)

        decision = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["planner"],
            confidence_threshold=3,
            enable_skip=True,
        )
        assert decision.action.startswith("skip:")
        assert decision.reason == "graph_recommendation"

    def test_skip_not_offered_when_only_one_legal_action(self) -> None:
        """Fix B: when only one skill is legal at the current state,
        skip:X must NOT be offered as a candidate. Otherwise the policy
        could deterministically choose skip:X (via the +inf rule) and
        terminate the run with no work done — a degenerate covert-
        termination path. The semantic intent is: skip is a choice
        between alternatives, not a choice to do nothing."""
        plan = _evidence_heavy_plan()
        # At the very-initial Handoff(), only planner has met preconditions.
        state = Handoff()
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # Push planner to a confident state with mediocre reward, so the
        # graph recommendation path fires. If Fix B isn't in place, the
        # unvisited skip:planner +inf would beat planner — and the test
        # would assert the WRONG outcome on purpose.
        _populate_confident(graph, sig, "planner", reward=0.5, n=10)

        decision = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=[],
            confidence_threshold=3,
            enable_skip=True,
        )
        # Even with skip mechanism enabled, the only-one-legal-action gate
        # prevents skip:planner from being offered. Planner is invoked.
        assert decision.action == "planner"
        assert not decision.action.startswith("skip:")

    def test_skip_enabled_picks_invoke_when_skip_value_lower(self) -> None:
        """Inverse: when invoke and skip both have visit data and invoke
        has the higher mean value, the router picks invoke. Steady-state
        behaviour after exploration has accumulated comparable data on
        both candidates."""
        plan = _evidence_heavy_plan()
        # Two legal actions so skip-expansion fires under Fix B.
        state = Handoff(goal="g", subproblem="s")
        sig = belief_signature(state, "evidence_heavy")
        graph = PolicyGraph()
        # Solver looks great, skip:solver looks bad — solver should win.
        _populate_confident(graph, sig, "solver", reward=0.9, n=10)
        _populate_confident(graph, sig, "skip:solver", reward=0.1, n=10)
        # Memory and skip:memory both populated so neither has a +inf
        # advantage from being unvisited.
        _populate_confident(graph, sig, "memory", reward=0.5, n=10)
        _populate_confident(graph, sig, "skip:memory", reward=0.5, n=10)

        decision = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["planner"],
            confidence_threshold=3,
            enable_skip=True,
        )
        assert decision.action == "solver"

    def test_rule_based_prior_never_recommends_skip(self) -> None:
        """When the graph isn't confident, the rule-based fallback fires.
        It must always recommend a real (non-skip) action — skipping is
        only a learned policy decision, never a default."""
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        graph = PolicyGraph()  # empty — no confidence anywhere
        decision = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["planner"],
            enable_skip=True,  # mechanism enabled, but graph not confident
        )
        assert decision.reason == "rule_based_prior"
        assert not decision.action.startswith("skip:")


class TestSkipBudgetAccounting:
    """Skip events count toward `max_steps` so a runaway skip-skip loop
    can't eat budget without bound."""

    def test_skip_actions_consume_step_budget(self) -> None:
        plan = _evidence_heavy_plan()
        state = Handoff()
        graph = PolicyGraph()
        # 5 prior actions in the trace, max_steps=5 → budget exhausted,
        # whether they were invokes or skips.
        decision_invokes = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["a", "b", "c", "d", "e"], max_steps=5,
        )
        decision_skips = select_next_action(
            current_state=state, plan=plan, policy_graph=graph,
            actions_taken=["skip:a", "skip:b", "skip:c", "skip:d", "skip:e"],
            max_steps=5, enable_skip=True,
        )
        assert decision_invokes.reason == "budget_exhausted"
        assert decision_skips.reason == "budget_exhausted"


class TestActionsTakenDedup:
    """Regression test for the chunk-9 epoch-8 bug.

    Before the fix: `actions_taken` was computed from trace events with
    `error is None`, so tool agents that recorded a failure event (with
    error=set) were filtered out — and the router would re-pick them
    indefinitely until LangGraph's recursion ceiling fired.

    After the fix: `actions_taken` deduplicates by agent name regardless
    of error status. Any event for an agent means the agent was
    attempted; the router won't re-pick it.

    These tests verify `_legal_actions` still works correctly when the
    actions_taken list contains agents that were attempted-but-failed
    (the router supplies it; we just verify the legal-actions filter
    handles repetition + skip prefixes correctly)."""

    def test_legal_excludes_attempted_agent_regardless_of_error_status(self) -> None:
        """Once an agent name appears in actions_taken (whether the
        underlying event was success or error), it must be excluded
        from legal next actions. The router's dedup ensures attempted
        agents are in the list; this verifies the filter still works."""
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        # web_search_exa was attempted (and failed) — should be filtered out.
        # In the buggy version, error events weren't in actions_taken, so
        # web_search_exa stayed legal and got re-picked → infinite loop.
        # We can't easily put web_search_exa in this plan, but we can
        # verify the structural behavior with planner since the legal-
        # actions filter is generic.
        legal = _legal_actions(plan, state, actions_taken=["planner", "memory"])
        assert "planner" not in legal
        assert "memory" not in legal

    def test_repeated_agent_in_actions_taken_filtered_once(self) -> None:
        """Even if the same agent name appears multiple times in
        actions_taken (which can happen when multiple Instructor retry
        events all carry the same agent name), the legal-actions filter
        excludes it consistently."""
        plan = _evidence_heavy_plan()
        state = Handoff(goal="g", subproblem="s")
        # Simulate: planner had 3 trace events (2 retries + 1 success)
        # — the dedup'd actions_taken would just have "planner".
        # But if it accidentally appeared 3 times in actions_taken,
        # the legal filter shouldn't behave differently.
        legal_dedup = _legal_actions(plan, state, actions_taken=["planner"])
        legal_repeat = _legal_actions(
            plan, state, actions_taken=["planner", "planner", "planner"]
        )
        assert legal_dedup == legal_repeat
