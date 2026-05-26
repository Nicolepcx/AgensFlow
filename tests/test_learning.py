"""
Tests for the Layer 1 learning subpackage.

Pure-function tests — no LLM calls. The integration test (the policy graph
actually growing during a real run) is covered by the demo script
`examples/02_policy_learning_demo.py`; here we cover the data structures and
invariants.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agensflow.learning.belief import update_belief
from agensflow.learning.persistence import load_policy_graph, save_policy_graph
from agensflow.learning.policy_graph import GraphNode, PolicyGraph
from agensflow.learning.reward import RewardInputs, compute_reward
from agensflow.learning.signature import _bucket, belief_signature
from agensflow.schema import Belief, Handoff


# --------------------------------------------------------------------------- #
# Belief updates
# --------------------------------------------------------------------------- #


class TestBeliefUpdate:
    def test_planner_increases_handoff_quality(self) -> None:
        prior = Belief()
        post = update_belief(
            prior,
            agent="planner",
            handoff=Handoff(goal="g", subproblem="s"),
        )
        assert post.estimated_handoff_quality > prior.estimated_handoff_quality
        assert post.estimated_uncertainty < prior.estimated_uncertainty

    def test_memory_scales_with_evidence_count(self) -> None:
        prior = Belief()
        few = update_belief(
            prior, agent="memory", handoff=Handoff(evidence=["one"])
        )
        many = update_belief(
            prior, agent="memory", handoff=Handoff(evidence=["a", "b", "c", "d"])
        )
        assert many.estimated_evidence_sufficiency > few.estimated_evidence_sufficiency

    def test_solver_increases_correctness_when_draft_present(self) -> None:
        prior = Belief()
        post = update_belief(
            prior, agent="solver", handoff=Handoff(draft_answer="x")
        )
        assert post.estimated_correctness > prior.estimated_correctness

    def test_solver_no_change_when_no_draft(self) -> None:
        prior = Belief()
        post = update_belief(
            prior, agent="solver", handoff=Handoff()
        )
        assert post.estimated_correctness == prior.estimated_correctness

    def test_critic_increases_contradiction_risk(self) -> None:
        prior = Belief(estimated_contradiction_risk=0.3)
        post = update_belief(
            prior, agent="critic", handoff=Handoff(critique="found a problem")
        )
        assert post.estimated_contradiction_risk > prior.estimated_contradiction_risk

    def test_verifier_supported_increases_correctness(self) -> None:
        prior = Belief()
        verdict = json.dumps(
            {"verdict": "supported", "rationale": "ok", "uncertain_claims": []}
        )
        post = update_belief(
            prior, agent="verifier", handoff=Handoff(verification=verdict)
        )
        assert post.estimated_correctness > prior.estimated_correctness
        assert post.estimated_uncertainty < prior.estimated_uncertainty

    def test_verifier_unsupported_decreases_correctness(self) -> None:
        prior = Belief(estimated_correctness=0.6)
        verdict = json.dumps(
            {"verdict": "unsupported", "rationale": "bad", "uncertain_claims": []}
        )
        post = update_belief(
            prior, agent="verifier", handoff=Handoff(verification=verdict)
        )
        assert post.estimated_correctness < prior.estimated_correctness
        assert post.estimated_contradiction_risk > prior.estimated_contradiction_risk

    def test_verifier_handles_malformed_verification_string(self) -> None:
        prior = Belief()
        # Not JSON — should silently produce no change rather than crash.
        post = update_belief(
            prior, agent="verifier", handoff=Handoff(verification="not json")
        )
        assert post.estimated_correctness == prior.estimated_correctness

    def test_clipping_to_unit_interval(self) -> None:
        # Push correctness above 1.0 by stacking solver updates.
        b = Belief(estimated_correctness=0.95)
        post = update_belief(
            b, agent="solver", handoff=Handoff(draft_answer="x")
        )
        assert 0.0 <= post.estimated_correctness <= 1.0


# --------------------------------------------------------------------------- #
# Signature folding
# --------------------------------------------------------------------------- #


class TestBucket:
    def test_default_granularity_buckets_to_tenths(self) -> None:
        assert _bucket(0.42, 0.1) == 0.4
        assert _bucket(0.46, 0.1) == 0.5
        assert _bucket(0.0, 0.1) == 0.0
        assert _bucket(1.0, 0.1) == 1.0

    def test_clipped_to_unit(self) -> None:
        assert _bucket(2.5, 0.1) == 1.0
        assert _bucket(-0.3, 0.1) == 0.0


class TestBeliefSignature:
    def test_equivalent_states_fold_to_same_signature(self) -> None:
        h1 = Handoff(
            goal="g", subproblem="s",
            belief=Belief(estimated_correctness=0.41),
        )
        h2 = Handoff(
            goal="other goal", subproblem="other s",
            belief=Belief(estimated_correctness=0.43),
        )
        # Different content, same observable status flags + same bucketed belief.
        assert belief_signature(h1, "evidence_heavy") == belief_signature(h2, "evidence_heavy")

    def test_different_regimes_fold_separately(self) -> None:
        h = Handoff(goal="g", subproblem="s")
        sig_a = belief_signature(h, "straightforward")
        sig_b = belief_signature(h, "evidence_heavy")
        assert sig_a != sig_b

    def test_different_handoff_status_flags_fold_separately(self) -> None:
        h_no_evidence = Handoff(goal="g", subproblem="s", evidence=[])
        h_with_evidence = Handoff(goal="g", subproblem="s", evidence=["fact"])
        assert belief_signature(h_no_evidence, "evidence_heavy") != belief_signature(
            h_with_evidence, "evidence_heavy"
        )

    def test_signatures_are_hashable_and_picklable(self) -> None:
        import pickle
        sig = belief_signature(Handoff(), "straightforward")
        # Hashable
        assert hash(sig)
        # Picklable
        roundtrip = pickle.loads(pickle.dumps(sig))
        assert roundtrip == sig

    def test_invalid_granularity_rejected(self) -> None:
        with pytest.raises(ValueError):
            belief_signature(Handoff(), "straightforward", granularity=0.0)
        with pytest.raises(ValueError):
            belief_signature(Handoff(), "straightforward", granularity=1.5)


# --------------------------------------------------------------------------- #
# PolicyGraph
# --------------------------------------------------------------------------- #


class TestGraphNode:
    def test_value_is_zero_when_unvisited(self) -> None:
        node = GraphNode(signature=("x",))
        assert node.value == 0.0
        assert node.action_value("any") == 0.0

    def test_action_value_is_mean(self) -> None:
        node = GraphNode(signature=("x",))
        node.action_visits["go"] = 3
        node.action_value_sums["go"] = 1.5
        assert node.action_value("go") == pytest.approx(0.5)

    def test_ucb_unvisited_is_infinite(self) -> None:
        node = GraphNode(signature=("x",))
        node.visits = 5
        assert node.ucb_score("never_tried") == float("inf")

    def test_ucb_higher_for_higher_value(self) -> None:
        node = GraphNode(signature=("x",))
        node.visits = 10
        node.action_visits["a"] = 5
        node.action_visits["b"] = 5
        node.action_value_sums["a"] = 4.0  # mean 0.8
        node.action_value_sums["b"] = 1.0  # mean 0.2
        assert node.ucb_score("a") > node.ucb_score("b")


class TestPolicyGraph:
    def test_empty(self) -> None:
        g = PolicyGraph()
        assert len(g) == 0
        assert g.stats() == {
            "n_nodes": 0, "total_visits": 0, "total_edges": 0, "confident_nodes": 0
        }

    def test_get_or_create_creates(self) -> None:
        g = PolicyGraph()
        n = g.get_or_create(("sig",))
        assert isinstance(n, GraphNode)
        assert len(g) == 1

    def test_get_or_create_returns_existing(self) -> None:
        g = PolicyGraph()
        n1 = g.get_or_create(("sig",))
        n2 = g.get_or_create(("sig",))
        assert n1 is n2

    def test_record_transition_links_signatures(self) -> None:
        g = PolicyGraph()
        g.record_transition(("a",), "go", ("b",))
        assert g.has_signature(("a",))
        assert g.has_signature(("b",))
        assert g.nodes[("a",)].outgoing["go"] == ("b",)

    def test_backup_increments_visits_and_value(self) -> None:
        g = PolicyGraph()
        path = [(("a",), "go"), (("b",), "stop")]
        g.backup(path, reward=1.0)
        assert g.nodes[("a",)].visits == 1
        assert g.nodes[("a",)].value == 1.0
        assert g.nodes[("a",)].action_value("go") == 1.0
        assert g.nodes[("b",)].visits == 1
        assert g.nodes[("b",)].action_value("stop") == 1.0

    def test_backup_accumulates_across_runs(self) -> None:
        g = PolicyGraph()
        g.backup([(("a",), "go")], reward=1.0)
        g.backup([(("a",), "go")], reward=0.0)
        # Mean of 1.0 and 0.0 == 0.5.
        assert g.nodes[("a",)].action_value("go") == pytest.approx(0.5)
        assert g.nodes[("a",)].visits == 2

    def test_best_action_returns_none_below_threshold(self) -> None:
        g = PolicyGraph()
        g.backup([(("a",), "go")], reward=1.0)
        # Only 1 visit, default threshold 3 → not confident yet.
        assert g.best_action(("a",), ["go", "stop"]) is None

    def test_best_action_picks_highest_ucb(self) -> None:
        g = PolicyGraph()
        # Build up enough visits to be "confident".
        for _ in range(3):
            g.backup([(("a",), "go")], reward=1.0)
        for _ in range(3):
            g.backup([(("a",), "stop")], reward=0.0)
        assert g.best_action(("a",), ["go", "stop"]) == "go"

    def test_stats_after_multiple_runs(self) -> None:
        g = PolicyGraph()
        g.record_transition(("a",), "go", ("b",))
        g.backup([(("a",), "go"), (("b",), "stop")], reward=0.7)
        g.backup([(("a",), "go"), (("b",), "stop")], reward=0.5)
        g.backup([(("a",), "go"), (("b",), "stop")], reward=0.6)
        stats = g.stats()
        assert stats["n_nodes"] == 2
        assert stats["total_visits"] == 6  # 3 visits to each of 2 nodes
        assert stats["confident_nodes"] == 2  # both have ≥3 visits


# --------------------------------------------------------------------------- #
# Chunk 11.C1: discounted backup
# --------------------------------------------------------------------------- #


class TestDiscountedBackup:
    """The `gamma` kwarg on `backup()` discounts each edge's credit by its
    distance from the path's terminal action. `gamma=1.0` (default) is the
    chunk-2..10 undiscounted behavior; `gamma<1.0` makes earlier edges
    accumulate less reward per run.

    Invariants tested:
      - gamma=1.0 produces IDENTICAL state to the pre-C1 backup (regression).
      - The terminal edge ALWAYS gets the full reward (gamma^0 = 1).
      - Earlier edges decay geometrically with distance from terminal.
      - Welford reward variance updates use the edge's discounted reward
        consistently (mean + variance stay internally coherent).
      - Token Welford updates are NOT discounted (tokens are facts about
        the action's invocation, not its position).
    """

    def test_gamma_1_matches_undiscounted(self) -> None:
        # gamma=1.0 must produce the same state as the default call.
        g_default = PolicyGraph()
        g_default.backup([(("a",), "go"), (("b",), "stop")], reward=1.0)

        g_explicit = PolicyGraph()
        g_explicit.backup([(("a",), "go"), (("b",), "stop")], reward=1.0, gamma=1.0)

        for sig in [("a",), ("b",)]:
            assert g_default.nodes[sig].visits == g_explicit.nodes[sig].visits
            assert g_default.nodes[sig].value_sum == g_explicit.nodes[sig].value_sum
            for action in g_default.nodes[sig].action_visits:
                assert (
                    g_default.nodes[sig].action_value(action)
                    == g_explicit.nodes[sig].action_value(action)
                )

    def test_terminal_edge_gets_full_reward(self) -> None:
        # Last edge (gamma^0 = 1) always gets the unmodified reward,
        # regardless of gamma.
        for gamma in [1.0, 0.9, 0.5, 0.1]:
            g = PolicyGraph()
            path = [(("a",), "go"), (("b",), "mid"), (("c",), "stop")]
            g.backup(path, reward=1.0, gamma=gamma)
            # Terminal edge (c, stop) gets full reward.
            assert g.nodes[("c",)].action_value("stop") == pytest.approx(1.0), (
                f"terminal edge should get full reward at gamma={gamma}"
            )

    def test_earlier_edges_decay_geometrically(self) -> None:
        # 3-edge path with gamma=0.5:
        #   first edge:  0.5^2 * 1.0 = 0.25
        #   middle edge: 0.5^1 * 1.0 = 0.5
        #   last edge:   0.5^0 * 1.0 = 1.0
        g = PolicyGraph()
        path = [(("a",), "go"), (("b",), "mid"), (("c",), "stop")]
        g.backup(path, reward=1.0, gamma=0.5)
        assert g.nodes[("a",)].action_value("go") == pytest.approx(0.25)
        assert g.nodes[("b",)].action_value("mid") == pytest.approx(0.5)
        assert g.nodes[("c",)].action_value("stop") == pytest.approx(1.0)

    def test_welford_variance_uses_discounted_reward(self) -> None:
        # With gamma=0.9 on a 2-edge path and reward=1.0:
        #   first edge:  0.9 * 1.0 = 0.9
        #   last edge:   1.0
        # Run twice with the same reward → variance should be 0 for both
        # (each edge sees the same discounted value across both backups).
        g = PolicyGraph()
        path = [(("a",), "go"), (("b",), "stop")]
        g.backup(path, reward=1.0, gamma=0.9)
        g.backup(path, reward=1.0, gamma=0.9)
        assert g.nodes[("a",)].action_reward_variance("go") == pytest.approx(0.0)
        assert g.nodes[("b",)].action_reward_variance("stop") == pytest.approx(0.0)
        # And mean reward at each edge is the discounted value:
        assert g.nodes[("a",)].action_value("go") == pytest.approx(0.9)
        assert g.nodes[("b",)].action_value("stop") == pytest.approx(1.0)

    def test_token_welford_not_discounted(self) -> None:
        # Tokens are NOT discounted — they're facts about the action's
        # invocation, not its position on the path. Same gamma run, same
        # action_tokens → token_mean should equal the raw token value.
        g = PolicyGraph()
        path = [(("a",), "go"), (("b",), "stop")]
        g.backup(
            path, reward=1.0, gamma=0.5,
            action_tokens={"go": 1000, "stop": 1500},
        )
        # Token means are the raw token counts, NOT discounted.
        assert g.nodes[("a",)].action_token_mean("go") == 1000
        assert g.nodes[("b",)].action_token_mean("stop") == 1500

    def test_single_edge_path_unaffected_by_gamma(self) -> None:
        # A 1-edge path's only edge IS the terminal edge → always full reward.
        for gamma in [1.0, 0.5, 0.0]:
            g = PolicyGraph()
            g.backup([(("a",), "go")], reward=0.7, gamma=gamma)
            assert g.nodes[("a",)].action_value("go") == pytest.approx(0.7), (
                f"single-edge path should get full reward at gamma={gamma}"
            )

    def test_gamma_zero_zeros_all_but_terminal(self) -> None:
        # gamma=0 is the degenerate case: only the terminal edge gets
        # credit; all earlier edges get zero. Useful as the "purely myopic"
        # regression test.
        g = PolicyGraph()
        path = [(("a",), "go"), (("b",), "mid"), (("c",), "stop")]
        g.backup(path, reward=1.0, gamma=0.0)
        assert g.nodes[("a",)].action_value("go") == pytest.approx(0.0)
        assert g.nodes[("b",)].action_value("mid") == pytest.approx(0.0)
        assert g.nodes[("c",)].action_value("stop") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Mechanism A+C: per-edge failure tracking + reliability-aware UCB
# --------------------------------------------------------------------------- #


class TestFailureTracking:
    """The reliability-over-time substrate.

    These tests cover the storyline that needs to hold for chunk 7: a model
    binding that repeatedly trips Instructor validation should be
    downweighted by UCB even if its recovered-retry reward looks fine.
    """

    def test_action_failure_rate_zero_when_no_data(self) -> None:
        node = GraphNode(signature=("x",))
        assert node.action_failure_rate("any") == 0.0

    def test_action_failure_rate_partial(self) -> None:
        node = GraphNode(signature=("x",))
        node.action_visits["go"] = 3  # 3 completed
        node.action_failure_count["go"] = 1  # 1 failure
        # 1 / (3 + 1) = 0.25
        assert node.action_failure_rate("go") == pytest.approx(0.25)

    def test_action_failure_rate_all_failures(self) -> None:
        node = GraphNode(signature=("x",))
        node.action_failure_count["go"] = 4  # never completed, only failed
        assert node.action_failure_rate("go") == pytest.approx(1.0)

    def test_record_failure_creates_node(self) -> None:
        g = PolicyGraph()
        g.record_failure(("a",), "solver_haiku")
        assert g.has_signature(("a",))
        assert g.nodes[("a",)].action_failure_count["solver_haiku"] == 1

    def test_record_failure_increments(self) -> None:
        g = PolicyGraph()
        g.record_failure(("a",), "solver_haiku")
        g.record_failure(("a",), "solver_haiku")
        g.record_failure(("a",), "solver_haiku")
        assert g.nodes[("a",)].action_failure_count["solver_haiku"] == 3

    def test_failure_count_independent_of_visits(self) -> None:
        """A recovered retry counts as a completed visit AND a failure."""
        g = PolicyGraph()
        g.backup([(("a",), "solver_haiku")], reward=0.7)  # recovered → visit
        g.record_failure(("a",), "solver_haiku")  # but it tripped on the way
        node = g.nodes[("a",)]
        assert node.action_visits["solver_haiku"] == 1
        assert node.action_failure_count["solver_haiku"] == 1
        # Failure rate = 1 / (1 + 1) = 0.5
        assert node.action_failure_rate("solver_haiku") == pytest.approx(0.5)

    def test_ucb_truly_unvisited_returns_inf(self) -> None:
        """An action with zero visits AND zero failures still gets +inf
        for cold-start exploration — reliability term is skipped."""
        node = GraphNode(signature=("x",))
        node.visits = 5
        assert node.ucb_score("never_seen") == float("inf")

    def test_ucb_failures_only_is_finite_and_negative(self) -> None:
        """An action whose only attempts have all failed must NOT return
        +inf — that would force the policy to keep retrying it."""
        node = GraphNode(signature=("x",))
        node.visits = 5
        node.action_failure_count["always_fails"] = 3
        score = node.ucb_score("always_fails", reliability_weight=0.5)
        assert score != float("inf")
        # Pure reliability penalty contribution is -0.5 (failure_rate=1.0 × λ=0.5),
        # plus a small positive exploration term against `action_failures` denominator.
        # Net should still be finite and small.
        assert -0.5 - 1.0 < score < 1.0

    def test_ucb_reliability_penalises_unreliable_action(self) -> None:
        """Two actions with identical mean reward and visits — the one with
        more failures should score lower."""
        node = GraphNode(signature=("x",))
        node.visits = 20
        # Both completed 10 times with mean reward 0.5.
        node.action_visits["reliable"] = 10
        node.action_value_sums["reliable"] = 5.0
        node.action_visits["flaky"] = 10
        node.action_value_sums["flaky"] = 5.0
        # `flaky` also recorded 5 validation failures along the way.
        node.action_failure_count["flaky"] = 5
        s_reliable = node.ucb_score("reliable", reliability_weight=0.5)
        s_flaky = node.ucb_score("flaky", reliability_weight=0.5)
        assert s_reliable > s_flaky
        # Reliability gap should match λ · (failure_rate_flaky − 0)
        # failure_rate_flaky = 5 / (10 + 5) = 1/3
        # gap ≈ 0.5 × 1/3 ≈ 0.1667
        assert (s_reliable - s_flaky) == pytest.approx(0.5 * (5 / 15), abs=1e-9)

    def test_ucb_reliability_weight_zero_disables_term(self) -> None:
        """With λ=0 the reliability term should vanish entirely."""
        node = GraphNode(signature=("x",))
        node.visits = 10
        node.action_visits["a"] = 5
        node.action_value_sums["a"] = 2.5
        node.action_failure_count["a"] = 3
        s_with = node.ucb_score("a", reliability_weight=0.5)
        s_without = node.ucb_score("a", reliability_weight=0.0)
        assert s_without > s_with  # removing the penalty improves the score
        # Specifically, the gap is exactly 0.5 × failure_rate.
        gap = s_without - s_with
        assert gap == pytest.approx(0.5 * node.action_failure_rate("a"))

    def test_best_action_prefers_reliable_under_high_lambda(self) -> None:
        """End-to-end: a high enough reliability_weight flips the policy
        from 'pick the higher-reward action' to 'pick the reliable one'."""
        g = PolicyGraph()
        # Both options visited 10 times, identical mean reward.
        for _ in range(10):
            g.backup([(("s",), "fast_but_flaky")], reward=0.6)
            g.backup([(("s",), "slow_but_solid")], reward=0.6)
        # fast_but_flaky tripped validation 6 times along the way.
        for _ in range(6):
            g.record_failure(("s",), "fast_but_flaky")
        # With a high reliability weight, the reliable variant should win.
        choice = g.best_action(
            ("s",),
            ["fast_but_flaky", "slow_but_solid"],
            reliability_weight=2.0,
            confidence_threshold=3,
        )
        assert choice == "slow_but_solid"

    def test_best_action_threads_reliability_weight(self) -> None:
        """Verify default DEFAULT_RELIABILITY_WEIGHT=0.5 is honored when
        reliability_weight is not passed explicitly."""
        g = PolicyGraph()
        for _ in range(10):
            g.backup([(("s",), "a")], reward=0.5)
            g.backup([(("s",), "b")], reward=0.5)
        # `a` has many failures, `b` has none.
        for _ in range(8):
            g.record_failure(("s",), "a")
        # With the default λ=0.5, b should win.
        choice = g.best_action(
            ("s",), ["a", "b"], confidence_threshold=3
        )
        assert choice == "b"


# --------------------------------------------------------------------------- #
# Chunk-9: Welford variance tracking for reward and tokens.
# --------------------------------------------------------------------------- #


class TestWelfordVariance:
    """Per-(signature, action) running variance for reward + tokens.

    Validates the chunk-9 substrate addition: variance is a first-class
    signal in the policy graph, not just an offline aggregation."""

    def test_zero_visits_returns_zero_variance(self) -> None:
        node = GraphNode(signature=("s",))
        assert node.action_reward_variance("a") == 0.0
        assert node.action_token_variance("a") == 0.0

    def test_single_visit_returns_zero_variance(self) -> None:
        """Sample variance with n=1 is undefined; we return 0.0 rather than
        raise so downstream consumers can plot uniformly."""
        g = PolicyGraph()
        g.backup([(("s",), "a")], reward=0.7, action_tokens={"a": 100})
        node = g.nodes[("s",)]
        assert node.action_visits["a"] == 1
        assert node.action_reward_variance("a") == 0.0
        assert node.action_token_variance("a") == 0.0

    def test_reward_variance_matches_textbook_formula(self) -> None:
        """Welford should agree with the closed-form sample variance
        when computed from a known sequence."""
        import statistics
        rewards = [0.1, 0.3, 0.5, 0.9, 0.7]
        g = PolicyGraph()
        for r in rewards:
            g.backup([(("s",), "a")], reward=r)
        node = g.nodes[("s",)]
        expected = statistics.variance(rewards)  # sample variance, n-1 denom
        assert node.action_reward_variance("a") == pytest.approx(expected, abs=1e-9)

    def test_token_variance_matches_textbook_formula(self) -> None:
        import statistics
        tokens = [100, 120, 80, 200, 150]
        g = PolicyGraph()
        for t in tokens:
            g.backup([(("s",), "a")], reward=0.5, action_tokens={"a": t})
        node = g.nodes[("s",)]
        expected = statistics.variance(tokens)
        assert node.action_token_variance("a") == pytest.approx(expected, abs=1e-9)
        # And the mean should match.
        assert node.action_token_mean("a") == pytest.approx(
            statistics.mean(tokens), abs=1e-9
        )

    def test_token_tracking_skipped_when_not_supplied(self) -> None:
        """A backup without `action_tokens` shouldn't update token state.
        Lets chunk-7-era backup paths keep working without changes."""
        g = PolicyGraph()
        g.backup([(("s",), "a")], reward=0.5)  # no action_tokens
        g.backup([(("s",), "a")], reward=0.5)
        node = g.nodes[("s",)]
        assert node.action_token_sums.get("a", 0.0) == 0.0
        assert node.action_token_variance("a") == 0.0
        # But reward variance still works since reward was supplied.
        # 2 identical rewards → variance 0.
        assert node.action_reward_variance("a") == pytest.approx(0.0, abs=1e-9)

    def test_variance_tracks_per_action_independently(self) -> None:
        """Two actions at the same signature should accumulate independent
        variance state. Important: the chunk-9 viz needs per-(sig, action)
        granularity, not a node-level aggregate."""
        g = PolicyGraph()
        for r in [0.0, 1.0]:
            g.backup([(("s",), "stable")], reward=0.5)  # both 0.5 → variance 0
            g.backup([(("s",), "noisy")], reward=r)     # 0 + 1 → variance 0.5
        node = g.nodes[("s",)]
        assert node.action_reward_variance("stable") == pytest.approx(0.0)
        assert node.action_reward_variance("noisy") == pytest.approx(0.5)

    def test_pickle_round_trip_preserves_welford_state(self) -> None:
        """Variance state must survive serialization — chunk-9 graphs
        get warm-started just like chunk-7/8 graphs."""
        import pickle, tempfile
        from pathlib import Path
        from agensflow.learning.persistence import (
            load_policy_graph, save_policy_graph,
        )
        g = PolicyGraph()
        for r, t in [(0.5, 100), (0.7, 150), (0.3, 80)]:
            g.backup([(("s",), "a")], reward=r, action_tokens={"a": t})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pkl"
            save_policy_graph(g, path)
            loaded = load_policy_graph(path)
        node = loaded.nodes[("s",)]
        # Variance computations must match the original.
        original_node = g.nodes[("s",)]
        assert node.action_reward_variance("a") == pytest.approx(
            original_node.action_reward_variance("a"), abs=1e-12
        )
        assert node.action_token_variance("a") == pytest.approx(
            original_node.action_token_variance("a"), abs=1e-12
        )
        assert node.action_token_mean("a") == pytest.approx(
            original_node.action_token_mean("a"), abs=1e-12
        )

    def test_back_fill_on_legacy_pickle(self) -> None:
        """Legacy pickles (pre-chunk-9) must back-fill the new Welford
        fields cleanly. The back-fill in `load_policy_graph` should
        already handle this, just like the chunk-7→chunk-8 migration."""
        import pickle, tempfile
        from pathlib import Path
        from agensflow.learning.persistence import load_policy_graph

        g = PolicyGraph()
        g.backup([(("legacy",), "a")], reward=0.4)
        # Strip the chunk-9 attrs to simulate a pre-chunk-9 pickle.
        node = g.nodes[("legacy",)]
        delattr(node, "action_reward_m2")
        delattr(node, "action_token_sums")
        delattr(node, "action_token_m2")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pkl"
            with path.open("wb") as f:
                pickle.dump(g.nodes, f)
            loaded = load_policy_graph(path)

        n = loaded.nodes[("legacy",)]
        assert n.action_reward_m2 == {}
        assert n.action_token_sums == {}
        assert n.action_token_m2 == {}
        # Variance methods return 0 cleanly.
        assert n.action_reward_variance("a") == 0.0
        assert n.action_token_variance("a") == 0.0


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #


class TestReward:
    def test_done_with_supported_verifier(self) -> None:
        r = compute_reward(
            RewardInputs(
                done=True,
                verification_str=json.dumps({"verdict": "supported", "rationale": "ok"}),
                total_tokens=2000,
                n_validation_retries=0,
            )
        )
        assert r > 0  # positive reward for clean success

    def test_failure_with_unsupported_verifier(self) -> None:
        r = compute_reward(
            RewardInputs(
                done=False,
                verification_str=json.dumps({"verdict": "unsupported", "rationale": "bad"}),
                total_tokens=2000,
                n_validation_retries=0,
            )
        )
        assert r < 0

    def test_higher_token_cost_lowers_reward(self) -> None:
        common = {"done": True, "verification_str": None, "n_validation_retries": 0}
        cheap = compute_reward(RewardInputs(total_tokens=500, **common))
        expensive = compute_reward(RewardInputs(total_tokens=8000, **common))
        assert cheap > expensive

    def test_validation_retries_lower_reward(self) -> None:
        common = {"done": True, "verification_str": None, "total_tokens": 2000}
        clean = compute_reward(RewardInputs(n_validation_retries=0, **common))
        with_retry = compute_reward(RewardInputs(n_validation_retries=1, **common))
        assert clean > with_retry

    def test_no_verifier_field_does_not_crash(self) -> None:
        r = compute_reward(
            RewardInputs(
                done=True, verification_str=None, total_tokens=1000,
                n_validation_retries=0,
            )
        )
        assert isinstance(r, float)

    def test_malformed_verification_silently_ignored(self) -> None:
        r1 = compute_reward(
            RewardInputs(
                done=True, verification_str="not json", total_tokens=1000,
                n_validation_retries=0,
            )
        )
        r2 = compute_reward(
            RewardInputs(
                done=True, verification_str=None, total_tokens=1000,
                n_validation_retries=0,
            )
        )
        # Malformed verification contributes the same as no verification.
        assert r1 == r2


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_round_trip_empty_graph(self) -> None:
        g = PolicyGraph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.pkl"
            save_policy_graph(g, path)
            assert path.exists()
            loaded = load_policy_graph(path)
            assert len(loaded) == 0

    def test_round_trip_populated_graph(self) -> None:
        g = PolicyGraph()
        g.record_transition(("a",), "go", ("b",))
        g.backup([(("a",), "go"), (("b",), "stop")], reward=0.7)
        g.backup([(("a",), "go"), (("b",), "stop")], reward=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "graph.pkl"  # creates parent dir
            save_policy_graph(g, path)
            loaded = load_policy_graph(path)
        assert len(loaded) == 2
        assert loaded.nodes[("a",)].visits == 2
        assert loaded.nodes[("a",)].action_value("go") == pytest.approx(0.8)
        assert loaded.nodes[("a",)].outgoing["go"] == ("b",)

    def test_load_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "does_not_exist.pkl"
            loaded = load_policy_graph(path)
            assert isinstance(loaded, PolicyGraph)
            assert len(loaded) == 0

    def test_load_back_fills_missing_fields(self) -> None:
        """A graph pickled before Mechanism A+C added `action_failure_count`
        must still load cleanly — the loader back-fills new fields with
        their dataclass defaults so old pickles continue to work.

        Simulated by stripping the field off a live GraphNode before pickling.
        """
        import pickle

        g = PolicyGraph()
        g.backup([(("legacy_sig",), "solver_haiku")], reward=0.6)
        # Reach into the dataclass instance and remove the attr so the
        # pickle resembles a chunk-6.5-era graph.
        delattr(g.nodes[("legacy_sig",)], "action_failure_count")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_graph.pkl"
            with path.open("wb") as f:
                pickle.dump(g.nodes, f)
            loaded = load_policy_graph(path)

        node = loaded.nodes[("legacy_sig",)]
        # Field is back-filled to its empty default.
        assert node.action_failure_count == {}
        # And reliability operations work on the migrated node.
        assert node.action_failure_rate("solver_haiku") == 0.0
        # UCB is callable too — it relies on action_failure_count internally.
        score = node.ucb_score("solver_haiku")
        assert isinstance(score, float)
