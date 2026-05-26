"""
Folded policy graph — online UCB on (signature, action) cells.

The graph stores per-(signature, action) value estimates that accumulate
across runs. This is the *learning substrate* of the framework: every run
contributes its outcome back to the graph, and future runs consult the graph
when the rule-based prior is uncertain.

Lineage note: the design is POMCGS-inspired (signatures fold partial
observability into a finite-state value table; the graph is the table).
The implementation is *not* full POMCGS — there are no rollouts, no
learned transition/value estimates, no tree-search with backed-up
expansion. What's shipped is one-step UCB1 with annealed exploration
and per-edge reliability tracking — closer to a contextual bandit
over the folded signature space than to full POMDP planning. Public-
facing language: "online folded-signature UCB"; the POMCGS lineage is
preserved here as a design note.

The graph is intentionally simple in this first implementation:
  - Nodes keyed by belief signatures (tuples).
  - Per-action visit counts and value sums.
  - Per-edge "next signature" pointers (so the graph is structurally a
    directed multigraph, not just a node table).
  - Backup is non-discounted: full reward propagates to every visited node
    on the path.

Future iterations could add:
  - Discounted backup (TD-style).
  - Confidence intervals for UCB-aware exploration.
  - Pruning of low-value or rarely-visited subgraphs.
  - Parallel-update arbitration if multiple runs compete for the same node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agensflow.learning.signature import Signature

UCB_C = 1.4
"""Default UCB exploration constant. Larger c -> more exploration."""

UCB_C_FLOOR = 0.5
"""Lower bound on annealed exploration constant — prevents the policy from
collapsing to pure exploitation forever, so it can recover if a previously
good action stops working."""

UCB_ANNEAL_HALF_LIFE = 50
"""Visits at which the annealed exploration constant has decayed to half of
its initial value. Larger -> slower decay -> more sustained exploration."""

DEFAULT_CONFIDENCE_THRESHOLD = 5
"""Default minimum visits to a signature before the graph's recommendation is
trusted over the rule-based prior. Raised from 3 (chunk 4 default) after the
chunk 5 finding that 3 was too eager — UCB started overriding the prior
before there was enough data to trust the alternative routes."""

DEFAULT_RELIABILITY_WEIGHT = 0.5
"""Default weight on the per-(signature, action) failure-rate term in the UCB
score. With this weight, an action that has failed in 50% of attempts gets a
-0.25 penalty against its UCB score — meaningful but not dominant. Higher
weights make the policy more reliability-sensitive (preferring lower-failure
actions even at some cost to mean reward); lower weights treat failures as
just noise that the reward signal already captures."""


def annealed_exploration_c(
    *,
    base_c: float,
    node_visits: int,
    half_life: int = UCB_ANNEAL_HALF_LIFE,
    floor: float = UCB_C_FLOOR,
) -> float:
    """
    Decay the UCB exploration constant geometrically with node visits.

    c_effective = max(floor, base_c * 0.5 ** (visits / half_life))

    Early in a node's life (low visit count), exploration is wide. As visits
    accumulate, exploration narrows toward the floor — but never reaches
    zero, so the policy retains the ability to revisit alternatives if their
    value changes later.
    """
    if node_visits <= 0:
        return base_c
    decayed = base_c * (0.5 ** (node_visits / max(1, half_life)))
    return max(floor, decayed)


@dataclass
class GraphNode:
    """One folded state node in the policy graph."""

    signature: Signature
    visits: int = 0
    value_sum: float = 0.0
    action_visits: dict[str, int] = field(default_factory=dict)
    action_value_sums: dict[str, float] = field(default_factory=dict)
    outgoing: dict[str, Signature] = field(default_factory=dict)
    # Per-(signature, action) failure tally. Incremented by
    # `PolicyGraph.record_failure` when an Instructor validation retry fires
    # at this edge. Independent from `action_visits`, which only counts
    # *completed* actions on the policy path. Used by `ucb_score` to
    # downweight unreliable actions even when their reward looks fine
    # (because the recovered retry produced a normal reward).
    action_failure_count: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Chunk-9: Welford running-variance tracking for reward and tokens.
    # ------------------------------------------------------------------ #
    #
    # The chunk-9 systems-perspective claim — that (skill, model)
    # combinations differ in *reliability* and not just *mean quality* —
    # requires variance to be observable per edge. Welford's online
    # algorithm gives us numerically-stable running variance with the
    # same data we already pass through `backup` (reward) plus a new
    # token signal we feed in alongside.
    #
    # Two parallel signals tracked:
    #   - reward variance per (signature, action): how stable is the
    #     output quality across attempts at this edge?
    #   - token variance per (signature, action): how stable is the
    #     output length / cost across attempts? (Skill-as-constraint test
    #     — does the SKILL.md actually constrain the model's verbosity?)
    #
    # Welford state per action: M2 (sum of squared deviations from running
    # mean). Variance = M2 / (n - 1) for sample variance, M2 / n for
    # population variance. We use sample variance throughout since the
    # observations are samples from a stochastic policy.
    action_reward_m2: dict[str, float] = field(default_factory=dict)
    action_token_sums: dict[str, float] = field(default_factory=dict)
    action_token_m2: dict[str, float] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Mean reward across visits to this node."""
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def action_visit_count(self, action: str) -> int:
        return self.action_visits.get(action, 0)

    def action_value(self, action: str) -> float:
        """Mean reward for taking `action` from this node."""
        n = self.action_visits.get(action, 0)
        if n == 0:
            return 0.0
        return self.action_value_sums.get(action, 0.0) / n

    def action_reward_variance(self, action: str) -> float:
        """Sample variance of reward across visits to (signature, action).

        Returns 0.0 for actions with <2 visits (variance undefined).
        Useful for pairs with high mean reward  but high variance are
        "lucky on average, unreliable per attempt"
        The framework should learn to deprioritize even
        when mean reward looks fine.
        """
        n = self.action_visits.get(action, 0)
        if n < 2:
            return 0.0
        m2 = self.action_reward_m2.get(action, 0.0)
        return m2 / (n - 1)

    def action_token_mean(self, action: str) -> float:
        """Mean total tokens across visits to (signature, action).

        Tracks output cost per edge separately from reward. Lets it
        surface per-(skill, model) cost directly without re-aggregating
        from the trace, and gives action_token_variance a denominator.
        """
        n = self.action_visits.get(action, 0)
        if n == 0:
            return 0.0
        return self.action_token_sums.get(action, 0.0) / n

    def action_token_variance(self, action: str) -> float:
        """Sample variance of total tokens across visits to (sig, action).

        Direct test of the skills as runtime constraints claim:
        a skill spec that genuinely constrains a model's output length
        should produce lower variance in token count than the same
        model with an unconstrained skill — even if the mean is similar.
        """
        n = self.action_visits.get(action, 0)
        if n < 2:
            return 0.0
        m2 = self.action_token_m2.get(action, 0.0)
        return m2 / (n - 1)

    def action_failure_rate(self, action: str) -> float:
        """
        Fraction of attempts at `action` that produced a validation failure.

        Computed against the union of completed visits and recorded failures
        (so a 100%-failing action that has never *completed* still gets a
        meaningful failure rate of 1.0, not 0/0). This matches what we want
        for reliability-aware UCB: an action that always fails is maximally
        unreliable regardless of how many "completed visits" it has.
        """
        n_failures = self.action_failure_count.get(action, 0)
        n_visits = self.action_visits.get(action, 0)
        denom = n_visits + n_failures
        if denom == 0:
            return 0.0
        return n_failures / denom

    def ucb_score(
        self,
        action: str,
        *,
        c: float = UCB_C,
        reliability_weight: float = DEFAULT_RELIABILITY_WEIGHT,
    ) -> float:
        """
        UCB1 score for `action`, with a reliability penalty.

        Composition:
            score = mean_reward + c·√(ln(N+1)/n) − λ·failure_rate

        The reliability term is applied even when the action has never
        completed (n=0), because a never-completed action that has only
        produced failures should not be treated the same as a never-tried
        action. The tie-breaking with +inf for unvisited completion paths
        is preserved by checking `action_visits == 0 AND failure_count == 0`
        (truly unvisited) before applying the reliability term.
        """
        action_n = self.action_visits.get(action, 0)
        action_failures = self.action_failure_count.get(action, 0)
        if action_n == 0 and action_failures == 0:
            # Truly unvisited — give +inf to drive exploration.
            return float("inf")
        if action_n == 0:
            # All attempts have been failures → pure reliability penalty,
            # no exploitation term, and exploration uses failure attempts
            # as the denominator so we don't blow up.
            return -reliability_weight * 1.0 + c * math.sqrt(
                math.log(self.visits + 1) / max(1, action_failures)
            )
        exploitation = self.action_value(action)
        exploration = c * math.sqrt(math.log(self.visits + 1) / action_n)
        reliability = -reliability_weight * self.action_failure_rate(action)
        return exploitation + exploration + reliability


class PolicyGraph:
    """In-memory folded policy graph."""

    def __init__(self) -> None:
        self.nodes: dict[Signature, GraphNode] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def get_or_create(self, signature: Signature) -> GraphNode:
        if signature not in self.nodes:
            self.nodes[signature] = GraphNode(signature=signature)
        return self.nodes[signature]

    def has_signature(self, signature: Signature) -> bool:
        return signature in self.nodes

    def record_transition(
        self,
        from_sig: Signature,
        action: str,
        to_sig: Signature,
    ) -> None:
        """Register that `action` was taken at `from_sig` and led to `to_sig`."""
        node = self.get_or_create(from_sig)
        # Remember the most recent successor for this action — last write wins.
        # Sufficient for v1; later we may want to track the distribution.
        node.outgoing[action] = to_sig
        # Ensure the destination node exists too.
        self.get_or_create(to_sig)

    def record_failure(self, signature: Signature, action: str) -> None:
        """
        Increment the per-(signature, action) failure tally.

        Called when an Instructor validation retry (or any other recoverable
        agent-side failure) fires at the edge `(signature, action)`. The
        failure is *separate* from the action visit count, because a recovered
        retry still produces a normal completed visit with a normal reward —
        but we want UCB to know that this edge is unreliable, even if the
        downstream reward looks fine.

        This is the substrate for the "reliability over time" story: as the
        policy graph accumulates failure data per edge, UCB's reliability
        term (`−λ · failure_rate`) downweights edges whose model bindings
        repeatedly trip validation, so the framework learns which model is
        most reliable for which signature in the user's domain.
        """
        node = self.get_or_create(signature)
        node.action_failure_count[action] = (
            node.action_failure_count.get(action, 0) + 1
        )

    def backup(
        self,
        path: list[tuple[Signature, str]],
        reward: float,
        *,
        action_tokens: dict[str, int] | None = None,
        gamma: float = 1.0,
    ) -> None:
        """
        Propagate `reward` to every (signature, action) on the path.

        Discounted backup: each edge gets credited with
        `reward * gamma^(path_len - 1 - i)`, where `i` is the edge's
        position in `path` (0 = first edge, path_len-1 = last edge
        immediately preceding the outcome). The terminal edge always
        gets full credit (`gamma^0 = 1`); earlier edges accumulate less.

        `gamma=1.0` (default) recovers the original undiscounted backup —
        every edge gets the full reward. This preserves chunk-2..10
        semantics. Set `gamma < 1.0` to weight downstream decisions
        more heavily, e.g. `gamma=0.9` for mild trajectory-credit
        weighting on long paths.

        Welford running variance is updated for *both* reward and tokens
        on each backed-up edge:

          - reward variance: tracks per-edge output-quality stability,
            using the EDGE'S DISCOUNTED reward (so mean + variance stay
            internally consistent under any gamma).
          - token variance: tracks per-edge output-cost stability, using
            the per-action token count from `action_tokens` (when supplied).
            Tokens are NOT discounted — they're a fact about an action's
            invocation, independent of its position on the path.

        `action_tokens` maps an action name to the total tokens that
        action consumed *on this run*. The chunk-7 backup path doesn't
        pass it (variance falls back to 0); the chunk-8/9 harnesses
        compute it from the trace and pass it explicitly so the
        (skill, model) cost-stability story is observable in the viz.
        """
        action_tokens = action_tokens or {}
        path_len = len(path)
        for i, (sig, action) in enumerate(path):
            # Distance from terminal: last edge → 0, first edge → path_len-1.
            # Last edge always gets full reward; earlier edges decay.
            distance_from_terminal = path_len - 1 - i
            edge_reward = reward * (gamma ** distance_from_terminal)

            node = self.get_or_create(sig)
            node.visits += 1
            node.value_sum += edge_reward

            prev_action_n = node.action_visits.get(action, 0)
            new_action_n = prev_action_n + 1
            node.action_visits[action] = new_action_n
            node.action_value_sums[action] = (
                node.action_value_sums.get(action, 0.0) + edge_reward
            )

            # Welford for reward — uses this edge's discounted reward
            # consistently across mean + variance. Internally consistent
            # under any gamma; with gamma=1.0 reduces to the chunk-2..10
            # behavior.
            prev_reward_mean = (
                node.action_value_sums[action] - edge_reward
            ) / prev_action_n if prev_action_n > 0 else 0.0
            new_reward_mean = node.action_value_sums[action] / new_action_n
            # m2_new = m2_old + (x - mean_old) * (x - mean_new)
            delta_old = edge_reward - prev_reward_mean
            delta_new = edge_reward - new_reward_mean
            node.action_reward_m2[action] = (
                node.action_reward_m2.get(action, 0.0)
                + delta_old * delta_new
            )

            # Welford for tokens (only when action_tokens supplies a value
            # for this action). The action_tokens dict is consulted by
            # the action *name*, not by the (sig, action) edge — runs
            # that visit the same action multiple times at different
            # signatures will see the same per-action token total. That's
            # acceptable because each (sig, action) edge accumulates its
            # own per-edge mean+variance from those backups.
            #
            # Tokens are NOT discounted — token count is a fact about
            # the action's invocation, not its position on the path.
            tokens_for_action = action_tokens.get(action)
            if tokens_for_action is not None:
                tokens_float = float(tokens_for_action)
                prev_token_sum = node.action_token_sums.get(action, 0.0)
                new_token_sum = prev_token_sum + tokens_float
                node.action_token_sums[action] = new_token_sum
                prev_token_mean = (
                    prev_token_sum / prev_action_n if prev_action_n > 0 else 0.0
                )
                new_token_mean = new_token_sum / new_action_n
                delta_token_old = tokens_float - prev_token_mean
                delta_token_new = tokens_float - new_token_mean
                node.action_token_m2[action] = (
                    node.action_token_m2.get(action, 0.0)
                    + delta_token_old * delta_token_new
                )

    def best_action(
        self,
        signature: Signature,
        legal_actions: list[str],
        *,
        c: float = UCB_C,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        annealed: bool = True,
        reliability_weight: float = DEFAULT_RELIABILITY_WEIGHT,
    ) -> str | None:
        """
        Return the UCB-best legal action at `signature`, or None if the node
        has not been visited enough times to be considered confident.

        confidence_threshold: minimum visits to this signature before the
        graph's recommendation is trusted over the rule-based prior.
        Increasing this makes the graph more conservative — it waits for more
        data before overriding the prior.

        reliability_weight: λ in the `score = mean + c·exploration − λ·failure_rate`
        composition. Higher → more reliability-sensitive (prefers actions that
        rarely trip validation, even at some cost to mean reward).
        """
        if signature not in self.nodes:
            return None
        node = self.nodes[signature]
        if node.visits < confidence_threshold:
            return None
        if not legal_actions:
            return None
        c_effective = (
            annealed_exploration_c(base_c=c, node_visits=node.visits)
            if annealed
            else c
        )
        scored = [
            (
                node.ucb_score(
                    a, c=c_effective, reliability_weight=reliability_weight
                ),
                a,
            )
            for a in legal_actions
        ]
        return max(scored, key=lambda x: x[0])[1]

    def stats(self) -> dict[str, int]:
        """Quick-glance summary for demos and debugging."""
        total_visits = sum(n.visits for n in self.nodes.values())
        total_edges = sum(len(n.outgoing) for n in self.nodes.values())
        confident_nodes = sum(1 for n in self.nodes.values() if n.visits >= 3)
        return {
            "n_nodes": len(self.nodes),
            "total_visits": total_visits,
            "total_edges": total_edges,
            "confident_nodes": confident_nodes,
        }
