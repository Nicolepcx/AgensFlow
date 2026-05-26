"""
AgensFlow Layer 1 learning mechanism.

This subpackage implements the policy-learning substrate that distinguishes
AgensFlow from non-learning agent routers. The pieces:

  - belief.py         — symbolic belief updates after each agent call.
  - signature.py      — folding function from (regime, handoff, belief) to a
                        discrete signature. Equivalent states fold to the
                        same node in the policy graph.
  - policy_graph.py   — folded graph with per-(signature, action) value
                        estimates. UCB1 selection over action values.
  - reward.py         — reward computation from a completed run; the signal
                        that backpropagates through the graph.
  - persistence.py    — pickle-based save/load so learning compounds across
                        runs and across processes.

The runtime integrates these in `agensflow.runtime.runner`:
  1. Before a run, derive the initial signature; if the policy graph has
     confident value estimates at this signature, use them as planning
     advice. Otherwise, fall back to the rule-based activation plan.
  2. During a run, update the belief after each agent finishes; the
     evolving belief contributes to the signature at each step.
  3. After a run, compute the reward and backprop it through the visited
     (signature, action) path.
  4. Optionally persist the graph to disk so learning continues into the
     next process.
"""

from agensflow.learning.belief import update_belief
from agensflow.learning.persistence import load_policy_graph, save_policy_graph
from agensflow.learning.policy_graph import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    UCB_ANNEAL_HALF_LIFE,
    UCB_C,
    UCB_C_FLOOR,
    GraphNode,
    PolicyGraph,
    annealed_exploration_c,
)
from agensflow.learning.reward import (
    RewardConfig,
    RewardInputs,
    compute_hybrid_reward,
    compute_reward,
)
from agensflow.learning.router import (
    RoutingDecision,
    RoutingReason,
    select_next_action,
)
from agensflow.learning.relative_judge import (
    DEFAULT_RUBRIC,
    RelativeJudgeScoreGroup,
    RelativeJudgeScoreResult,
    SolverContribution,
    TrajectoryEvidence,
    TrajectoryToScore,
    VerifierContribution,
    build_trajectory_evidence,
    relative_judge_score_group,
)
from agensflow.learning.signature import Signature, belief_signature

__all__ = [
    # Belief
    "update_belief",
    # Signature
    "belief_signature",
    "Signature",
    # Policy graph
    "PolicyGraph",
    "GraphNode",
    "UCB_C",
    "UCB_C_FLOOR",
    "UCB_ANNEAL_HALF_LIFE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "annealed_exploration_c",
    # Reward
    "compute_reward",
    "compute_hybrid_reward",
    "RewardInputs",
    "RewardConfig",
    # RelativeJudge (relative scoring, primary signal in the hybrid reward)
    "relative_judge_score_group",
    "TrajectoryToScore",
    "RelativeJudgeScoreGroup",
    "RelativeJudgeScoreResult",
    "DEFAULT_RUBRIC",
    # Chunk 11.A1: structured trajectory evidence for the judge
    "TrajectoryEvidence",
    "SolverContribution",
    "VerifierContribution",
    "build_trajectory_evidence",
    # Router (Layer 1's "real feature" — turns the substrate into action)
    "select_next_action",
    "RoutingDecision",
    "RoutingReason",
    # Persistence
    "save_policy_graph",
    "load_policy_graph",
]
