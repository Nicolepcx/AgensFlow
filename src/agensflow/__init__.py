"""
AgensFlow: coordination as a learnable object.

From Latin *agēns* — present active participle of *agere*, "to drive, lead,
conduct, manage, do." Not the static agent, but the ongoing act of agency.

AgensFlow treats orchestration of multi-agent systems as a first-class learnable
object. Specialists stay fixed; the system learns how to coordinate them.

Public API:
    Schema:
        TaskFeatures, RegimeEstimate, SkillSpec, BranchRule, ActivationPlan,
        Handoff, RegimeLabel, MergeStrategy, SkillKind

    Regime detection:
        RegimeDetector, RuleBasedRegimeDetector, detect_regime

    Activation:
        make_activation_plan, instantiate_branches

    Registry:
        SkillRegistry, default_registry, register_skill

    Runtime (chunk 2):
        run, RunResult, Document, OpenRouterClient
"""

from agensflow.activation.branching import instantiate_branches
from agensflow.activation.planner import make_activation_plan
from agensflow.learning import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RUBRIC,
    GraphNode,
    PolicyGraph,
    RewardConfig,
    RewardInputs,
    RoutingDecision,
    RoutingReason,
    RelativeJudgeScoreGroup,
    RelativeJudgeScoreResult,
    Signature,
    SolverContribution,
    TrajectoryEvidence,
    TrajectoryToScore,
    VerifierContribution,
    annealed_exploration_c,
    belief_signature,
    build_trajectory_evidence,
    compute_hybrid_reward,
    compute_reward,
    load_policy_graph,
    relative_judge_score_group,
    save_policy_graph,
    select_next_action,
    update_belief,
)
from agensflow.regime.base import RegimeDetector
from agensflow.regime.rule_based import RuleBasedRegimeDetector, detect_regime
from agensflow.registry import SkillRegistry, default_registry, register_skill
from agensflow.runtime import (
    Document,
    OpenRouterClient,
    RunResult,
    TraceCollector,
    TraceEvent,
    run,
)
from agensflow.schema import (
    ActivationPlan,
    Belief,
    BranchRule,
    Handoff,
    MergeStrategy,
    RegimeEstimate,
    RegimeLabel,
    SkillKind,
    SkillSpec,
    TaskFeatures,
)

__version__ = "0.1.0"

__all__ = [
    # Schema
    "TaskFeatures",
    "RegimeEstimate",
    "SkillSpec",
    "BranchRule",
    "ActivationPlan",
    "Handoff",
    "Belief",
    "RegimeLabel",
    "MergeStrategy",
    "SkillKind",
    # Regime detection
    "RegimeDetector",
    "RuleBasedRegimeDetector",
    "detect_regime",
    # Activation
    "make_activation_plan",
    "instantiate_branches",
    # Registry
    "SkillRegistry",
    "default_registry",
    "register_skill",
    # Runtime
    "run",
    "RunResult",
    "Document",
    "OpenRouterClient",
    "TraceCollector",
    "TraceEvent",
    # Layer 1 learning
    "PolicyGraph",
    "GraphNode",
    "Signature",
    "belief_signature",
    "update_belief",
    "compute_reward",
    "compute_hybrid_reward",
    "RewardInputs",
    "RewardConfig",
    "relative_judge_score_group",
    "TrajectoryToScore",
    "RelativeJudgeScoreGroup",
    "RelativeJudgeScoreResult",
    "DEFAULT_RUBRIC",
    # Chunk 11.A1: structured trajectory evidence
    "TrajectoryEvidence",
    "SolverContribution",
    "VerifierContribution",
    "build_trajectory_evidence",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "annealed_exploration_c",
    "select_next_action",
    "RoutingDecision",
    "RoutingReason",
    "save_policy_graph",
    "load_policy_graph",
    # Version
    "__version__",
]
