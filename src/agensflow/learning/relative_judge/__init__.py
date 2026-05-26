"""
RelativeJudge — peer-relative scoring of agent trajectories.

An in-house method (not a wrapper around any external framework): the
judge sees N trajectories produced for the same task and ranks them
relative to each other against an explicit rubric. Inspired by the
relative-scoring idea behind RULER (Brown et al., distinct external
work), but reimplemented here so it integrates with `OpenRouterClient`,
Instructor-validated structured output, and the substrate's
cross-judge averaging and per-axis decomposition.

See `README.md` in this directory for full documentation.
"""

from agensflow.learning.relative_judge.config import RelativeJudgeConfig
from agensflow.learning.relative_judge.core import (
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

__all__ = [
    "DEFAULT_RUBRIC",
    "RelativeJudgeConfig",
    "RelativeJudgeScoreGroup",
    "RelativeJudgeScoreResult",
    "SolverContribution",
    "TrajectoryEvidence",
    "TrajectoryToScore",
    "VerifierContribution",
    "build_trajectory_evidence",
    "relative_judge_score_group",
]
