"""
Reward package — `compute_reward` (v1, baseline) and
`compute_hybrid_reward` (v2, RelativeJudge-anchored, recommended).

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.learning.reward import compute_hybrid_reward`),
so the move from `reward.py` to `reward/` is invisible to callers.

See `README.md` for documentation.
"""

from agensflow.learning.reward.config import RewardConfig
from agensflow.learning.reward.core import (
    RewardInputs,
    compute_hybrid_reward,
    compute_reward,
)

__all__ = [
    "RewardConfig",
    "RewardInputs",
    "compute_hybrid_reward",
    "compute_reward",
]
