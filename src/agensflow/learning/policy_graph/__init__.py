"""
Policy graph package — folded-signature UCB learning substrate
(POMCGS-inspired but not full MCGS — see core.py docstring).

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.learning.policy_graph import PolicyGraph`),
so the move from `policy_graph.py` to `policy_graph/` is invisible to
callers.

See `README.md` for documentation.
"""

from agensflow.learning.policy_graph.config import PolicyGraphConfig
from agensflow.learning.policy_graph.core import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RELIABILITY_WEIGHT,
    UCB_ANNEAL_HALF_LIFE,
    UCB_C,
    UCB_C_FLOOR,
    GraphNode,
    PolicyGraph,
    annealed_exploration_c,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_RELIABILITY_WEIGHT",
    "GraphNode",
    "PolicyGraph",
    "PolicyGraphConfig",
    "UCB_ANNEAL_HALF_LIFE",
    "UCB_C",
    "UCB_C_FLOOR",
    "annealed_exploration_c",
]
