"""
Router package — `select_next_action` and routing helpers.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.learning.router import select_next_action`),
so the move from `router.py` to `router/` is invisible to callers.

See `README.md` for documentation.
"""

from agensflow.learning.router.config import RouterConfig
from agensflow.learning.router.core import (
    SKIP_PREFIX,
    RoutingDecision,
    RoutingReason,
    _evaluator_marked_done,
    _handoff_has_field,
    _legal_actions,
    _preconditions_met,
    select_next_action,
)

__all__ = [
    "RouterConfig",
    "RoutingDecision",
    "RoutingReason",
    "SKIP_PREFIX",
    "select_next_action",
    # Underscored — internal helpers exposed for the graph builder's
    # forensic logging path and for tests. Not for end users.
    "_evaluator_marked_done",
    "_handoff_has_field",
    "_legal_actions",
    "_preconditions_met",
]
