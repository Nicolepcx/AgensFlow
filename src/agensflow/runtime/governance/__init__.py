"""
Governance package — error taxonomy, policy enforcement, run reports.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.governance import GovernancePolicy`),
so the move from `governance.py` to `governance/` is invisible to callers.

See `README.md` in this directory for the user-facing documentation of
the layer's purpose, knobs, and integration points.
"""

from agensflow.runtime.governance.config import GovernancePolicy
from agensflow.runtime.governance.core import (
    TERMINAL_REASONS,
    AgentErrorReason,
    BrokenAgentError,
    GovernanceState,
    GovernanceViolation,
    _suggest_fix_for,
    bind_governance_to_trace,
    classify_error,
    governance_logger,
    is_terminal,
    trace_logger,
)

__all__ = [
    "TERMINAL_REASONS",
    "AgentErrorReason",
    "BrokenAgentError",
    "GovernancePolicy",
    "GovernanceState",
    "GovernanceViolation",
    "bind_governance_to_trace",
    "classify_error",
    "governance_logger",
    "is_terminal",
    "trace_logger",
    # Internal helper exposed for sibling modules (preflight). Underscored
    # to mark "use at your own risk; not for end users" — but importable
    # via the package, so we don't force importers to reach into core.
    "_suggest_fix_for",
]
