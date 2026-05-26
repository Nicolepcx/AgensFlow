"""
Report package — `RunReport` and `SessionReport` for end-of-run artifacts.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.report import RunReport`), so the
move from `report.py` to `report/` is invisible to callers.

See `README.md` in this directory for the user-facing documentation of
the layer's purpose, knobs, and integration points.
"""

from agensflow.runtime.report.config import ReportConfig
from agensflow.runtime.report.core import (
    AgentActivitySummary,
    RunReport,
    RunStatus,
    SessionReport,
    _summarize_agents,
)

__all__ = [
    "AgentActivitySummary",
    "ReportConfig",
    "RunReport",
    "RunStatus",
    "SessionReport",
    # Internal helper exposed for tests / power users that want to
    # rebuild summaries from a raw event list outside of RunReport.
    "_summarize_agents",
]
