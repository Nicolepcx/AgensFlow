"""
Preflight package — validate external dependencies before LLM cost.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.preflight import run_preflight_checks`),
so the move from `preflight.py` to `preflight/` is invisible to callers.

See `README.md` in this directory for the user-facing documentation of
the layer's purpose, knobs, and integration points.
"""

from agensflow.runtime.preflight.config import PreflightConfig
from agensflow.runtime.preflight.core import (
    DEFAULT_CHECKS,
    CheckResult,
    PreflightResult,
    check_exa,
    check_openrouter,
    check_tavily,
    run_preflight_checks,
)

__all__ = [
    "DEFAULT_CHECKS",
    "CheckResult",
    "PreflightConfig",
    "PreflightResult",
    "check_exa",
    "check_openrouter",
    "check_tavily",
    "run_preflight_checks",
]
