"""
Client package — OpenRouter HTTP client + Instructor-typed I/O.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.client import OpenRouterClient`),
so the move from `client.py` to `client/` is invisible to callers.

See `README.md` for the user-facing documentation.
"""

from agensflow.runtime.client.config import ClientConfig
from agensflow.runtime.client.core import (
    CompletionResult,
    OpenRouterClient,
    _CallContext,
    _current_call,
)

__all__ = [
    "ClientConfig",
    "CompletionResult",
    "OpenRouterClient",
    # Underscored — internal helpers exposed for tests and for advanced
    # users threading per-call attribution outside `complete_typed`.
    "_CallContext",
    "_current_call",
]
