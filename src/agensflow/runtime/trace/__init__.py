"""
Trace package — `TraceCollector` and `TraceEvent` for in-memory run logging.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.trace import TraceCollector`), so the
move from `trace.py` to `trace/` is invisible to callers.

See `README.md` in this directory for the user-facing documentation.

This module deliberately exposes NO `config.py` / no YAML schema —
trace collection has no user-tunable knobs at present. The runtime
mechanisms (`enable_router_logging`, `on_event`) are configured at the
call sites that build the collector (graph builder, governance bind),
not via library-wide config. If we add user-tunable knobs later (event
retention cap, sampling rate), this is where the config dataclass would
live.
"""

from agensflow.runtime.trace.core import TraceCollector, TraceEvent

__all__ = ["TraceCollector", "TraceEvent"]
