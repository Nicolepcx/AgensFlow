"""
Lightweight trace collection.

Every agent invocation produces one TraceEvent. The collector accumulates events
in memory during a run; the runner exposes them on the RunResult.

This is deliberately small in chunk 2. The fuller trace + metric layer (Layer 2)
will sit on top of this skeleton, computing HFE, ACE, AR, SP, and cost from the
event stream. The TraceEvent shape is designed to be metric-ready: it carries
the input handoff snapshot, the output update, the model used, and the token /
latency cost — which is everything the metric layer needs as input.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """A single agent invocation, recorded for offline inspection and metrics."""

    agent: str
    model: str
    input_state: dict[str, Any]
    output_update: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


@dataclass
class TraceCollector:
    """In-memory trace accumulator. One per run."""

    events: list[TraceEvent] = field(default_factory=list)
    # Per-router-iteration forensic log. Populated only when
    # `enable_router_logging=True` is passed to build_learning_graph().
    # Each entry captures one trip through the router_node's while-True
    # loop: the actions_taken state going in, the legal actions computed,
    # the UCB candidates considered, and the decision emitted. This is
    # what lets us debug failed runs that hit LangGraph's recursion
    # ceiling without ever recording a successful agent invocation.
    #
    # Default off so chunks 6/7/8 traces are unaffected. Chunk-9
    # diagnostic re-runs flip this on.
    router_log: list[dict[str, Any]] = field(default_factory=list)

    # Optional post-record callback. When set, fires after every event is
    # appended. The runtime uses this to plug in `GovernanceState.check_event`
    # — letting the trace stay generic about what governance means while
    # still firing the policy check at the right moment. The callback may
    # raise (e.g. BrokenAgentError); the exception propagates up through
    # record() to the agent factory, the LangGraph runtime, and finally
    # the harness, which catches it specially to skip policy-graph backup.
    on_event: Callable[["TraceEvent"], None] | None = None

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def record_router_iteration(self, entry: dict[str, Any]) -> None:
        """Append one router-loop iteration record. Cheap appendleft cost
        (a list append per inner-loop iteration) when enabled, no-op when
        the caller doesn't pass `enable_router_logging`."""
        self.router_log.append(entry)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.events)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.events)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.events)

    @property
    def total_latency_seconds(self) -> float:
        return sum(e.latency_seconds for e in self.events)

    @property
    def agent_call_sequence(self) -> list[str]:
        """The ordered sequence of agents invoked. Useful input for AR/ACE metrics later."""
        return [e.agent for e in self.events]

    def summary(self) -> str:
        """Human-readable one-line-per-event summary, plus totals."""
        lines = [
            f"  {i+1}. {e.agent:>12s} ({e.model:>30s})  "
            f"{e.total_tokens:>5d} tok  {e.latency_seconds:>5.2f}s"
            + (f"  ERROR: {e.error}" if e.error else "")
            for i, e in enumerate(self.events)
        ]
        lines.append(
            f"  total: {len(self.events)} calls  "
            f"{self.total_tokens} tokens  {self.total_latency_seconds:.2f}s"
        )
        return "\n".join(lines)
