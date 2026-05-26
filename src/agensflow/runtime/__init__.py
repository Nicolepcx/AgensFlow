"""
Runtime: take an ActivationPlan and execute it as a real multi-agent run.

The runtime is layered over the policy primitives in `agensflow.activation` and
`agensflow.regime`. Those layers decide *what* coalition to run; the runtime
decides *how* to run it: build a LangGraph, call real LLMs via OpenRouter,
collect a trace.

For chunk 2 we ship the linear-plan runtime end-to-end (planner -> memory ->
solver -> verifier -> evaluator), which matches the evidence_heavy regime.
Branching runtime, critic/synthesizer agents, and merge strategies are
follow-ups.
"""

from agensflow.runtime.agent_outputs import (
    EvaluatorOutput,
    MemoryOutput,
    PlannerOutput,
    SolverOutput,
    VerifierOutput,
    VerifierVerdict,
)
from agensflow.runtime.client import CompletionResult, OpenRouterClient
from agensflow.runtime.errors import InvalidAgentOutputError
from agensflow.runtime.graph import build_graph
from agensflow.runtime.models import DEFAULT_MODEL_ASSIGNMENT, get_model_for_skill
from agensflow.runtime.runner import RunResult, run
from agensflow.runtime.trace import TraceCollector, TraceEvent
from agensflow.runtime.types import Document

__all__ = [
    # Entry points
    "run",
    "RunResult",
    "Document",
    # LLM transport
    "OpenRouterClient",
    "CompletionResult",
    # Tracing
    "TraceCollector",
    "TraceEvent",
    # Graph builder
    "build_graph",
    # Model assignment
    "DEFAULT_MODEL_ASSIGNMENT",
    "get_model_for_skill",
    # Agent output schemas (the per-edge contracts)
    "PlannerOutput",
    "MemoryOutput",
    "SolverOutput",
    "VerifierOutput",
    "VerifierVerdict",
    "EvaluatorOutput",
    # Errors
    "InvalidAgentOutputError",
]
