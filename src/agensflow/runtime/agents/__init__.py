"""
Agents package — LangGraph-compatible factory functions for the
five base skills (planner, memory, solver, verifier, evaluator).

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.agents import make_planner`), so
the move from `agents.py` to `agents/` is invisible to callers.

See `README.md` for documentation.

This module deliberately exposes NO `config.py` — agent factories take
their tunables (`model_override`, `skill_name`) at construction; the
LLM-call defaults (max_retries, temperature, max_tokens) live on
`ClientConfig` and are not duplicated here. If we add user-tunable
agent-level knobs later (e.g. per-agent system-prompt overrides
unattached to skill cards), this is where the config dataclass would
land.
"""

from agensflow.runtime.agents.core import (
    AGENT_FACTORIES,
    NodeFn,
    make_evaluator,
    make_memory,
    make_planner,
    make_solver,
    make_verifier,
)

__all__ = [
    "AGENT_FACTORIES",
    "NodeFn",
    "make_evaluator",
    "make_memory",
    "make_planner",
    "make_solver",
    "make_verifier",
]
