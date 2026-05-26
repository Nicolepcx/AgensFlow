"""
Graph package — LangGraph builders (linear + dynamic-routing).

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.graph import build_learning_graph`),
so the move from `graph.py` to `graph/` is invisible to callers.

See `README.md` for documentation.
"""

from agensflow.runtime.graph.config import GraphConfig
from agensflow.runtime.graph.core import (
    ROUTER_NODE_NAME,
    NodeFn,
    build_graph,
    build_learning_graph,
)

__all__ = [
    "GraphConfig",
    "NodeFn",
    "ROUTER_NODE_NAME",
    "build_graph",
    "build_learning_graph",
]
