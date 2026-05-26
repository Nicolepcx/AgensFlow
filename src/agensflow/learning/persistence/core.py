"""
Persistence for the policy graph.

Pickle-based save/load. The graph is plain-Python data (dataclasses + dicts +
tuples), all picklable. We write to a path the user controls; the framework
does not auto-persist anywhere by default.

The whole point of persistence is that learning compounds across runs. Without
persistence, the policy graph resets every Python process and the framework
behaves identically to the rule-based prior on every fresh launch.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from agensflow.learning.persistence.config import PersistenceConfig
from agensflow.learning.policy_graph import GraphNode, PolicyGraph


def save_policy_graph(
    graph: PolicyGraph,
    path: str | Path | None = None,
    *,
    config: PersistenceConfig | None = None,
) -> None:
    """
    Pickle the graph's nodes to `path`. Creates the parent directory if
    `config.auto_create_parent_dir` (default True).

    `path=None` resolves to `config.default_snapshot_path` (raises
    ValueError if both are unset). `config=None` uses defaults.

    Pickle format is intentionally kept simple — we serialize the `nodes` dict
    rather than the PolicyGraph instance, so future schema changes to
    PolicyGraph itself do not break old graphs (as long as GraphNode stays
    structurally compatible).
    """
    cfg = config if config is not None else PersistenceConfig()
    if path is None:
        if not cfg.default_snapshot_path:
            raise ValueError(
                "save_policy_graph: no path supplied and "
                "config.default_snapshot_path is empty. Pass path explicitly "
                "or configure a default."
            )
        path = cfg.default_snapshot_path
    p = Path(path)
    if cfg.auto_create_parent_dir:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(graph.nodes, f, protocol=cfg.pickle_protocol)


def load_policy_graph(path: str | Path) -> PolicyGraph:
    """
    Load a pickled graph. Returns an empty PolicyGraph if the file does not
    exist (so first-time callers get the expected empty-graph behavior).

    Performs a forward-compatible migration on load: any GraphNode field
    introduced after the pickle was written is back-filled with its default.
    This is what lets a chunk-6.5 graph (pickled before Mechanism A+C added
    `action_failure_count`) be warm-started into chunk-7 cleanly. The
    migration is idempotent — newly-pickled graphs already have the fields
    and the back-fill is a no-op.
    """
    p = Path(path)
    graph = PolicyGraph()
    if not p.exists():
        return graph
    with p.open("rb") as f:
        nodes = pickle.load(f)  # noqa: S301 - intentional, format is internal
    if not isinstance(nodes, dict):
        raise ValueError(
            f"Pickle at {path} did not contain a node dict; refusing to load."
        )
    # Back-fill any GraphNode fields that didn't exist when the pickle was
    # written. The field list is built from the live dataclass so future
    # additions get migrated automatically.
    from dataclasses import fields, MISSING
    node_fields = fields(GraphNode)
    for node in nodes.values():
        for f in node_fields:
            if not hasattr(node, f.name):
                if f.default is not MISSING:
                    default = f.default
                elif f.default_factory is not MISSING:  # type: ignore[misc]
                    default = f.default_factory()  # type: ignore[misc]
                else:
                    continue  # required field missing — let it explode loudly
                setattr(node, f.name, default)
    graph.nodes = nodes
    return graph
