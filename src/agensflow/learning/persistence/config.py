"""
PersistenceConfig — typed configuration for policy-graph save/load.

Most usage stays at default (`save_policy_graph(graph, path)` /
`load_policy_graph(path)`); the config lets users tune pickle protocol
and the default snapshot path so experiments / harnesses can override
once at startup rather than threading params through every call site.

See `README.md` in this directory for the per-knob explanation.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass


@dataclass
class PersistenceConfig:
    """Configuration for policy-graph persistence.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, never mutate.
    """

    # Pickle protocol version. Defaults to HIGHEST_PROTOCOL for
    # compactness + speed; lower if you need cross-Python-version
    # interop with very old runtimes.
    pickle_protocol: int = pickle.HIGHEST_PROTOCOL

    # Whether to auto-create parent directories on save. Useful for the
    # canonical "graphs/<experiment>/<epoch>.pkl" pattern; disable if
    # you want save to fail loudly when the dir doesn't exist (e.g.
    # belt-and-suspenders against typos in the path).
    auto_create_parent_dir: bool = True

    # Default path for graph snapshots when callers don't supply one.
    # Empty string means "no default" — callers MUST supply a path.
    # Set this to e.g. `"./graphs/policy_graph.pkl"` for harnesses that
    # want consistent snapshot location across runs.
    default_snapshot_path: str = ""
