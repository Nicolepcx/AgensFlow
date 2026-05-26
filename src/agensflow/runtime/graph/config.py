"""
GraphConfig — typed configuration for the LangGraph builders.

Exposes the LangGraph-runtime knobs (recursion limit math, enable_skip
default, enable_router_logging default) that today live as inline
constants and per-call kwargs. Most users will leave these at default;
this config is here for users tightening the recursion limit (CI runs)
or relaxing it (deep-research workloads).

The substrate-level routing knobs (`max_steps`, `confidence_threshold`,
etc.) live on `RouterConfig` and `PolicyGraphConfig` respectively. This
config only owns what's *graph-builder specific*.

See `README.md` for per-knob explanation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphConfig:
    """Configuration for `build_graph` + `build_learning_graph`.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention.
    """

    # LangGraph compiles with `recursion_limit = max(floor, multiplier
    # * max_steps + buffer)`. Each substrate "step" triggers multiple
    # LangGraph internal accounting increments (router invocation +
    # Command emission + agent node + Instructor's per-attempt
    # internals); the multiplier is the empirical worst-case budget.
    #
    # Defaults match the in-code `max(200, 12 * max_steps + 32)` from
    # before the config-conversion. Tighten in CI to fail-fast on
    # routing bugs; loosen for very long coordination paths.
    recursion_limit_floor: int = 200
    recursion_limit_per_step_multiplier: int = 12
    recursion_limit_buffer: int = 32
