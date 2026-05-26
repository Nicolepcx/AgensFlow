"""
RouterConfig — typed configuration for the routing decision function.

The router's per-call kwargs (`max_steps`, `exploration_c`,
`confidence_threshold`, `reliability_weight`, `enable_skip`) remain
acceptable for callers that want per-call control. This config exists
so a YAML-driven setup can supply consistent defaults across all
routing decisions in a run without threading params through every call
site.

Note: `exploration_c`, `confidence_threshold`, and `reliability_weight`
also appear on `PolicyGraphConfig`. The router's config is for
routing-specific defaults (e.g. `max_steps`); the policy-graph values
are the substrate-level defaults consumed by `best_action`. By default
we read from `PolicyGraphConfig` so the two stay aligned; the router
config only adds knobs that are routing-specific.

See `README.md` for per-knob explanation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    """Configuration for the routing decision function.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention.
    """

    # Maximum number of agent invocations the router will allow in one
    # run before forcing termination ("budget_exhausted"). 12 covers
    # every regime in chunks 6–9; raise for deeper coordination
    # workloads (research-style tasks with many sub-questions).
    max_steps: int = 12

    # Whether the router considers `skip:X` as a candidate action when
    # there are alternatives. Off by default — chunks 6/7 ran without
    # skip; chunk-8/9 turn it on to let the policy learn coordination
    # *choice between alternatives*.
    enable_skip: bool = False

    # Whether the router records per-iteration forensic entries to
    # `trace.router_log`. Off by default; flipped on for debugging
    # recursion-loop runs (the chunk-9 epoch-8 diagnostic). Cheap when
    # off (no list appends); modest overhead when on.
    enable_router_logging: bool = False
