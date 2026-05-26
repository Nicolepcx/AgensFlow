"""
AgensFlow central configuration.

Per-module config is the substrate users actually interact with. Every
module that has tunable behavior ships:

  - A frozen `dataclass` schema (in `<module>/config.py`) — typed defaults
  - A YAML file with the same defaults (in `agensflow/configs/defaults/`)
  - A `README.md` in the module package explaining what each knob does

The top-level `AgensflowConfig` composes all per-module configs into one
object. Users get a single entry point — `load_config(...)` — that:

  1. Loads the library's shipped defaults (per-module YAML files)
  2. Merges any user YAML(s) on top, preserving omitted defaults
  3. Validates the merged result against the structured `AgensflowConfig`
     schema (catching typos / wrong types early)
  4. Returns a typed `AgensflowConfig` instance for the runtime to consume

Example user YAML (override only what you need):

    governance:
      max_consecutive_failures_per_agent: 3
    web_search:
      exa_max_results: 5

Strict vs permissive:
  - `strict=True` (default) — unknown keys in the user YAML raise. Catches
    typos like `max_consecutivve_failures` early.
  - `strict=False` — unknown keys log a warning and are dropped. Useful
    for forward-compatibility (older configs against newer libraries).

CLI integration:
  Experiment runners take ONE `--config path.yaml` flag instead of dozens.
  Default to running with library defaults if no config provided.
"""

from agensflow.config.loader import (
    AgensflowConfig,
    ConfigError,
    UnknownKeyError,
    load_config,
    write_default_config,
)

__all__ = [
    "AgensflowConfig",
    "ConfigError",
    "UnknownKeyError",
    "load_config",
    "write_default_config",
]
