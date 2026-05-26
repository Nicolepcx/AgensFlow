"""
PreflightConfig — typed configuration schema for pre-flight checks.

Per-provider HTTP timeouts and the default check-set live here. The
accompanying YAML defaults are in
`agensflow/configs/defaults/preflight.yaml`. The `agensflow.config`
loader merges defaults with any user YAML and validates the result
against this dataclass.

See `README.md` in this directory for the human explanation of each
knob and when to tune it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreflightConfig:
    """Configuration for the pre-flight check runner.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, pass into
    `run_preflight_checks(config=...)`, never mutate afterward.
    """

    # ----- per-provider HTTP timeouts ----- #
    # Timeout (seconds) on the OpenRouter `/models` GET. The endpoint is
    # cheap and usually responds in <2s; 10s is a safe upper bound that
    # still fail-fasts on a real outage.
    openrouter_timeout_s: float = 10.0

    # Timeout (seconds) on the EXA preflight search. Slightly longer than
    # OpenRouter because EXA's neural search has higher cold-start latency.
    exa_timeout_s: float = 15.0

    # Timeout (seconds) on the Tavily preflight search.
    tavily_timeout_s: float = 15.0

    # ----- check selection ----- #
    # Default list of checks to run when `run_preflight_checks` is called
    # without an explicit `checks=` argument. Empty list means "run every
    # check the registry knows about" (preserves backward compat).
    #
    # Tune this when an experiment doesn't need all dependencies — e.g.,
    # an LLM-only sweep can drop "exa" and "tavily" to save 30s of probe
    # time and avoid false-failures on un-set keys.
    default_checks: list[str] = field(
        default_factory=lambda: ["openrouter", "exa", "tavily"]
    )
