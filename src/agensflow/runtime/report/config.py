"""
ReportConfig — typed configuration schema for the run-report layer.

Knobs here are display-oriented: the dataclasses themselves
(`RunReport`, `SessionReport`, `AgentActivitySummary`) capture
everything; the config controls what `format_human()` *renders*.

The accompanying YAML defaults are in
`agensflow/configs/defaults/report.yaml`. The `agensflow.config`
loader merges defaults with any user YAML and validates the result
against this dataclass.

See `README.md` in this directory for the human explanation of each
knob and when to tune it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReportConfig:
    """Configuration for `RunReport.format_human()` /
    `SessionReport.format_human()`.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, pass the instance
    into report formatters, never mutate after that.
    """

    # ----- per-run table layout ----- #
    # Column widths in the per-agent activity table of `RunReport.format_human`.
    # Increase if your agent / model names are long enough to wrap; decrease
    # to fit narrower terminals.
    run_agent_col_width: int = 28
    run_model_col_width: int = 32

    # Cap on rows shown in the per-agent table of `RunReport.format_human`.
    # 0 means "no cap" (show every agent that fired). Useful for runs with
    # huge variant pools where the headline matters more than the long tail.
    run_max_agents_in_table: int = 0

    # Whether `format_human` includes the per-agent error breakdown line
    # (the "↳ errors: rate_limited=3, ..." sub-line). Disable for
    # ultra-compact output (e.g. one-line-per-run dashboards); leave on
    # when humans are reading the report directly.
    include_agent_error_detail: bool = True

    # ----- governance violation rendering ----- #
    # Cap on characters of the violation `detail` field to print in
    # `format_human`. The full detail is always preserved in the
    # serialized form (`to_dict()`); this only truncates the human view
    # for runs where the detail string is enormous.
    violation_detail_max_chars: int = 400

    # ----- session table layout ----- #
    # Column widths for `SessionReport.format_human`'s rolled-up agent table.
    session_agent_col_width: int = 32

    # Cap on rows in the session per-agent rollup. 0 means "no cap".
    # For sustained-traffic sweeps with hundreds of variants, set to 20-30
    # to focus the human report on the highest-volume agents.
    session_max_agents_in_table: int = 0
