"""
GovernancePolicy — typed configuration schema for the governance layer.

The policy doubles as both the runtime contract object (passed to
`GovernanceState`) and the OmegaConf-loadable schema (registered on
`AgensflowConfig.governance`). Keeping a single dataclass for both roles
avoids drift between "what the code consumes" and "what the YAML
exposes".

Why this file is just a re-export:

  The dataclass itself lives in `core.py` alongside the runtime that
  uses it (GovernanceState, BrokenAgentError, classify_error). That
  keeps the actual mechanism in one place. This `config.py` exists so
  the loader's import surface (`<module>/config.py exposes the schema`)
  is uniform across modules — the loader doesn't need a special-case
  for governance.

See `README.md` for per-knob explanations and tuning guidance.
"""

from __future__ import annotations

from agensflow.runtime.governance.core import GovernancePolicy

__all__ = ["GovernancePolicy"]
