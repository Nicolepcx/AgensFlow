"""
RewardConfig — typed configuration for the hybrid reward function.

Lives in `core.py` next to the runtime that uses it (`compute_hybrid_reward`).
This `config.py` exists so the loader's import surface
(`<module>/config.py exposes the schema`) is uniform across modules.

See `README.md` for per-knob explanation.
"""

from __future__ import annotations

from agensflow.learning.reward.core import RewardConfig

__all__ = ["RewardConfig"]
