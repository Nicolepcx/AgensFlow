"""
Models package — skill→model bindings + variant resolution.

Public re-exports preserve the import path users had before the package
conversion (`from agensflow.runtime.models import get_model_for_skill`),
so the move from `models.py` to `models/` is invisible to callers.

See `README.md` for documentation.
"""

from agensflow.runtime.models.config import ModelsConfig
from agensflow.runtime.models.core import (
    DEFAULT_MODEL_ASSIGNMENT,
    SKILL_VARIANT_BINDINGS,
    SKILL_VARIANT_CARDS,
    get_base_skill,
    get_card_for_skill,
    get_model_for_skill,
    is_variant,
)

__all__ = [
    "DEFAULT_MODEL_ASSIGNMENT",
    "ModelsConfig",
    "SKILL_VARIANT_BINDINGS",
    "SKILL_VARIANT_CARDS",
    "get_base_skill",
    "get_card_for_skill",
    "get_model_for_skill",
    "is_variant",
]
