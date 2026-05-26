"""Regime detection: classifying tasks into coarse coordination regimes."""

from agensflow.regime.base import RegimeDetector
from agensflow.regime.rule_based import RuleBasedRegimeDetector, detect_regime

__all__ = ["RegimeDetector", "RuleBasedRegimeDetector", "detect_regime"]
