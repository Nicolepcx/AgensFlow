"""
Regime detector interface.

The detector is an *interface* not a function, because the rule-based default
shipped here is intended to be replaced. Layer 2/3 will plug in learned
classifiers (and eventually regime-and-belief joint inference) via this
interface without changing the activation planner or runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agensflow.schema import RegimeEstimate, TaskFeatures


@runtime_checkable
class RegimeDetector(Protocol):
    """
    Maps task features to a regime estimate.

    Implementations may be rule-based (default), learned (Layer 2/3), or
    hybrid. The interface is intentionally minimal: features in, estimate out.
    Beliefs and observations are *not* part of this interface; they live in
    the runtime layer and are folded into features by the caller.
    """

    def detect(self, features: TaskFeatures) -> RegimeEstimate:
        """Return a regime estimate for the given task features."""
        ...
