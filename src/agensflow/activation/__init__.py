"""Activation planning: regime → coalition + branching + merge strategy."""

from agensflow.activation.branching import instantiate_branches
from agensflow.activation.planner import make_activation_plan

__all__ = ["make_activation_plan", "instantiate_branches"]
