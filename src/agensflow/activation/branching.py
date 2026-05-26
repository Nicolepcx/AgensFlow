"""
Branch instantiation.

Takes an ActivationPlan and produces concrete branch trajectories: each branch
is a sequence of specialist names to invoke, terminated with the evaluator.

This is deliberately separate from the planner so the runtime can decide
whether to instantiate branches eagerly (for parallel execution) or lazily
(for serial exploration with early stopping).
"""

from __future__ import annotations

from agensflow.schema import ActivationPlan


def instantiate_branches(plan: ActivationPlan) -> list[list[str]]:
    """
    Expand an ActivationPlan into one or more concrete agent sequences.

    If branching is disabled, returns a single sequence containing the plan's
    selected skills. If branching is enabled, returns up to `max_branches`
    sequences, each prefixed by the base coalition and suffixed with
    "evaluator".
    """
    selected = list(plan.selected_skills)
    branch_rule = plan.branch_rule

    if not branch_rule.enabled:
        return [selected]

    branches: list[list[str]] = []
    for branch_skills in branch_rule.branch_skill_sets:
        # Avoid duplicating skills that are already in the base coalition,
        # while preserving order of first occurrence.
        seen: set[str] = set(selected)
        addition: list[str] = []
        for s in branch_skills:
            if s not in seen:
                addition.append(s)
                seen.add(s)
        # Always terminate a branch with the evaluator.
        tail = ["evaluator"] if "evaluator" not in seen else []
        branches.append(selected + addition + tail)

    return branches[: branch_rule.max_branches]
