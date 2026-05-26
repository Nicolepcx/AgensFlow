"""
Belief signature — the POMCGS folding function.

The folded policy graph keys nodes on signatures, not on raw states.
Signatures are coarse enough that semantically-equivalent states map to the
same node, allowing value estimates to be reused across runs. This is the
mechanism by which the framework learns coordination at the orchestration
level without retraining models.

Signature components:
  - regime label (the regime detected for this task)
  - observable handoff state (which fields are populated)
  - bucketed belief estimates (rounded to a configurable granularity)

Two states with the same signature are treated as equivalent. Two states with
different signatures are treated as distinct nodes in the graph. The
granularity of the rounding is the main lever controlling generalisation:
finer granularity → more nodes, less reuse; coarser → fewer nodes, more
reuse but more aliasing of distinct situations.
"""

from __future__ import annotations

from agensflow.schema import Handoff, RegimeLabel

# A Signature is a tuple — hashable, comparable, picklable. Order matters and
# is fixed.
Signature = tuple[
    str,    # regime label
    bool,   # has goal
    bool,   # has subproblem
    bool,   # has evidence
    bool,   # has draft
    bool,   # has critique
    bool,   # has verification
    bool,   # has merged_answer
    float,  # rounded estimated_correctness
    float,  # rounded estimated_uncertainty
    float,  # rounded estimated_contradiction_risk
    float,  # rounded estimated_evidence_sufficiency
]


def belief_signature(
    handoff: Handoff,
    regime_label: RegimeLabel,
    *,
    granularity: float = 0.1,
) -> Signature:
    """
    Compute a folded signature for the current state.

    Granularity controls how aggressively belief estimates are bucketed.
    Default 0.1 produces 11 buckets per estimate (0.0, 0.1, ..., 1.0).
    """
    if granularity <= 0 or granularity > 1:
        raise ValueError(f"granularity must be in (0, 1]; got {granularity}")

    return (
        regime_label,
        handoff.goal is not None,
        handoff.subproblem is not None,
        bool(handoff.evidence),
        handoff.draft_answer is not None,
        handoff.critique is not None,
        handoff.verification is not None,
        handoff.merged_answer is not None,
        _bucket(handoff.belief.estimated_correctness, granularity),
        _bucket(handoff.belief.estimated_uncertainty, granularity),
        _bucket(handoff.belief.estimated_contradiction_risk, granularity),
        _bucket(handoff.belief.estimated_evidence_sufficiency, granularity),
    )


def _bucket(value: float, granularity: float) -> float:
    """Round value to the nearest multiple of granularity, clipped to [0, 1]."""
    bucketed = round(value / granularity) * granularity
    return max(0.0, min(1.0, round(bucketed, 4)))
