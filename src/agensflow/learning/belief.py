"""
Belief update rules.

After each agent call, the latent belief estimates are updated based on what
the agent contributed. The update rules are deliberately heuristic at this
stage — symbolic deltas modeled on the notebook draft. A future iteration
will replace these with learned updaters anchored to observable proxies
(verifier outcomes, agreement across agents, logprobs where available).

The function is pure: it takes a prior Belief and a record of what just
happened, and returns the posterior. The runtime calls it after each agent
finishes, threading the new Belief into the next state.
"""

from __future__ import annotations

import json

from agensflow.schema import Belief, Handoff


def update_belief(
    prior: Belief,
    *,
    agent: str,
    handoff: Handoff,
) -> Belief:
    """
    Compute a posterior Belief given the prior and the agent that just ran.

    Update rules:
      - planner: improves handoff_quality (cleaner downstream contract),
        modestly reduces uncertainty.
      - memory: improves evidence_sufficiency proportional to retrieved
        evidence count, reduces uncertainty.
      - solver: when a draft was produced, increases correctness, reduces
        uncertainty, improves handoff_quality.
      - critic: when a critique was produced, increases contradiction_risk
        (the critic surfaces friction; that's its job) and slightly raises
        uncertainty (we have new doubts to integrate).
      - verifier: when verification was produced, parse the verdict; a
        "supported" verdict increases correctness and reduces uncertainty
        and contradiction_risk. "unsupported" decreases correctness.
      - evaluator: marks the run-state, no large belief deltas.

    All values are clipped to [0, 1] after the update.
    """
    correctness = prior.estimated_correctness
    uncertainty = prior.estimated_uncertainty
    handoff_quality = prior.estimated_handoff_quality
    contradiction_risk = prior.estimated_contradiction_risk
    evidence_sufficiency = prior.estimated_evidence_sufficiency

    if agent == "planner":
        if handoff.subproblem:
            handoff_quality += 0.10
            uncertainty -= 0.05

    elif agent == "memory":
        n_evidence = len(handoff.evidence)
        evidence_sufficiency += min(0.30, 0.10 * n_evidence)
        handoff_quality += 0.05
        uncertainty -= 0.10

    elif agent == "solver":
        if handoff.draft_answer:
            correctness += 0.20
            uncertainty -= 0.10
            handoff_quality += 0.10

    elif agent == "critic":
        if handoff.critique:
            contradiction_risk += 0.20
            uncertainty += 0.05

    elif agent == "verifier":
        if handoff.verification:
            verdict = _extract_verifier_verdict(handoff.verification)
            if verdict == "supported":
                correctness += 0.25
                uncertainty -= 0.20
                contradiction_risk -= 0.15
                evidence_sufficiency += 0.10
            elif verdict == "partially_supported":
                correctness += 0.10
                uncertainty -= 0.05
                contradiction_risk -= 0.05
            elif verdict == "unsupported":
                correctness -= 0.20
                uncertainty += 0.10
                contradiction_risk += 0.20

    elif agent == "synthesizer":
        if handoff.merged_answer:
            correctness += 0.08
            handoff_quality += 0.12
            uncertainty -= 0.05

    # evaluator: no large belief deltas at this stage. The evaluator decides
    # done-ness; its judgement feeds the reward function, not the belief.

    return Belief(
        estimated_correctness=_clip(correctness),
        estimated_uncertainty=_clip(uncertainty),
        estimated_handoff_quality=_clip(handoff_quality),
        estimated_contradiction_risk=_clip(contradiction_risk),
        estimated_evidence_sufficiency=_clip(evidence_sufficiency),
    )


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_verifier_verdict(verification_str: str) -> str | None:
    """
    Pull the structured verdict out of the verifier's serialized JSON.

    The verifier currently writes its VerifierOutput as a JSON string into
    Handoff.verification (a schema constraint we accept until the field is
    promoted to a structured type). We parse it defensively here.
    """
    try:
        parsed = json.loads(verification_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    return verdict if isinstance(verdict, str) else None
