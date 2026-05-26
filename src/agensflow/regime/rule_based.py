"""
Rule-based regime detector.

This is the default implementation, extracted from the original notebook draft
and cleaned up. It is deliberately simple and threshold-based so that the
behavior is fully inspectable and reproducible.

A learned classifier should outperform this on real workloads. The point of
shipping a rule-based default is to give users a working baseline and a clear
extension point, not to claim this is the right regime detector.
"""

from __future__ import annotations

from agensflow.regime.base import RegimeDetector
from agensflow.schema import RegimeEstimate, TaskFeatures


class RuleBasedRegimeDetector:
    """
    Thresholded rule-based regime detector.

    The thresholds and label assignments below are starting points, not
    tuned values. They are intended to be overridden by users with domain
    knowledge or replaced with a learned classifier.
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.7,
        ambiguity_threshold: float = 0.7,
        evidence_threshold: float = 0.7,
        verification_threshold_evidence: float = 0.6,
        verification_threshold_high_risk: float = 0.8,
    ) -> None:
        self.contradiction_threshold = contradiction_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.evidence_threshold = evidence_threshold
        self.verification_threshold_evidence = verification_threshold_evidence
        self.verification_threshold_high_risk = verification_threshold_high_risk

    def detect(self, features: TaskFeatures) -> RegimeEstimate:
        ambiguity = features.ambiguity_level
        contradiction = features.contradiction_risk
        evidence = features.evidence_availability
        verification = features.verification_need

        if contradiction > self.contradiction_threshold:
            return RegimeEstimate(
                label="contradictory",
                confidence=0.8,
                alternative_labels=["ambiguous", "high_risk"],
            )

        if ambiguity > self.ambiguity_threshold:
            return RegimeEstimate(
                label="ambiguous",
                confidence=0.75,
                alternative_labels=["exploratory", "contradictory"],
            )

        if (
            evidence > self.evidence_threshold
            and verification > self.verification_threshold_evidence
        ):
            return RegimeEstimate(
                label="evidence_heavy",
                confidence=0.8,
                alternative_labels=["high_risk"],
            )

        if verification > self.verification_threshold_high_risk:
            return RegimeEstimate(
                label="high_risk",
                confidence=0.7,
                alternative_labels=["evidence_heavy"],
            )

        return RegimeEstimate(
            label="straightforward",
            confidence=0.7,
            alternative_labels=["exploratory"],
        )


_default_detector: RegimeDetector = RuleBasedRegimeDetector()


def detect_regime(features: TaskFeatures) -> RegimeEstimate:
    """
    Convenience wrapper around the default rule-based detector.

    For non-default behavior (custom thresholds or a learned detector),
    instantiate the detector directly and call `.detect(features)`.
    """
    return _default_detector.detect(features)
