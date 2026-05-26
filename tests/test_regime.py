"""Tests for the rule-based regime detector."""

from __future__ import annotations

from agensflow.regime.base import RegimeDetector
from agensflow.regime.rule_based import RuleBasedRegimeDetector, detect_regime
from agensflow.schema import TaskFeatures


class TestRuleBasedRegimeDetector:
    def test_detects_contradictory_when_contradiction_high(self) -> None:
        f = TaskFeatures(contradiction_risk=0.9)
        e = detect_regime(f)
        assert e.label == "contradictory"

    def test_detects_ambiguous_when_ambiguity_high(self) -> None:
        f = TaskFeatures(ambiguity_level=0.9, contradiction_risk=0.1)
        e = detect_regime(f)
        assert e.label == "ambiguous"

    def test_detects_evidence_heavy_when_evidence_and_verification_high(self) -> None:
        f = TaskFeatures(evidence_availability=0.9, verification_need=0.7)
        e = detect_regime(f)
        assert e.label == "evidence_heavy"

    def test_detects_high_risk_when_verification_very_high(self) -> None:
        f = TaskFeatures(verification_need=0.9, evidence_availability=0.1)
        e = detect_regime(f)
        assert e.label == "high_risk"

    def test_falls_back_to_straightforward(self) -> None:
        f = TaskFeatures()
        e = detect_regime(f)
        assert e.label == "straightforward"

    def test_contradiction_dominates_ambiguity(self) -> None:
        f = TaskFeatures(ambiguity_level=0.9, contradiction_risk=0.9)
        e = detect_regime(f)
        assert e.label == "contradictory"

    def test_alternative_labels_are_provided(self) -> None:
        f = TaskFeatures(contradiction_risk=0.9)
        e = detect_regime(f)
        assert "ambiguous" in e.alternative_labels


class TestCustomThresholds:
    def test_can_lower_thresholds_to_be_more_sensitive(self) -> None:
        loose = RuleBasedRegimeDetector(contradiction_threshold=0.3)
        f = TaskFeatures(contradiction_risk=0.4)
        assert loose.detect(f).label == "contradictory"


class TestProtocolConformance:
    def test_rule_based_detector_satisfies_protocol(self) -> None:
        detector: RegimeDetector = RuleBasedRegimeDetector()
        result = detector.detect(TaskFeatures())
        assert result.label == "straightforward"
