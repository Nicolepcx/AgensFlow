"""
Security-advisory task distribution for chunk 13 (e09 cross-domain
validation).

60 queries across the same 8 scenario classes as e03 (C1-C8), driven
against the synthetic 12-doc security corpus in
`e09_cross_domain_security/corpus.py`. Per-class TaskFeatures profiles
are copied verbatim from `e03_production_traffic/tasks.py` so the
rule-based regime detector produces the same regime labels per class
across domains (the framework should learn *cross-domain transfer* of
topology, not have to re-discover that "C5 is ambiguous"). The only
change vs e03 is the corpus + the user_task content.

The classes (same as e03):

  C1 — Single-doc factual extraction (simple lookups, fast solver wins,
       verifier skip)
  C2 — Multi-doc synthesis (combine 2-3 advisories, capable solver,
       verifier sometimes useful)
  C3 — Cross-doc consistency check (does the corpus contradict itself
       on this point? capable solver + verifier)
  C4 — Underspecified question with answer in corpus (capable solver
       qualifies the answer; verifier helps catch overreach)
  C5 — Question with no answer in corpus (capable solver + verifier
       must catch the gap; web search may help for context)
  C6 — Definitional / lookup (tight, single-source, fast solver wins)
  C7 — Multi-step reasoning (chain claims across advisories; capable
       solver + verifier; web search may add value)
  C8 — Numerical / specific-fact extraction (precise extraction,
       fast solver usually OK, lightweight verification useful)

Per-task fields (identical to e03's ProductionTask shape; named
`SecurityTask` here so the documents property points at OUR corpus,
not e03's):
  - id: stable identifier
  - scenario_class: C1..C8
  - user_task: the natural-language query
  - corpus_doc_ids: the corpus subset to expose to memory
  - corpus_has_answer: whether the corpus contains the answer (False for C5)
  - expected_optimal_variant: hint for analysis — which solver variant we
       expect the policy to converge to. Not given to the framework.
  - expected_verifier_value: "skip", "useful", "essential" — analysis hint
  - expected_web_value: "no", "complement", "essential" — analysis hint
  - notes_for_rubric: canonical-answer summary passed to RULER alongside
       the rubric
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agensflow import Document, TaskFeatures

from experiments.e09_cross_domain_security.corpus import get_corpus_subset


# --------------------------------------------------------------------------- #
# Per-scenario-class TaskFeatures — copied verbatim from e03.
# Calibrated so the regime detector lands each class in the right bucket
# (C1/C6 straightforward, C5 ambiguous, rest evidence_heavy).
# --------------------------------------------------------------------------- #

CLASS_FEATURES: dict[str, TaskFeatures] = {
    "C1": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.10, contradiction_risk=0.05, novelty_level=0.10,
        evidence_availability=0.85, verification_need=0.30,
        time_horizon_complexity=0.10,
    ),
    "C2": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.25, contradiction_risk=0.10, novelty_level=0.30,
        evidence_availability=0.90, verification_need=0.70,
        time_horizon_complexity=0.40,
    ),
    "C3": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.30, contradiction_risk=0.50, novelty_level=0.20,
        evidence_availability=0.85, verification_need=0.80,
        time_horizon_complexity=0.40,
    ),
    "C4": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.55, contradiction_risk=0.15, novelty_level=0.30,
        evidence_availability=0.75, verification_need=0.75,
        time_horizon_complexity=0.30,
    ),
    "C5": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.80, contradiction_risk=0.10, novelty_level=0.50,
        evidence_availability=0.60, verification_need=0.85,
        time_horizon_complexity=0.50,
    ),
    "C6": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.05, contradiction_risk=0.0, novelty_level=0.05,
        evidence_availability=0.85, verification_need=0.20,
        time_horizon_complexity=0.05,
    ),
    "C7": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.40, contradiction_risk=0.20, novelty_level=0.50,
        evidence_availability=0.80, verification_need=0.85,
        time_horizon_complexity=0.70,
    ),
    "C8": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.10, contradiction_risk=0.05, novelty_level=0.10,
        evidence_availability=0.85, verification_need=0.50,
        time_horizon_complexity=0.10,
    ),
}


ScenarioClass = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
ExpectedVariant = Literal["solver_fast", "solver_mini", "solver_haiku"]
VerifierValue = Literal["skip", "useful", "essential"]
WebValue = Literal["no", "complement", "essential"]


@dataclass(frozen=True)
class SecurityTask:
    """Identical shape to e03's ProductionTask, but its `documents`
    property resolves against THIS experiment's corpus. The harness
    uses duck-typing on .id / .scenario_class / .user_task / .features /
    .documents, so SecurityTask works as a drop-in for ProductionTask
    in the e03-derived harness."""

    id: str
    scenario_class: ScenarioClass
    user_task: str
    corpus_doc_ids: list[str]
    corpus_has_answer: bool
    expected_optimal_variant: ExpectedVariant
    expected_verifier_value: VerifierValue
    expected_web_value: WebValue
    notes_for_rubric: str = ""

    @property
    def documents(self) -> list[Document]:
        return get_corpus_subset(self.corpus_doc_ids)

    @property
    def features(self) -> TaskFeatures:
        return CLASS_FEATURES[self.scenario_class]


# --------------------------------------------------------------------------- #
# C1 — Single-doc factual extraction (8 tasks)
# Simple direct lookups. Fast solver wins. Verifier skip. Web no.
# --------------------------------------------------------------------------- #

C1_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C1.1",
        scenario_class="C1",
        user_task="What CVSS score does CVE-DEMO-2026-001 have?",
        corpus_doc_ids=["cve-demo-001-buffer-overflow"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: 7.5 (HIGH).",
    ),
    SecurityTask(
        id="C1.2",
        scenario_class="C1",
        user_task="Who reported CVE-DEMO-2026-002?",
        corpus_doc_ids=["cve-demo-002-sql-injection"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: S. Tanaka, Quartz Security Team (internal).",
    ),
    SecurityTask(
        id="C1.3",
        scenario_class="C1",
        user_task="On what date was CVE-DEMO-2026-005 disclosed?",
        corpus_doc_ids=["cve-demo-005-auth-bypass-jwt"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: 2025-09-22.",
    ),
    SecurityTask(
        id="C1.4",
        scenario_class="C1",
        user_task="Which product is affected by CVE-DEMO-2026-007?",
        corpus_doc_ids=["cve-demo-007-supply-chain-dep"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: Beacon Mesh (versions 1.2.0 through 1.4.1).",
    ),
    SecurityTask(
        id="C1.5",
        scenario_class="C1",
        user_task=(
            "What CVSS severity category (LOW/MEDIUM/HIGH/CRITICAL) does "
            "CVE-DEMO-2026-005 fall under?"
        ),
        corpus_doc_ids=["cve-demo-005-auth-bypass-jwt"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: CRITICAL (CVSS 9.1).",
    ),
    SecurityTask(
        id="C1.6",
        scenario_class="C1",
        user_task="Who reported CVE-DEMO-2026-009?",
        corpus_doc_ids=["cve-demo-009-dos-resource-exhaustion"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: A. Volkov, Quartet Security Research.",
    ),
    SecurityTask(
        id="C1.7",
        scenario_class="C1",
        user_task=(
            "What is the affected version range stated in CVE-DEMO-2026-010?"
        ),
        corpus_doc_ids=["cve-demo-010-race-toctou"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: Polaris Vault 2.7.0 through 2.10.4.",
    ),
    SecurityTask(
        id="C1.8",
        scenario_class="C1",
        user_task="What CVSS score does CVE-DEMO-2026-008 have?",
        corpus_doc_ids=["cve-demo-008-crypto-weak-rng"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: 5.9 (MEDIUM).",
    ),
]


# --------------------------------------------------------------------------- #
# C2 — Multi-doc synthesis (8 tasks)
# Combine 2-3 advisories. Capable solver, verifier sometimes useful.
# --------------------------------------------------------------------------- #

C2_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C2.1",
        scenario_class="C2",
        user_task=(
            "Compare the mitigations recommended for CVE-DEMO-2026-004 "
            "(Helios Gateway SSRF) and CVE-DEMO-2026-011 (Beacon Mesh "
            "IDOR). What architectural pattern do both rely on for "
            "containing cross-tenant impact?"
        ),
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
            "cve-demo-011-idor-api",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Both advisories rely on per-tenant isolation (per-tenant IAM "
            "roles for 004; tenancy checks at the API layer for 011) to "
            "contain cross-tenant impact. The shared architectural "
            "pattern is enforcing tenancy boundaries at every layer "
            "rather than only at the UI gateway."
        ),
    ),
    SecurityTask(
        id="C2.2",
        scenario_class="C2",
        user_task=(
            "Looking at CVE-DEMO-2026-001 and CVE-DEMO-2026-003, what is "
            "the minimum AcmeCMS version that is patched against both?"
        ),
        corpus_doc_ids=[
            "cve-demo-001-buffer-overflow",
            "cve-demo-003-xss-stored",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "AcmeCMS 7.4.9 patches both CVE-DEMO-2026-001 (fixed in "
            "7.4.7) and CVE-DEMO-2026-003 (fixed in 7.4.9). LTS 6.7.4 "
            "covers CVE-003 but not CVE-001 (which is unaffected in 6.x)."
        ),
    ),
    SecurityTask(
        id="C2.3",
        scenario_class="C2",
        user_task=(
            "CVE-DEMO-2026-005 has CVSS 9.1; CVE-DEMO-2026-008 has CVSS "
            "5.9. Both affect Skyline Identity. Why is the gap so large "
            "given they are related vulnerabilities in the same product?"
        ),
        corpus_doc_ids=[
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-008-crypto-weak-rng",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-005 has PR:N (no authentication required) and full "
            "C+I impact (forged token grants any identity); CVE-008 has "
            "AC:H (attacker must observe valid tokens first) and only "
            "C impact (no integrity violation since session prediction "
            "doesn't forge new sessions). PR:N + UI:N + AC:L raises "
            "exploitability substantially for CVE-005."
        ),
    ),
    SecurityTask(
        id="C2.4",
        scenario_class="C2",
        user_task=(
            "Across CVE-DEMO-2026-002 and CVE-DEMO-2026-012, what is "
            "the patching strategy for a QuartzDB operator who must "
            "address both?"
        ),
        corpus_doc_ids=[
            "cve-demo-002-sql-injection",
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Upgrade to QuartzDB Server 4.3.5 or later, which patches "
            "both (4.3.3 for CVE-002, 4.3.5 for CVE-012). Operators on "
            "older releases that cannot upgrade should both disable "
            "/v1/query/raw (rest.raw_query.enabled=false) AND set "
            "quartzdb.logger.format_lookups=false."
        ),
    ),
    SecurityTask(
        id="C2.5",
        scenario_class="C2",
        user_task=(
            "Compare the mitigations across the two Helios Gateway "
            "advisories (CVE-DEMO-2026-004 and CVE-DEMO-2026-009). What "
            "Gateway version patches both?"
        ),
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
            "cve-demo-009-dos-resource-exhaustion",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Helios Gateway 3.4.7 patches both — 3.4.3 fixed CVE-004 "
            "(deny-list for URL rewrite), 3.4.7 fixed CVE-009 (recursion "
            "depth cap). Operators on older 3.x can apply both "
            "workarounds: gateway.rewrite.deny_link_local=true plus "
            "config linting via helios-config-lint."
        ),
    ),
    SecurityTask(
        id="C2.6",
        scenario_class="C2",
        user_task=(
            "Across the two Beacon Mesh advisories (CVE-DEMO-2026-007 "
            "and CVE-DEMO-2026-011), which Beacon Mesh version is "
            "patched against both, and does upgrading alone fully "
            "remediate?"
        ),
        corpus_doc_ids=[
            "cve-demo-007-supply-chain-dep",
            "cve-demo-011-idor-api",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Beacon Mesh 1.4.4 or later patches both (1.4.2 for the "
            "supply-chain pin; 1.4.4 for the IDOR tenancy check). "
            "Upgrading alone does NOT fully remediate CVE-007: "
            "operators must additionally rotate any secrets that were "
            "present in the CI environment during the 2025-05-14 to "
            "2025-06-02 affected window."
        ),
    ),
    SecurityTask(
        id="C2.7",
        scenario_class="C2",
        user_task=(
            "Compare the attack vectors (AV: component of the CVSS "
            "vector string) for CVE-DEMO-2026-006 (Polaris Vault priv-"
            "esc) and CVE-DEMO-2026-005 (Skyline JWT bypass). What "
            "operational implication does the difference have?"
        ),
        corpus_doc_ids=[
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-006-priv-esc-local",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-005 has AV:N (network-reachable) — any unauthenticated "
            "internet attacker can exploit. CVE-006 has AV:L (local) — "
            "attacker needs an existing foothold on the affected host. "
            "Operationally: CVE-005 must be patched at internet edge "
            "urgently; CVE-006 is a post-exploitation amplifier whose "
            "urgency depends on whether other footholds exist."
        ),
    ),
    SecurityTask(
        id="C2.8",
        scenario_class="C2",
        user_task=(
            "QuartzDB has two advisories in the corpus: CVE-DEMO-2026-002 "
            "(CVSS 7.6) and CVE-DEMO-2026-012 (CVSS 8.8). What does the "
            "score difference, combined with the shared architectural "
            "assumption both advisories reference, imply about the worst-"
            "case impact on the service account?"
        ),
        corpus_doc_ids=[
            "cve-demo-002-sql-injection",
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Both reference QuartzDB's single shared service account "
            "for REST traffic. CVE-002 gives SQLi (read+write to tables "
            "visible to that account); CVE-012 gives RCE under the same "
            "account. The higher CVSS for CVE-012 reflects A:H (full "
            "service-process compromise) on top of the same C:H/I:H. "
            "Combined: any exploit reaching the service account also "
            "reaches every table that account can see, including "
            "audit logs."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C3 — Cross-doc consistency check (7 tasks)
# Capable solver + verifier (consistency claims demand verification).
# --------------------------------------------------------------------------- #

C3_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C3.1",
        scenario_class="C3",
        user_task=(
            "Do both AcmeCMS advisories (CVE-DEMO-2026-001 and "
            "CVE-DEMO-2026-003) agree that multi-factor authentication "
            "is not a mitigation? Do they give the same reason?"
        ),
        corpus_doc_ids=[
            "cve-demo-001-buffer-overflow",
            "cve-demo-003-xss-stored",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Both say MFA does not prevent exploitation, but for "
            "different reasons. CVE-001: attacker only needs to "
            "authenticate as any editor (low-privileged); MFA doesn't "
            "stop someone with a valid editor account. CVE-003: "
            "attacker rides an admin's already-established session "
            "via stored XSS; MFA only gates initial login, not "
            "session continuation."
        ),
    ),
    SecurityTask(
        id="C3.2",
        scenario_class="C3",
        user_task=(
            "Both QuartzDB advisories (CVE-DEMO-2026-002 and "
            "CVE-DEMO-2026-012) reference a shared architectural "
            "feature of how QuartzDB handles REST traffic. Are the "
            "two descriptions of that architecture consistent?"
        ),
        corpus_doc_ids=[
            "cve-demo-002-sql-injection",
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Consistent. Both say QuartzDB uses a single shared "
            "service account for all REST traffic. CVE-002 introduces "
            "the pattern; CVE-012 reinforces it ('runs under the "
            "shared service account used for all REST traffic, so the "
            "impact is consistent across that advisory's architectural "
            "assumption')."
        ),
    ),
    SecurityTask(
        id="C3.3",
        scenario_class="C3",
        user_task=(
            "The two Helios Gateway advisories (CVE-DEMO-2026-004 and "
            "CVE-DEMO-2026-009) both specify version ranges. Are the "
            "ranges consistent in their starting version? In their "
            "ending version?"
        ),
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
            "cve-demo-009-dos-resource-exhaustion",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Starting versions: both 3.0.0 — consistent. Ending "
            "versions differ: CVE-004 affects through 3.4.2 (fixed in "
            "3.4.3); CVE-009 affects through 3.4.6 (fixed in 3.4.7). "
            "CVE-009 explicitly notes its range extends two point "
            "releases further because the routing-table fix landed "
            "after the SSRF fix."
        ),
    ),
    SecurityTask(
        id="C3.4",
        scenario_class="C3",
        user_task=(
            "Do CVE-DEMO-2026-006 and CVE-DEMO-2026-010 (both Polaris "
            "Vault) attribute their root cause to a shared code-review "
            "pattern? If so, what is that pattern?"
        ),
        corpus_doc_ids=[
            "cve-demo-006-priv-esc-local",
            "cve-demo-010-race-toctou",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Yes — CVE-010 explicitly states both stem from the same "
            "code-review oversight: privileged filesystem operations "
            "are guarded by permission checks that do not account for "
            "race conditions or environment manipulation. CVE-006 is "
            "the env-manipulation case (LD_PRELOAD-style); CVE-010 is "
            "the race case (TOCTOU between access() and open())."
        ),
    ),
    SecurityTask(
        id="C3.5",
        scenario_class="C3",
        user_task=(
            "Do CVE-DEMO-2026-005 (Skyline JWT bypass) and "
            "CVE-DEMO-2026-008 (Skyline weak PRNG) describe a "
            "chainable attack? Quote the source."
        ),
        corpus_doc_ids=[
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-008-crypto-weak-rng",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Yes — CVE-008 explicitly describes chainability: 'When "
            "combined with CVE-DEMO-2026-005, the impact is amplified: "
            "the attacker can both forge tokens (via the alg=none "
            "bypass) and predict legitimate tokens, depending on which "
            "path the downstream service trusts. Operators have "
            "reported successful end-to-end attack chains using both "
            "CVEs in sequence.'"
        ),
    ),
    SecurityTask(
        id="C3.6",
        scenario_class="C3",
        user_task=(
            "Do the two Beacon Mesh advisories (CVE-DEMO-2026-007 and "
            "CVE-DEMO-2026-011) cover overlapping affected versions?"
        ),
        corpus_doc_ids=[
            "cve-demo-007-supply-chain-dep",
            "cve-demo-011-idor-api",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Yes — they overlap in the 1.3.x and 1.4.x lines. CVE-007 "
            "affects 1.2.0-1.4.1; CVE-011 affects 1.3.0-1.4.3. The "
            "overlap is 1.3.0 through 1.4.1."
        ),
    ),
    SecurityTask(
        id="C3.7",
        scenario_class="C3",
        user_task=(
            "Among the corpus's advisories with CVSS 8.0 or higher, do "
            "any rely on a 'shared service account' as an aggravating "
            "factor in their impact statement? Identify each."
        ),
        corpus_doc_ids=[
            "cve-demo-002-sql-injection",
            "cve-demo-004-ssrf-cloud",
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-007-supply-chain-dep",
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Among 8.0+ advisories: CVE-004 (8.5), CVE-005 (9.1), "
            "CVE-007 (8.3), CVE-012 (8.8). Of these, only CVE-012 "
            "explicitly references QuartzDB's shared service account "
            "as an aggravating factor (echoing CVE-002). CVE-004 "
            "references a shared fleet IAM role (similar pattern but "
            "different framing). CVE-005 and CVE-007 do not invoke a "
            "shared-service-account pattern."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C4 — Underspecified question with answer in corpus (8 tasks)
# Capable solver + verifier (the verifier helps catch overreach).
# --------------------------------------------------------------------------- #

C4_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C4.1",
        scenario_class="C4",
        user_task=(
            "An organization runs AcmeCMS 7.3.0 and Skyline Identity "
            "5.5.0. Which of the two systems is more urgently "
            "exploitable from the internet?"
        ),
        corpus_doc_ids=[
            "cve-demo-001-buffer-overflow",
            "cve-demo-003-xss-stored",
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-008-crypto-weak-rng",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Skyline Identity is more urgent: CVE-005 (CVSS 9.1, "
            "CRITICAL, PR:N) lets an unauthenticated internet attacker "
            "forge any identity, including admin. AcmeCMS 7.3.0 is "
            "vulnerable to CVE-001 (BOF, CVSS 7.5, PR:L — needs "
            "editor account) and CVE-003 (XSS, CVSS 6.4, PR:L). Both "
            "AcmeCMS issues require an authenticated foothold; CVE-005 "
            "does not. Therefore Skyline is the higher priority."
        ),
    ),
    SecurityTask(
        id="C4.2",
        scenario_class="C4",
        user_task=(
            "Is QuartzDB Server 4.0.5 vulnerable to the log-format "
            "expansion RCE described in the corpus?"
        ),
        corpus_doc_ids=[
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "No — CVE-DEMO-2026-012 affects QuartzDB Server 4.1.0 "
            "through 4.3.4. Version 4.0.5 is in the unaffected 4.0.x "
            "range. (4.0.x is also outside the 3.x range that the "
            "advisory notes uses a different logging library; the "
            "advisory does not explicitly address 4.0.x, but the "
            "stated affected range begins at 4.1.0.)"
        ),
    ),
    SecurityTask(
        id="C4.3",
        scenario_class="C4",
        user_task=(
            "Across the corpus, if an operator can only patch ONE "
            "advisory immediately, which provides the largest "
            "CVSS-weighted risk reduction for a typical multi-tenant "
            "cloud deployment?"
        ),
        corpus_doc_ids=[
            "cve-demo-001-buffer-overflow",
            "cve-demo-002-sql-injection",
            "cve-demo-003-xss-stored",
            "cve-demo-004-ssrf-cloud",
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-006-priv-esc-local",
            "cve-demo-007-supply-chain-dep",
            "cve-demo-008-crypto-weak-rng",
            "cve-demo-009-dos-resource-exhaustion",
            "cve-demo-010-race-toctou",
            "cve-demo-011-idor-api",
            "cve-demo-012-log-injection-rce",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-005 (Skyline JWT bypass, CVSS 9.1 CRITICAL, AV:N, "
            "PR:N, UI:N) — it is the only CRITICAL, requires no "
            "authentication, and the impact is identity forgery across "
            "downstream services trusting Skyline Identity as SSO. "
            "For a multi-tenant cloud deployment, this is the highest "
            "single-patch ROI."
        ),
    ),
    SecurityTask(
        id="C4.4",
        scenario_class="C4",
        user_task=(
            "Does exploitation of CVE-DEMO-2026-006 (Polaris Vault "
            "setuid helper) require an attacker to first authenticate "
            "to Polaris Vault?"
        ),
        corpus_doc_ids=["cve-demo-006-priv-esc-local"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "No — CVE-006 requires only local user access on a host "
            "with Polaris Vault installed; the setuid bit on the "
            "vault-mount helper is the only precondition. The "
            "attacker does not need a Polaris Vault account or any "
            "encrypted volume. ('Exploitation does not require any "
            "Polaris Vault account or volume.')"
        ),
    ),
    SecurityTask(
        id="C4.5",
        scenario_class="C4",
        user_task=(
            "Are Beacon Mesh deployments whose binaries were built "
            "after 2025-06-02 completely safe from CVE-DEMO-2026-007?"
        ),
        corpus_doc_ids=["cve-demo-007-supply-chain-dep"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Builds after 2025-06-02 do not contain the malicious "
            "typosquatted dependency, but the advisory explicitly says "
            "they should still be upgraded ('builds from before or "
            "after that window are clean but should still be "
            "upgraded'). 'Completely safe' is therefore an overreach: "
            "they're clean of THIS supply-chain compromise but the "
            "fix in 1.4.2 also adds the registry-source allowlist "
            "that prevents recurrence."
        ),
    ),
    SecurityTask(
        id="C4.6",
        scenario_class="C4",
        user_task=(
            "Can a reader fully understand the chainable Skyline "
            "Identity attack path using only the corpus, or do they "
            "need external context?"
        ),
        corpus_doc_ids=[
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-008-crypto-weak-rng",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Fully sufficient from the corpus. CVE-008 explicitly "
            "discusses chainability with CVE-005 in its impact "
            "section. Both forging-via-alg-none and session "
            "prediction are described with enough detail to "
            "understand the end-to-end chain without external context."
        ),
    ),
    SecurityTask(
        id="C4.7",
        scenario_class="C4",
        user_task=(
            "Is the Helios Gateway DoS in CVE-DEMO-2026-009 exploitable "
            "by external internet attackers without administrator "
            "cooperation?"
        ),
        corpus_doc_ids=["cve-demo-009-dos-resource-exhaustion"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Not directly. Exploitation requires a crafted configuration "
            "to be applied — triggered by any valid admin action or "
            "SIGHUP. The advisory notes 'No authentication is required "
            "if administrators apply configuration files from an "
            "untrusted source,' meaning the attacker needs the admin "
            "to apply attacker-supplied config (e.g., via supply-chain "
            "or social engineering). The DoS is not a direct network-"
            "packet attack despite the AV:N component of the vector."
        ),
    ),
    SecurityTask(
        id="C4.8",
        scenario_class="C4",
        user_task=(
            "Across the corpus, which advisories require user "
            "interaction (UI:R in the CVSS vector) for exploitation?"
        ),
        corpus_doc_ids=[
            "cve-demo-003-xss-stored",
            "cve-demo-007-supply-chain-dep",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Two advisories have UI:R: CVE-003 (stored XSS, requires "
            "admin/editor to preview the comment in moderation queue) "
            "and CVE-007 (supply-chain, requires user interaction "
            "with a build pipeline executing the malicious dependency). "
            "All others have UI:N."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C5 — Question with no answer in corpus (7 tasks)
# Capable + verifier; web may complement. Correct answer is essentially
# "not in corpus" (plus optional web-sourced context).
# --------------------------------------------------------------------------- #

C5_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C5.1",
        scenario_class="C5",
        user_task=(
            "What CVE ID is associated with the Spectre side-channel "
            "vulnerability?"
        ),
        corpus_doc_ids=[
            # No corpus doc is relevant; pass a representative subset so
            # memory has something to consider before reporting absence.
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-006-priv-esc-local",
        ],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric=(
            "Correct: the corpus does not contain any documents on "
            "Spectre or hardware side-channel attacks. A faithful "
            "answer should state that the corpus has no advisory on "
            "Spectre; web search may supply the real-world ID "
            "(CVE-2017-5753 / CVE-2017-5715) but that is supplementary "
            "context, not corpus-grounded."
        ),
    ),
    SecurityTask(
        id="C5.2",
        scenario_class="C5",
        user_task="How does BGP hijacking work as an attack vector?",
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
            "cve-demo-009-dos-resource-exhaustion",
        ],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric=(
            "Correct: the corpus contains no documents on BGP or "
            "network-layer routing attacks. A faithful answer should "
            "acknowledge the corpus gap; web search can provide BGP "
            "context but the answer should be flagged as not "
            "corpus-grounded."
        ),
    ),
    SecurityTask(
        id="C5.3",
        scenario_class="C5",
        user_task=(
            "The corpus contains a synthetic log-format RCE advisory "
            "(CVE-DEMO-2026-012). What is the CVE ID of the real-world "
            "Log4Shell vulnerability?"
        ),
        corpus_doc_ids=["cve-demo-012-log-injection-rce"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric=(
            "Correct: the corpus contains a synthetic log-format RCE "
            "(CVE-DEMO-2026-012) but does not document the real-world "
            "Log4Shell. The corpus answer is 'not present'; web search "
            "supplies the real ID CVE-2021-44228 but only as supplementary "
            "context."
        ),
    ),
    SecurityTask(
        id="C5.4",
        scenario_class="C5",
        user_task=(
            "Does the corpus document any Kubernetes container escape "
            "vulnerabilities?"
        ),
        corpus_doc_ids=[
            "cve-demo-006-priv-esc-local",
            "cve-demo-010-race-toctou",
        ],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: no advisory in the corpus addresses Kubernetes "
            "or container-runtime escapes. CVE-006 and CVE-010 cover "
            "Linux local privilege escalation in Polaris Vault but are "
            "unrelated to container boundaries."
        ),
    ),
    SecurityTask(
        id="C5.5",
        scenario_class="C5",
        user_task=(
            "Are there any DNS rebinding vulnerabilities documented "
            "in the corpus?"
        ),
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
        ],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: no DNS-rebinding advisory in the corpus. "
            "CVE-004 is SSRF via URL-rewrite redirects (related but "
            "distinct technique); a faithful answer should not "
            "conflate it with DNS rebinding."
        ),
    ),
    SecurityTask(
        id="C5.6",
        scenario_class="C5",
        user_task=(
            "What firmware-level or pre-boot supply-chain "
            "vulnerabilities does the corpus cover?"
        ),
        corpus_doc_ids=["cve-demo-007-supply-chain-dep"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: none. CVE-007 is a software supply-chain "
            "compromise via a typosquatted package dependency at "
            "build time — operating-system / application layer. The "
            "corpus contains no firmware or pre-boot advisories."
        ),
    ),
    SecurityTask(
        id="C5.7",
        scenario_class="C5",
        user_task=(
            "Is there a cloud-IAM misconfiguration advisory in the "
            "corpus?"
        ),
        corpus_doc_ids=["cve-demo-004-ssrf-cloud"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: none. CVE-004 mentions IAM role credentials in "
            "the impact section (the SSRF retrieves them via the "
            "metadata service), but it is a code vulnerability in "
            "Helios Gateway's URL-rewrite engine, not a misconfiguration "
            "of IAM policy. The corpus contains no IAM-misconfig advisory."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C6 — Definitional / lookup (7 tasks)
# Fast solver. Concept-level, often single doc.
# --------------------------------------------------------------------------- #

C6_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C6.1",
        scenario_class="C6",
        user_task=(
            "Based on the corpus, what does the abbreviation 'CVSS' "
            "refer to in the advisories?"
        ),
        corpus_doc_ids=["cve-demo-001-buffer-overflow"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "CVSS = Common Vulnerability Scoring System, a standardized "
            "severity rating with a numeric score and a vector string "
            "(AV/AC/PR/UI/S/C/I/A) capturing exploitability and impact. "
            "The corpus uses it consistently across all 12 advisories."
        ),
    ),
    SecurityTask(
        id="C6.2",
        scenario_class="C6",
        user_task=(
            "What is a TOCTOU (time-of-check / time-of-use) race "
            "condition, as described in the corpus?"
        ),
        corpus_doc_ids=["cve-demo-010-race-toctou"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "A TOCTOU race is when a privileged operation checks a "
            "condition (e.g., file permissions via access()) and then "
            "acts on the result (e.g., open()), and an attacker can "
            "race to change the underlying state (e.g., swap the path "
            "with a symlink) between the check and the use, so the "
            "action operates on a different resource than what was "
            "checked. CVE-010 is the canonical example in the corpus."
        ),
    ),
    SecurityTask(
        id="C6.3",
        scenario_class="C6",
        user_task=(
            "What is SSRF (server-side request forgery) in the "
            "context of the corpus?"
        ),
        corpus_doc_ids=["cve-demo-004-ssrf-cloud"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "SSRF is a vulnerability where attacker-controlled input "
            "causes the server to make outbound requests to "
            "destinations the attacker chooses, bypassing network-"
            "level controls. In CVE-004 the SSRF vector is a "
            "URL-rewrite rule whose target redirects to internal "
            "metadata services (169.254.169.254) — the server "
            "fetches credentials on the attacker's behalf."
        ),
    ),
    SecurityTask(
        id="C6.4",
        scenario_class="C6",
        user_task=(
            "What is a JWT (JSON Web Token), as the corpus uses the "
            "term?"
        ),
        corpus_doc_ids=["cve-demo-005-auth-bypass-jwt"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "A JWT is a signed token asserting an identity (and "
            "claims) that downstream services trust without re-"
            "checking with the issuer. CVE-005 illustrates the "
            "trust chain: Skyline Identity issues JWTs that other "
            "Skyline services accept based on the signature; if "
            "signature verification is bypassed, any service trusting "
            "the JWT trusts a forged identity."
        ),
    ),
    SecurityTask(
        id="C6.5",
        scenario_class="C6",
        user_task=(
            "What is a setuid binary, in the context of the Polaris "
            "Vault advisory?"
        ),
        corpus_doc_ids=["cve-demo-006-priv-esc-local"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "A setuid binary is a Linux executable with the setuid "
            "bit set, causing it to run with the file owner's "
            "privileges (typically root) regardless of who invoked "
            "it. CVE-006's vault-mount helper is setuid-root so "
            "unprivileged users can mount volumes; the vulnerability "
            "is that the helper retains root effective UID through "
            "environment-controlled shared-library loading."
        ),
    ),
    SecurityTask(
        id="C6.6",
        scenario_class="C6",
        user_task=(
            "What is IDOR (insecure direct object reference) "
            "according to the Beacon Mesh advisory?"
        ),
        corpus_doc_ids=["cve-demo-011-idor-api"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "IDOR is when an API resolves resource identifiers in "
            "user input (e.g., a UUID in a URL path) without checking "
            "whether the caller is authorized to access that "
            "particular resource. In CVE-011 the telemetry API "
            "resolves UUIDs against the global resource table and "
            "returns metrics without enforcing tenancy, so a tenant "
            "user can read other tenants' resource telemetry by "
            "guessing or enumerating UUIDs."
        ),
    ),
    SecurityTask(
        id="C6.7",
        scenario_class="C6",
        user_task=(
            "What is a software supply-chain attack, as illustrated "
            "in the corpus?"
        ),
        corpus_doc_ids=["cve-demo-007-supply-chain-dep"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric=(
            "A supply-chain attack compromises software by attacking "
            "an upstream dependency or build-time component rather "
            "than the target product directly. CVE-007 is a "
            "typosquatting attack: a malicious package whose name "
            "differs from a legitimate dependency only in punctuation "
            "(`tiny-yaml-parser` vs `tiny_yaml_parser`) was served by "
            "the registry mirror used by Beacon Mesh's CI, causing "
            "build-time secret exfiltration and a backdoored "
            "downstream binary."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C7 — Multi-step reasoning (7 tasks)
# Capable + verifier; web may add value. Chain claims across advisories.
# --------------------------------------------------------------------------- #

C7_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C7.1",
        scenario_class="C7",
        user_task=(
            "Given CVE-DEMO-2026-005 (JWT bypass) and CVE-DEMO-2026-008 "
            "(weak PRNG), describe an end-to-end attack scenario that "
            "combines both. Be specific about the order of exploitation "
            "and what each step gains."
        ),
        corpus_doc_ids=[
            "cve-demo-005-auth-bypass-jwt",
            "cve-demo-008-crypto-weak-rng",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Step 1: attacker observes several valid session tokens "
            "(e.g., from an exposed admin context or a leaked log) "
            "and exploits CVE-008 to recover the PRNG seed (30 bits "
            "of entropy), predicting subsequent legitimate tokens. "
            "Step 2: for downstream services that trust JWTs from "
            "Skyline Identity, the attacker uses CVE-005 to forge "
            "JWTs with alg=none asserting any desired identity. "
            "Combined: attacker can both predict legitimate tokens "
            "(impersonate active sessions) and forge new tokens "
            "(impersonate arbitrary users including admin) — service "
            "trust in Skyline Identity gives access to every "
            "downstream system relying on that SSO."
        ),
    ),
    SecurityTask(
        id="C7.2",
        scenario_class="C7",
        user_task=(
            "An organization runs Helios Gateway 3.4.2 in a public "
            "cloud. Combining CVE-DEMO-2026-004 (SSRF) and "
            "CVE-DEMO-2026-009 (DoS via unbounded recursion), what is "
            "the worst-case operational scenario, and what makes it "
            "worse than either advisory alone?"
        ),
        corpus_doc_ids=[
            "cve-demo-004-ssrf-cloud",
            "cve-demo-009-dos-resource-exhaustion",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Worst case: an authenticated tenant operator uses CVE-004 "
            "to steal the fleet's IAM role credentials via SSRF to the "
            "169.254.169.254 metadata service. The attacker then uses "
            "those credentials to push a malicious routing configuration "
            "(e.g., via the cloud provider's config API trusted by the "
            "Gateway), triggering CVE-009 on the fleet's next reload — "
            "a sustained, fleet-wide DoS that persists across restarts "
            "until the config is rolled back. The combination is worse "
            "than either alone because CVE-004 supplies the credential "
            "needed to apply the malicious config, removing the "
            "admin-cooperation precondition for CVE-009."
        ),
    ),
    SecurityTask(
        id="C7.3",
        scenario_class="C7",
        user_task=(
            "If an organization runs both QuartzDB 4.2 and AcmeCMS 7.3, "
            "can CVE-DEMO-2026-002 be combined with CVE-DEMO-2026-003 "
            "to escalate beyond what either provides alone? Justify."
        ),
        corpus_doc_ids=[
            "cve-demo-002-sql-injection",
            "cve-demo-003-xss-stored",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Plausibly, but not directly from the corpus alone. CVE-002 "
            "(QuartzDB SQLi) gives DB-level read/write under a shared "
            "service account; CVE-003 (AcmeCMS XSS) hijacks an admin's "
            "browser session. The advisories do not document direct "
            "integration. An indirect chain: CVE-003 hijacks an "
            "AcmeCMS admin session whose user also has QuartzDB credentials "
            "cached in the browser; the attacker leverages those creds "
            "via CVE-002. The corpus does not assert this integration "
            "explicitly; a faithful answer should flag the chain as "
            "hypothetical."
        ),
    ),
    SecurityTask(
        id="C7.4",
        scenario_class="C7",
        user_task=(
            "Given CVE-DEMO-2026-006 (setuid priv-esc) and "
            "CVE-DEMO-2026-010 (TOCTOU race) on a Polaris Vault host, "
            "what is the most reliable path to root?"
        ),
        corpus_doc_ids=[
            "cve-demo-006-priv-esc-local",
            "cve-demo-010-race-toctou",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-006 is the more reliable path. It requires only "
            "local user access (no race winning), exploits the setuid "
            "bit and environment-variable-controlled library loading "
            "for deterministic root execution. CVE-010 is a 30%-per-"
            "attempt race for file-disclosure only (not direct code "
            "execution), and grants read access to other users' vault "
            "files rather than root. The shortest path to root: "
            "CVE-006 via the vault-mount helper with a crafted "
            "environment override."
        ),
    ),
    SecurityTask(
        id="C7.5",
        scenario_class="C7",
        user_task=(
            "If an organization is compromised via CVE-DEMO-2026-007 "
            "(supply-chain) on Beacon Mesh, how does CVE-DEMO-2026-011 "
            "(IDOR) change the threat model for the affected deployment?"
        ),
        corpus_doc_ids=[
            "cve-demo-007-supply-chain-dep",
            "cve-demo-011-idor-api",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-007 already implies a backdoored binary plus "
            "exfiltrated CI secrets, so the attacker has high-privilege "
            "internal access. CVE-011 (IDOR) is normally a tenant-vs-"
            "tenant disclosure issue — but in a compromised deployment, "
            "the attacker can use IDOR to enumerate cross-tenant "
            "telemetry to identify and prioritize the highest-value "
            "victim tenants for follow-on attacks (using the backdoor "
            "rather than the IDOR itself for actual writes). The "
            "threat-model change: CVE-011 stops being a tenant-"
            "disclosure issue and becomes a target-selection tool."
        ),
    ),
    SecurityTask(
        id="C7.6",
        scenario_class="C7",
        user_task=(
            "Suppose an attacker compromised a tenant's CI environment "
            "during the CVE-DEMO-2026-007 affected window. What "
            "additional information about OTHER tenants on the same "
            "Beacon Mesh deployment would CVE-DEMO-2026-011 expose to "
            "that attacker?"
        ),
        corpus_doc_ids=[
            "cve-demo-007-supply-chain-dep",
            "cve-demo-011-idor-api",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "CVE-011 exposes other tenants' resource-level metrics "
            "(request rates, response sizes, error patterns) by UUID "
            "enumeration via the telemetry API. The attacker with "
            "CI-level access can additionally enumerate or guess UUIDs "
            "more easily (build artifacts may contain references; "
            "leaked telemetry can reveal customer identities, traffic "
            "patterns, and business-sensitive operational data). "
            "CVE-011 is read-only — writes are still gated by the API-"
            "layer tenancy check on mutation endpoints."
        ),
    ),
    SecurityTask(
        id="C7.7",
        scenario_class="C7",
        user_task=(
            "State precisely the exploitation precondition for "
            "CVE-DEMO-2026-009: what specific action by whom triggers "
            "the denial of service?"
        ),
        corpus_doc_ids=["cve-demo-009-dos-resource-exhaustion"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric=(
            "Precondition: a malformed configuration containing a "
            "cyclic group-reference (e.g., A→B→C→A) must be applied "
            "or accepted by the Helios Gateway. Application is "
            "triggered by any valid administrator action that causes "
            "reload, or by SIGHUP to the running process. The attack "
            "is NOT a direct network packet from an unauthenticated "
            "attacker; the AV:N component reflects that the config "
            "can arrive over the network, but the precondition is "
            "that an administrator applies the attacker-supplied "
            "config (e.g., via supply-chain or social engineering). "
            "Once applied, the crash is deterministic and persists "
            "across restarts until rollback."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# C8 — Numerical / specific-fact extraction (8 tasks)
# Fast solver + light verify. Tight, specific facts.
# --------------------------------------------------------------------------- #

C8_TASKS: list[SecurityTask] = [
    SecurityTask(
        id="C8.1",
        scenario_class="C8",
        user_task=(
            "What is the earliest affected AcmeCMS version listed in "
            "CVE-DEMO-2026-001?"
        ),
        corpus_doc_ids=["cve-demo-001-buffer-overflow"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct: 7.2.0.",
    ),
    SecurityTask(
        id="C8.2",
        scenario_class="C8",
        user_task=(
            "From which LTS version onward was the comment-attachment "
            "feature backported, per CVE-DEMO-2026-003?"
        ),
        corpus_doc_ids=["cve-demo-003-xss-stored"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct: 6.5.0.",
    ),
    SecurityTask(
        id="C8.3",
        scenario_class="C8",
        user_task=(
            "What is the patched QuartzDB Server version that fixes "
            "CVE-DEMO-2026-012?"
        ),
        corpus_doc_ids=["cve-demo-012-log-injection-rce"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct: 4.3.5 (or later).",
    ),
    SecurityTask(
        id="C8.4",
        scenario_class="C8",
        user_task=(
            "Quote the exact CVSS vector string from CVE-DEMO-2026-006."
        ),
        corpus_doc_ids=["cve-demo-006-priv-esc-local"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H."
        ),
    ),
    SecurityTask(
        id="C8.5",
        scenario_class="C8",
        user_task=(
            "What specific IP address does CVE-DEMO-2026-004 cite as "
            "the cloud-metadata service endpoint reached via the SSRF?"
        ),
        corpus_doc_ids=["cve-demo-004-ssrf-cloud"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct: 169.254.169.254.",
    ),
    SecurityTask(
        id="C8.6",
        scenario_class="C8",
        user_task=(
            "What approximate per-attempt exploitation reliability "
            "does CVE-DEMO-2026-010 report on stock kernels?"
        ),
        corpus_doc_ids=["cve-demo-010-race-toctou"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct: approximately 30% per attempt.",
    ),
    SecurityTask(
        id="C8.7",
        scenario_class="C8",
        user_task=(
            "What date range does CVE-DEMO-2026-007 give as the window "
            "during which the malicious dependency was served?"
        ),
        corpus_doc_ids=["cve-demo-007-supply-chain-dep"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: 2025-05-14 through 2025-06-02 (inclusive)."
        ),
    ),
    SecurityTask(
        id="C8.8",
        scenario_class="C8",
        user_task=(
            "What approximate entropy (in bits) does CVE-DEMO-2026-008 "
            "cite for active-session tokens generated by the weak PRNG?"
        ),
        corpus_doc_ids=["cve-demo-008-crypto-weak-rng"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric=(
            "Correct: approximately 30 bits (well below the 128 bits "
            "required for cryptographically unguessable tokens)."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Combined task pool
# --------------------------------------------------------------------------- #

ALL_TASKS: list[SecurityTask] = (
    C1_TASKS + C2_TASKS + C3_TASKS + C4_TASKS
    + C5_TASKS + C6_TASKS + C7_TASKS + C8_TASKS
)


def tasks_by_class() -> dict[ScenarioClass, list[SecurityTask]]:
    out: dict[ScenarioClass, list[SecurityTask]] = {}
    for task in ALL_TASKS:
        out.setdefault(task.scenario_class, []).append(task)
    return out
