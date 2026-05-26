"""
Production-traffic task distribution for chunk 6.

60 queries across 8 scenario classes over the synthetic distributed-systems
corpus. Each class is designed to give the policy a clear coordination
decision to learn — there are tasks where the fast solver suffices and
tasks where the capable solver is required, tasks where verification adds
real value and tasks where it adds only cost, tasks the corpus answers
cleanly and tasks that require external web search.

The classes:

  C1 — Single-doc factual extraction (simple lookups, fast solver wins,
       verifier skip)
  C2 — Multi-doc synthesis (combine 2-3 docs, capable solver, verifier
       sometimes useful)
  C3 — Cross-doc consistency check (does the corpus contradict itself
       on this point? capable solver + verifier)
  C4 — Underspecified question with answer in corpus (capable solver
       qualifies the answer; verifier helps catch overreach)
  C5 — Question with no answer in corpus (capable solver + verifier
       must catch the gap; web search may help if external info exists)
  C6 — Definitional / lookup (tight, single-source, fast solver wins)
  C7 — Multi-step reasoning (chain claims across docs; capable solver
       + verifier; web search may add value)
  C8 — Numerical / specific-fact extraction (precise extraction,
       fast solver usually OK, lightweight verification useful)

Per-task fields:
  - id: stable identifier
  - scenario_class: C1..C8
  - user_task: the natural-language query
  - corpus_doc_ids: the corpus subset to expose to memory
  - corpus_has_answer: whether the corpus contains the answer (False for C5)
  - expected_optimal_variant: hint for analysis — which solver variant we
       expect the policy to converge to. Not given to the framework.
  - expected_verifier_value: "skip", "useful", "essential" — analysis hint
  - expected_web_value: "no", "complement", "essential" — analysis hint
  - notes_for_rubric: short hint passed to RULER alongside the rubric
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agensflow import Document, TaskFeatures

from experiments.e03_production_traffic.corpus import get_corpus_subset


# --------------------------------------------------------------------------- #
# Per-scenario-class TaskFeatures.
#
# These differentiate the policy graph signature by class so the policy can
# learn per-class routing. Without them, every task at the post-planner step
# would produce the same signature and the policy would collapse to a single
# average-optimal coordination across all 60 tasks.
#
# The features are designed so the rule-based regime detector produces
# *different regime labels* across classes where appropriate — C1/C6 are
# simple-enough to fall through to `straightforward`, C5 has high enough
# ambiguity to flag as `ambiguous`, the rest land in `evidence_heavy`.
# --------------------------------------------------------------------------- #

CLASS_FEATURES: dict[str, TaskFeatures] = {
    # C1 — Single-doc factual extraction. Low ambiguity, evidence present,
    # verification not needed. Falls through to straightforward.
    "C1": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.10,
        contradiction_risk=0.05,
        novelty_level=0.10,
        evidence_availability=0.85,
        verification_need=0.30,
        time_horizon_complexity=0.10,
    ),
    # C2 — Multi-doc synthesis. Evidence-heavy regime.
    "C2": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.25,
        contradiction_risk=0.10,
        novelty_level=0.30,
        evidence_availability=0.90,
        verification_need=0.70,
        time_horizon_complexity=0.40,
    ),
    # C3 — Cross-doc consistency check. Moderate contradiction risk
    # (we're explicitly asking whether docs disagree).
    "C3": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.30,
        contradiction_risk=0.50,
        novelty_level=0.20,
        evidence_availability=0.85,
        verification_need=0.80,
        time_horizon_complexity=0.40,
    ),
    # C4 — Underspecified question with answer in corpus.
    "C4": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.55,
        contradiction_risk=0.15,
        novelty_level=0.30,
        evidence_availability=0.75,
        verification_need=0.75,
        time_horizon_complexity=0.30,
    ),
    # C5 — Question with no answer in corpus. High ambiguity (we don't know
    # if the corpus has the answer until verification). Hits ambiguous regime.
    "C5": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.80,
        contradiction_risk=0.10,
        novelty_level=0.50,
        evidence_availability=0.60,
        verification_need=0.85,
        time_horizon_complexity=0.50,
    ),
    # C6 — Definitional / lookup. Even simpler than C1; pure straightforward.
    "C6": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.05,
        contradiction_risk=0.0,
        novelty_level=0.05,
        evidence_availability=0.85,
        verification_need=0.20,
        time_horizon_complexity=0.05,
    ),
    # C7 — Multi-step reasoning. Evidence-heavy with high verification need.
    "C7": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.40,
        contradiction_risk=0.20,
        novelty_level=0.50,
        evidence_availability=0.80,
        verification_need=0.85,
        time_horizon_complexity=0.70,
    ),
    # C8 — Numerical / specific-fact extraction.
    "C8": TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.10,
        contradiction_risk=0.05,
        novelty_level=0.10,
        evidence_availability=0.85,
        verification_need=0.50,
        time_horizon_complexity=0.10,
    ),
}

ScenarioClass = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
ExpectedVariant = Literal[
    "solver_fast", "solver_mini", "solver_haiku",
    # Qwen variants are registered but excluded from chunk-6 plan (see
    # activation.py for the OpenRouter tool_choice compatibility note).
    # Keeping these in the Literal so historical task records remain valid.
    "solver_qwen_flash", "solver_qwen_max",
]
VerifierValue = Literal["skip", "useful", "essential"]
WebValue = Literal["no", "complement", "essential"]


@dataclass(frozen=True)
class ProductionTask:
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
        """Per-class TaskFeatures profile (drives regime label in signature)."""
        return CLASS_FEATURES[self.scenario_class]


# --------------------------------------------------------------------------- #
# C1 — Single-doc factual extraction (8 variants)
# Simple lookups with one canonical doc. Fast solver wins, verifier adds no
# value, web search adds no value.
# --------------------------------------------------------------------------- #

C1_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C1.1",
        scenario_class="C1",
        user_task="In Paxos, what are the two main protocol phases called?",
        corpus_doc_ids=["paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: Phase 1 (Prepare/Promise) and Phase 2 (Accept/Accepted).",
    ),
    ProductionTask(
        id="C1.2",
        scenario_class="C1",
        user_task="Who designed Raft and in what year?",
        corpus_doc_ids=["raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: Ongaro and Ousterhout (2014).",
    ),
    ProductionTask(
        id="C1.3",
        scenario_class="C1",
        user_task=(
            "In two-phase commit, what is the name of the first phase and what "
            "happens during it?"
        ),
        corpus_doc_ids=["two-phase-commit"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: Phase 1 is Prepare; coordinator asks each participant whether it can commit, participants vote and durably record their vote.",
    ),
    ProductionTask(
        id="C1.4",
        scenario_class="C1",
        user_task="What does each node maintain in vector clocks?",
        corpus_doc_ids=["vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: a vector of integer counters, one per node.",
    ),
    ProductionTask(
        id="C1.5",
        scenario_class="C1",
        user_task="What is the rule for updating Lamport timestamps when receiving a message?",
        corpus_doc_ids=["lamport-timestamps"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: receiver sets its counter to max(local, received) + 1.",
    ),
    ProductionTask(
        id="C1.6",
        scenario_class="C1",
        user_task="What are CmRDTs and how do they differ from CvRDTs?",
        corpus_doc_ids=["crdts"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: CmRDTs are operation-based (replicas exchange operations that must commute); CvRDTs are state-based (replicas exchange full state with associative/commutative/idempotent join).",
    ),
    ProductionTask(
        id="C1.7",
        scenario_class="C1",
        user_task="In consistent hashing, what is the role of virtual nodes?",
        corpus_doc_ids=["consistent-hashing"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: each physical server is mapped to multiple virtual nodes on the ring to improve load balance.",
    ),
    ProductionTask(
        id="C1.8",
        scenario_class="C1",
        user_task="In the Bully algorithm, what does a node do after detecting the leader has failed?",
        corpus_doc_ids=["bully-election"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Correct: sends an Election message to all nodes with higher IDs; if none respond within a timeout, declares itself the new leader.",
    ),
]


# --------------------------------------------------------------------------- #
# C2 — Multi-doc synthesis (8 variants)
# Combine information from 2-3 docs. Capable solver wins; verifier is sometimes
# useful for catching subtle synthesis errors.
# --------------------------------------------------------------------------- #

C2_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C2.1",
        scenario_class="C2",
        user_task=(
            "Compare Paxos and Raft on (a) the consensus problem they solve, "
            "(b) how each handles leadership, and (c) the majority requirement."
        ),
        corpus_doc_ids=["paxos-basics", "raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct synthesis touches: both solve consensus among nodes; Paxos uses Multi-Paxos with stable leader, Raft uses elected leader per term; both require majority (2f+1).",
    ),
    ProductionTask(
        id="C2.2",
        scenario_class="C2",
        user_task=(
            "Both Lamport timestamps and vector clocks track event ordering. "
            "Compare their guarantees and overhead."
        ),
        corpus_doc_ids=["lamport-timestamps", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct synthesis: Lamport gives partial ordering (clock-condition only one-way), O(1) per event; vector clocks give full causality (V → V' iff V[i] ≤ V'[i] for all i, V ≠ V'), O(N) per event.",
    ),
    ProductionTask(
        id="C2.3",
        scenario_class="C2",
        user_task=(
            "Compare CRDTs and gossip protocols: how do they relate, and how "
            "are they often used together in eventually consistent stores?"
        ),
        corpus_doc_ids=["crdts", "gossip-protocols", "eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Correct synthesis touches: gossip provides anti-entropy reconciliation; CRDTs provide a data structure that converges under independent updates; both are used in eventually consistent stores like Cassandra/Dynamo/Riak.",
    ),
    ProductionTask(
        id="C2.4",
        scenario_class="C2",
        user_task=(
            "How do consistent hashing and gossip protocols complement each "
            "other in distributed databases like Cassandra?"
        ),
        corpus_doc_ids=["consistent-hashing", "gossip-protocols"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Consistent hashing places keys on a ring across servers; gossip propagates membership/state. Both are used in Cassandra: ring for data placement, gossip for cluster membership.",
    ),
    ProductionTask(
        id="C2.5",
        scenario_class="C2",
        user_task=(
            "Compare the Bully algorithm and Raft's leader election: what "
            "assumptions does each make and what are the tradeoffs?"
        ),
        corpus_doc_ids=["bully-election", "raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Bully: ordered IDs, reliable delivery, total membership knowledge, O(N²) messages worst case. Raft: terms, randomized timeouts to reduce split votes, majority-based.",
    ),
    ProductionTask(
        id="C2.6",
        scenario_class="C2",
        user_task=(
            "Compare BFT consensus and crash-fault-tolerant consensus (Paxos / "
            "Raft) on (a) failure model assumed and (b) replica count required."
        ),
        corpus_doc_ids=["bft-overview", "paxos-basics", "raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="BFT tolerates arbitrary/malicious failures, requires 3f+1 nodes for f failures. Paxos/Raft tolerate crash failures only, require 2f+1.",
    ),
    ProductionTask(
        id="C2.7",
        scenario_class="C2",
        user_task=(
            "How does the FLP impossibility result relate to Paxos's safety / "
            "liveness guarantees?"
        ),
        corpus_doc_ids=["paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="complement",
        notes_for_rubric="Paxos guarantees safety but not liveness; FLP says competing proposers can prevent progress indefinitely. Web search may add depth on FLP context.",
    ),
    ProductionTask(
        id="C2.8",
        scenario_class="C2",
        user_task=(
            "Compare two-phase commit and three-phase commit on blocking "
            "behavior and practical use."
        ),
        corpus_doc_ids=["two-phase-commit"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_mini",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="2PC blocks if coordinator fails after participants vote yes but before broadcast. 3PC adds precommit phase to address blocking but is rarely used in practice due to additional complexity.",
    ),
]


# --------------------------------------------------------------------------- #
# C3 — Cross-doc consistency check (6 variants)
# Tests whether the corpus contradicts itself or aligns. Capable solver +
# verifier is essential because the failure mode is a confident-but-wrong
# claim that the corpus does or doesn't agree.
# --------------------------------------------------------------------------- #

C3_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C3.1",
        scenario_class="C3",
        user_task=(
            "Do the Paxos and Raft documents agree on the minimum node count "
            "required for the protocol to make progress? If they disagree, "
            "where?"
        ),
        corpus_doc_ids=["paxos-basics", "raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="They agree: both require majority of nodes available for progress. Correct answer notes the agreement explicitly.",
    ),
    ProductionTask(
        id="C3.2",
        scenario_class="C3",
        user_task=(
            "The vector clocks document and the Lamport timestamps document "
            "both discuss event ordering. Do they make consistent claims "
            "about what each mechanism guarantees?"
        ),
        corpus_doc_ids=["lamport-timestamps", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Yes, consistent: Lamport gives one-way clock-condition (causal precedence implies timestamp ordering), vector clocks give bidirectional (timestamp ordering iff causal precedence). Documents are aligned.",
    ),
    ProductionTask(
        id="C3.3",
        scenario_class="C3",
        user_task=(
            "Both the BFT and the Paxos documents discuss fault tolerance. "
            "Are the failure models they assume consistent or in tension?"
        ),
        corpus_doc_ids=["bft-overview", "paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Different failure models: Paxos assumes crash failures (2f+1); BFT assumes Byzantine/arbitrary failures (3f+1). Not contradictory but distinct assumptions.",
    ),
    ProductionTask(
        id="C3.4",
        scenario_class="C3",
        user_task=(
            "The CRDTs and eventual consistency documents both discuss "
            "Cassandra. Do they describe Cassandra's conflict resolution "
            "consistently?"
        ),
        corpus_doc_ids=["crdts", "eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Eventual consistency doc lists multiple conflict resolution mechanisms (LWW, vector clocks, CRDTs, app-level). CRDT doc describes the data-structure-level mechanism. Consistent — CRDTs are one option among several.",
    ),
    ProductionTask(
        id="C3.5",
        scenario_class="C3",
        user_task=(
            "Do the gossip and consistent hashing documents agree on which "
            "systems use these techniques?"
        ),
        corpus_doc_ids=["gossip-protocols", "consistent-hashing"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Both mention Cassandra. Consistent hashing also mentions Dynamo/Riak. Gossip mentions Cassandra/Consul/SWIM. Overlapping but not contradictory.",
    ),
    ProductionTask(
        id="C3.6",
        scenario_class="C3",
        user_task=(
            "Are the Chandy-Lamport snapshot and the vector clocks documents "
            "compatible in their views of distributed-system causality?"
        ),
        corpus_doc_ids=["chandy-lamport-snapshot", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Yes, compatible. Chandy-Lamport produces a causally consistent global state; vector clocks define causal precedence between events. Both rooted in the same causal-ordering framework.",
    ),
]


# --------------------------------------------------------------------------- #
# C4 — Underspecified question with answer in corpus (6 variants)
# Question is broad; corpus has partial answers. Capable solver qualifies
# the answer; verifier helps catch overreach (claiming more than the corpus
# supports).
# --------------------------------------------------------------------------- #

C4_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C4.1",
        scenario_class="C4",
        user_task="What is the best consensus protocol for distributed systems?",
        corpus_doc_ids=["paxos-basics", "raft-overview", "bft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Question is underspecified — 'best' depends on failure model and operational requirements. Correct answer qualifies: Raft is more understandable than Paxos; BFT is needed for Byzantine failures; tradeoffs depend on context.",
    ),
    ProductionTask(
        id="C4.2",
        scenario_class="C4",
        user_task="When should I use eventual consistency vs. strong consistency?",
        corpus_doc_ids=["eventual-consistency", "paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="complement",
        notes_for_rubric="Tradeoffs from CAP theorem: eventual for high availability and partition tolerance; strong (consensus-based) for invariant correctness. Correct answer cites CAP and gives concrete examples.",
    ),
    ProductionTask(
        id="C4.3",
        scenario_class="C4",
        user_task="How should I detect node failures in a distributed system?",
        corpus_doc_ids=["gossip-protocols", "bully-election"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="complement",
        notes_for_rubric="Corpus mentions SWIM-style gossip-based failure detection. Correct answer cites gossip protocols and notes Bully's reliance on detection. Web search may add depth on phi-accrual or other detectors.",
    ),
    ProductionTask(
        id="C4.4",
        scenario_class="C4",
        user_task="What's the right approach to global state in a distributed system?",
        corpus_doc_ids=["chandy-lamport-snapshot", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Correct answer cites Chandy-Lamport for consistent snapshots and vector clocks for causality, and qualifies that 'global state' is fundamentally a partial-ordering question.",
    ),
    ProductionTask(
        id="C4.5",
        scenario_class="C4",
        user_task="How do I handle conflicts in a replicated key-value store?",
        corpus_doc_ids=["crdts", "eventual-consistency", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Multiple options: LWW, vector clocks, CRDTs, application-level reconciliation. Tradeoff between automation and correctness.",
    ),
    ProductionTask(
        id="C4.6",
        scenario_class="C4",
        user_task="What ordering guarantees should I expect from a distributed log?",
        corpus_doc_ids=["raft-overview", "lamport-timestamps", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Raft provides total ordering of committed entries via leader. Without leader-based consensus, options are partial ordering (Lamport) or causal ordering (vector clocks).",
    ),
]


# --------------------------------------------------------------------------- #
# C5 — Question with no answer in corpus (6 variants)
# Verifier is essential — must catch the gap. Web search may help if the
# external internet has the answer; otherwise the policy should refuse.
# --------------------------------------------------------------------------- #

C5_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C5.1",
        scenario_class="C5",
        user_task=(
            "What is the specific snapshot mechanism Raft uses for log "
            "compaction, and what are its memory tradeoffs?"
        ),
        corpus_doc_ids=["raft-overview"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="essential",
        notes_for_rubric="Corpus does not describe Raft's snapshot mechanism. Correct answer either flags the gap explicitly OR retrieves accurate information via web search. Confabulation (inventing snapshot details from training data) is failure.",
    ),
    ProductionTask(
        id="C5.2",
        scenario_class="C5",
        user_task=(
            "How does PBFT specifically handle view changes during leader "
            "failure?"
        ),
        corpus_doc_ids=["bft-overview"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="essential",
        notes_for_rubric="Corpus mentions BFT in general but not PBFT specifics. Correct answer flags gap or web-searches; confabulating PBFT mechanism details is failure.",
    ),
    ProductionTask(
        id="C5.3",
        scenario_class="C5",
        user_task=(
            "What is the standard garbage collection strategy for OR-Set "
            "CRDTs in long-running systems?"
        ),
        corpus_doc_ids=["crdts"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="essential",
        notes_for_rubric="Corpus mentions OR-Set but not garbage collection. Correct answer flags gap; web search may surface real CRDT GC literature.",
    ),
    ProductionTask(
        id="C5.4",
        scenario_class="C5",
        user_task=(
            "What is the Paxos message complexity for cross-datacenter "
            "deployments with WAN latency optimization?"
        ),
        corpus_doc_ids=["paxos-basics"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="essential",
        notes_for_rubric="Corpus does not address cross-datacenter Paxos. Correct answer flags the gap.",
    ),
    ProductionTask(
        id="C5.5",
        scenario_class="C5",
        user_task=(
            "What was the carbon footprint of the Cassandra cluster used "
            "in the original gossip-protocol benchmarks?"
        ),
        corpus_doc_ids=["gossip-protocols"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Corpus does not mention any benchmarks or carbon footprint. Correct answer flags the gap and refuses to invent figures.",
    ),
    ProductionTask(
        id="C5.6",
        scenario_class="C5",
        user_task=(
            "What is the elliptic-curve cryptography library most commonly "
            "used in BFT implementations?"
        ),
        corpus_doc_ids=["bft-overview"],
        corpus_has_answer=False,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Corpus does not address cryptographic libraries. Correct answer flags the gap.",
    ),
]


# --------------------------------------------------------------------------- #
# C6 — Definitional / lookup (8 variants)
# Tight definitions, fast solver wins, verifier skip, no web search.
# --------------------------------------------------------------------------- #

C6_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C6.1",
        scenario_class="C6",
        user_task="Define 'eventual consistency' using the corpus.",
        corpus_doc_ids=["eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Direct definition: a consistency model that guarantees if no new updates are made, eventually all accesses to a data item return the last updated value.",
    ),
    ProductionTask(
        id="C6.2",
        scenario_class="C6",
        user_task="Define 'gossip protocol' using the corpus.",
        corpus_doc_ids=["gossip-protocols"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Definition: distributed information-spreading by periodic random-peer state exchange; epidemic.",
    ),
    ProductionTask(
        id="C6.3",
        scenario_class="C6",
        user_task="Define 'CRDT'.",
        corpus_doc_ids=["crdts"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Definition: Conflict-free Replicated Data Type — replicated structure that converges without coordination.",
    ),
    ProductionTask(
        id="C6.4",
        scenario_class="C6",
        user_task="What is the 'Byzantine Generals Problem'?",
        corpus_doc_ids=["bft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Foundational problem (Lamport, Shostak, Pease 1982) establishing the lower bound for tolerating Byzantine failures.",
    ),
    ProductionTask(
        id="C6.5",
        scenario_class="C6",
        user_task="Define 'two-phase commit'.",
        corpus_doc_ids=["two-phase-commit"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Definition: distributed transaction protocol with Prepare phase (coordinator queries participants) and Commit phase (broadcasts decision based on votes).",
    ),
    ProductionTask(
        id="C6.6",
        scenario_class="C6",
        user_task="Define 'consistent hashing'.",
        corpus_doc_ids=["consistent-hashing"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Definition: technique for distributing keys across servers such that adding/removing a server moves only K/N keys on average.",
    ),
    ProductionTask(
        id="C6.7",
        scenario_class="C6",
        user_task="What is the 'CAP theorem'?",
        corpus_doc_ids=["eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Brewer 2000: in a network partition, a system must choose between consistency and availability.",
    ),
    ProductionTask(
        id="C6.8",
        scenario_class="C6",
        user_task="Define 'leader election'.",
        corpus_doc_ids=["bully-election", "raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="skip",
        expected_web_value="no",
        notes_for_rubric="Process by which distributed nodes pick one node to act as coordinator/leader.",
    ),
]


# --------------------------------------------------------------------------- #
# C7 — Multi-step reasoning (10 variants)
# Chain claims across docs to answer a derived question. Capable solver
# essential; verifier essential; web search may add depth.
# --------------------------------------------------------------------------- #

C7_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C7.1",
        scenario_class="C7",
        user_task=(
            "If I'm building a system that needs Byzantine fault tolerance "
            "and 99% availability across geo-distributed datacenters, what "
            "node count and consistency model do I need at minimum, and what "
            "tradeoff does that imply?"
        ),
        corpus_doc_ids=[
            "bft-overview", "eventual-consistency", "paxos-basics",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Reasoning: BFT needs 3f+1 nodes for f Byzantine; geo-distributed implies CAP-bound tradeoff with consistency vs availability under partitions; for 99% availability, eventual consistency may be needed unless quorum-based BFT can be tuned.",
    ),
    ProductionTask(
        id="C7.2",
        scenario_class="C7",
        user_task=(
            "I have a key-value store using consistent hashing with 5 nodes. "
            "If I need to detect failed nodes and elect a coordinator without "
            "a fixed leader, which two corpus techniques would I combine, "
            "and why?"
        ),
        corpus_doc_ids=[
            "consistent-hashing", "gossip-protocols", "bully-election",
        ],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Reasoning: gossip for failure detection (e.g., SWIM-style); bully (or another election protocol) for coordinator selection. Combine because gossip detects, election decides.",
    ),
    ProductionTask(
        id="C7.3",
        scenario_class="C7",
        user_task=(
            "Why might a system using LWW-Register CRDTs with Lamport "
            "timestamps still produce surprising results in cross-region "
            "replication?"
        ),
        corpus_doc_ids=["crdts", "lamport-timestamps", "vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Reasoning: Lamport timestamps give partial ordering only — concurrent events with timestamp(a) < timestamp(b) may not be causally ordered. LWW resolves by timestamp, so concurrent writes appear ordered when they aren't, masking real concurrency.",
    ),
    ProductionTask(
        id="C7.4",
        scenario_class="C7",
        user_task=(
            "If a system wants both safety and liveness for consensus, can "
            "it use Paxos? Why or why not, given the FLP result?"
        ),
        corpus_doc_ids=["paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="Paxos guarantees safety always but liveness is FLP-limited under arbitrary asynchrony. In practice systems use timeouts and stable-leader optimization (Multi-Paxos) to achieve liveness in normal operation.",
    ),
    ProductionTask(
        id="C7.5",
        scenario_class="C7",
        user_task=(
            "If I'm building a distributed checkpoint mechanism for a system "
            "with network channels, which corpus algorithm should I use, and "
            "what guarantees does it give about the resulting snapshot?"
        ),
        corpus_doc_ids=["chandy-lamport-snapshot"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Chandy-Lamport snapshot algorithm. Resulting snapshot is causally consistent — captures a global state that could have existed during execution.",
    ),
    ProductionTask(
        id="C7.6",
        scenario_class="C7",
        user_task=(
            "Compare 2PC and Paxos for distributed agreement: which provides "
            "stronger fault tolerance, and what's the tradeoff?"
        ),
        corpus_doc_ids=["two-phase-commit", "paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Paxos tolerates minority failures and continues to make progress; 2PC blocks if coordinator fails. Tradeoff: 2PC is simpler when you trust the coordinator and need ACID across resources.",
    ),
    ProductionTask(
        id="C7.7",
        scenario_class="C7",
        user_task=(
            "An eventually consistent system uses gossip protocols for state "
            "propagation. Given typical gossip latency, what's the implication "
            "for an application reading 'fresh' data?"
        ),
        corpus_doc_ids=["gossip-protocols", "eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="complement",
        notes_for_rubric="Gossip is O(log N) rounds for full propagation. Application may see stale data until propagation completes. Eventual-consistency contract acknowledges this — strong-read applications need separate quorum reads.",
    ),
    ProductionTask(
        id="C7.8",
        scenario_class="C7",
        user_task=(
            "If I have a 7-node Raft cluster and 2 nodes fail, can the "
            "system continue to make progress? Justify from the corpus."
        ),
        corpus_doc_ids=["raft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="Yes. Raft requires majority (4 of 7); 7 - 2 = 5 ≥ 4. Reasoning: majority of nodes still available, so log replication can commit.",
    ),
    ProductionTask(
        id="C7.9",
        scenario_class="C7",
        user_task=(
            "Could vector clocks be used to detect Byzantine behavior in a "
            "consensus system? Why or why not?"
        ),
        corpus_doc_ids=["vector-clocks", "bft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="complement",
        notes_for_rubric="No. Vector clocks track causality of correct messages but cannot detect arbitrary/malicious behavior — a Byzantine node could send well-formed-looking but contradictory messages with valid-looking clocks.",
    ),
    ProductionTask(
        id="C7.10",
        scenario_class="C7",
        user_task=(
            "If a 2PC coordinator fails, what's the worst-case state that "
            "the participants can be left in, and why does that motivate "
            "consensus-based alternatives?"
        ),
        corpus_doc_ids=["two-phase-commit", "paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_haiku",
        expected_verifier_value="essential",
        expected_web_value="no",
        notes_for_rubric="Worst case: participants voted yes and are blocked waiting for the decision. Consensus alternatives (Paxos, Raft) tolerate the leader failure by electing a new one and continuing.",
    ),
]


# --------------------------------------------------------------------------- #
# C8 — Numerical / specific-fact extraction (8 variants)
# Precise extraction of a specific number, name, or year. Fast solver
# usually OK, lightweight verification useful (catches hallucinated specifics).
# --------------------------------------------------------------------------- #

C8_TASKS: list[ProductionTask] = [
    ProductionTask(
        id="C8.1",
        scenario_class="C8",
        user_task="What is the worst-case message complexity of the Bully algorithm?",
        corpus_doc_ids=["bully-election"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="O(N²).",
    ),
    ProductionTask(
        id="C8.2",
        scenario_class="C8",
        user_task="In what year was the Byzantine Generals Problem formalized, and by whom?",
        corpus_doc_ids=["bft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="1982, by Lamport, Shostak, and Pease.",
    ),
    ProductionTask(
        id="C8.3",
        scenario_class="C8",
        user_task="In what year was Paxos popularized, and by whom?",
        corpus_doc_ids=["paxos-basics"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="1998, by Lamport (introduced 1989).",
    ),
    ProductionTask(
        id="C8.4",
        scenario_class="C8",
        user_task="What is the per-event message overhead of vector clocks for N nodes?",
        corpus_doc_ids=["vector-clocks"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="O(N).",
    ),
    ProductionTask(
        id="C8.5",
        scenario_class="C8",
        user_task="What is the propagation time of a gossip protocol for N nodes (with high probability)?",
        corpus_doc_ids=["gossip-protocols"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="O(log N) rounds.",
    ),
    ProductionTask(
        id="C8.6",
        scenario_class="C8",
        user_task="In what year was the Chandy-Lamport snapshot algorithm published?",
        corpus_doc_ids=["chandy-lamport-snapshot"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="1985.",
    ),
    ProductionTask(
        id="C8.7",
        scenario_class="C8",
        user_task="What is the minimum number of nodes needed in a BFT system to tolerate 1 Byzantine failure?",
        corpus_doc_ids=["bft-overview"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="4 (3f+1 with f=1).",
    ),
    ProductionTask(
        id="C8.8",
        scenario_class="C8",
        user_task="In what year was the CAP theorem formulated and by whom?",
        corpus_doc_ids=["eventual-consistency"],
        corpus_has_answer=True,
        expected_optimal_variant="solver_fast",
        expected_verifier_value="useful",
        expected_web_value="no",
        notes_for_rubric="2000, by Brewer.",
    ),
]


# --------------------------------------------------------------------------- #
# Aggregated task list
# --------------------------------------------------------------------------- #

ALL_TASKS: list[ProductionTask] = (
    C1_TASKS + C2_TASKS + C3_TASKS + C4_TASKS
    + C5_TASKS + C6_TASKS + C7_TASKS + C8_TASKS
)


def tasks_by_class() -> dict[ScenarioClass, list[ProductionTask]]:
    out: dict[ScenarioClass, list[ProductionTask]] = {
        c: [] for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    }
    for t in ALL_TASKS:
        out[t.scenario_class].append(t)
    return out
