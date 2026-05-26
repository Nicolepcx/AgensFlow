"""
Synthetic distributed-systems algorithms corpus for chunk 6.

12 short paper-style summaries on classic distributed-systems topics.
Deliberately synthetic (not real papers) so the experiment is fully
reproducible — anyone replicating can run identical content with no link
rot, no licensing concerns, no judge-model contamination from training data
(the synthetic version uses fictional author names and stylized claims).

The corpus is designed with specific gaps:

  - Coverage: Paxos, Raft, two-phase commit, vector clocks, gossip
    protocols, CRDTs, consistent hashing, leader election (bully algorithm),
    Lamport timestamps, Byzantine fault tolerance basics, eventual
    consistency, snapshot algorithms.

  - Deliberate gaps (so Class 5 / no-answer tasks have real ground truth):
    no documents on PBFT specifics, no documents on Raft's specific
    snapshot mechanism, no documents on CRDT garbage collection, no
    documents on cross-datacenter Paxos optimizations.

  - Text length: short (~150-300 words each) so a small corpus fits
    comfortably in the memory agent's context window.
"""

from __future__ import annotations

from agensflow import Document


CORPUS: list[Document] = [
    Document(
        id="paxos-basics",
        text=(
            "Paxos is a consensus protocol introduced by Lamport (1989, "
            "popularized 1998). It allows a distributed group of nodes to "
            "agree on a single value despite node failures, as long as a "
            "majority remain available. The protocol runs in two phases: "
            "Phase 1 (Prepare/Promise) where a Proposer requests acceptance "
            "rights from Acceptors, and Phase 2 (Accept/Accepted) where the "
            "Proposer broadcasts a value once it has a majority of promises. "
            "Paxos guarantees safety (only one value is decided) but does not "
            "guarantee liveness — competing proposers can prevent progress "
            "indefinitely (the FLP impossibility result). Multi-Paxos extends "
            "the basic protocol for replicated logs by establishing a stable "
            "leader to amortize the Prepare phase across many decisions."
        ),
    ),
    Document(
        id="raft-overview",
        text=(
            "Raft is a consensus protocol designed by Ongaro and Ousterhout "
            "(2014) as an understandable alternative to Paxos. It decomposes "
            "consensus into three sub-problems: leader election, log "
            "replication, and safety. Time is divided into terms; in each "
            "term, at most one leader is elected. The leader receives client "
            "requests, appends them to its log, and replicates the entries "
            "to followers. An entry is committed when it is replicated to a "
            "majority of nodes. Raft uses randomized election timeouts to "
            "reduce split votes. Like Paxos, Raft requires a majority of "
            "nodes to be available for progress."
        ),
    ),
    Document(
        id="two-phase-commit",
        text=(
            "Two-phase commit (2PC) is a distributed transaction protocol "
            "that ensures atomic commitment across multiple participants. "
            "Phase 1 (Prepare): a coordinator asks each participant whether "
            "it can commit; participants vote yes or no and durably record "
            "their vote. Phase 2 (Commit/Abort): if all participants voted "
            "yes, the coordinator broadcasts commit; otherwise it broadcasts "
            "abort. 2PC is a blocking protocol — if the coordinator fails "
            "after participants have voted yes but before broadcasting the "
            "decision, participants must wait for the coordinator's "
            "recovery. Three-phase commit (3PC) adds a precommit phase to "
            "address blocking but introduces additional complexity and is "
            "rarely used in practice."
        ),
    ),
    Document(
        id="vector-clocks",
        text=(
            "Vector clocks are a mechanism for capturing causality between "
            "events in a distributed system. Each node maintains a vector "
            "of integer counters, one per node. On any local event the node "
            "increments its own counter; when sending a message, it attaches "
            "its current vector; on receiving, the receiver takes the "
            "elementwise maximum of its vector and the received vector, then "
            "increments its own counter. Two vector clocks V and V' satisfy "
            "V → V' (V causally precedes V') if and only if V[i] ≤ V'[i] for "
            "all i and V ≠ V'. Vector clocks are O(N) per event in the "
            "number of nodes N, which is the main practical limitation."
        ),
    ),
    Document(
        id="lamport-timestamps",
        text=(
            "Lamport timestamps provide a partial ordering of events in a "
            "distributed system using a single integer counter per node. "
            "Each node increments its counter on every local event; on "
            "sending a message, it attaches the current value; on receiving, "
            "the receiver sets its counter to max(local, received) + 1. "
            "Lamport timestamps satisfy the clock-condition: if event a "
            "causally precedes event b, then timestamp(a) < timestamp(b). "
            "However, the converse does not hold — two events with "
            "timestamp(a) < timestamp(b) may be concurrent. For full "
            "causality detection, vector clocks are required."
        ),
    ),
    Document(
        id="gossip-protocols",
        text=(
            "Gossip (epidemic) protocols spread information through a "
            "distributed system by having each node periodically exchange "
            "state with a small random subset of peers. Convergence is "
            "exponential — for N nodes, full propagation takes O(log N) "
            "rounds with high probability. Gossip is robust to node "
            "failures and network partitions because it does not depend on "
            "any specific topology. Common applications include cluster "
            "membership (Cassandra, Consul), failure detection (SWIM), and "
            "anti-entropy reconciliation in eventually consistent stores. "
            "The main tradeoff is latency: gossip provides eventual rather "
            "than synchronous propagation."
        ),
    ),
    Document(
        id="crdts",
        text=(
            "Conflict-free Replicated Data Types (CRDTs) are data structures "
            "designed for replicated systems that can be updated independently "
            "and concurrently without coordination, with replicas guaranteed "
            "to converge to the same value. Two main families exist: state-"
            "based CRDTs (CvRDTs), where replicas exchange full state and use "
            "a join operation that is associative, commutative, and "
            "idempotent; and operation-based CRDTs (CmRDTs), where replicas "
            "exchange operations that must commute. Common CRDTs include "
            "G-Counter (grow-only counter), PN-Counter (positive-negative), "
            "OR-Set (observed-remove set), and LWW-Register (last-writer-"
            "wins register)."
        ),
    ),
    Document(
        id="consistent-hashing",
        text=(
            "Consistent hashing is a technique for distributing keys across "
            "a dynamic set of servers such that adding or removing a server "
            "moves only K/N keys on average (where K is the total number of "
            "keys and N is the number of servers), rather than nearly all "
            "keys as naive modulo hashing would. The standard formulation "
            "places servers on a hash ring; each key is assigned to the next "
            "server clockwise from its hash. To improve load balance, each "
            "physical server is mapped to multiple virtual nodes on the ring. "
            "Consistent hashing is foundational to systems like Dynamo, "
            "Cassandra, Riak, and most modern distributed caches."
        ),
    ),
    Document(
        id="bully-election",
        text=(
            "The Bully algorithm (Garcia-Molina, 1982) is a leader election "
            "protocol for distributed systems with synchronous, ordered "
            "node identifiers. When a node detects the current leader has "
            "failed, it sends an Election message to all nodes with higher "
            "IDs. If none respond within a timeout, it declares itself the "
            "new leader. If a higher-ID node responds, that node takes over "
            "the election. The algorithm requires O(N²) messages in the "
            "worst case but completes in O(1) rounds when the highest-ID "
            "live node detects the failure first. Bully assumes reliable "
            "message delivery and total knowledge of node membership."
        ),
    ),
    Document(
        id="bft-overview",
        text=(
            "Byzantine Fault Tolerance (BFT) refers to consensus protocols "
            "that tolerate arbitrary (Byzantine) failures, including nodes "
            "that send conflicting messages or act maliciously. BFT requires "
            "at least 3f+1 nodes to tolerate f Byzantine failures, in "
            "contrast to crash-fault-tolerant protocols (Paxos, Raft) which "
            "require only 2f+1. The Byzantine Generals Problem, formalized "
            "by Lamport, Shostak, and Pease (1982), establishes the lower "
            "bound. BFT protocols are typically more complex and have higher "
            "message complexity than crash-fault-tolerant protocols. Modern "
            "BFT systems are widely used in blockchain consensus."
        ),
    ),
    Document(
        id="eventual-consistency",
        text=(
            "Eventual consistency is a consistency model that guarantees if "
            "no new updates are made to a given data item, eventually all "
            "accesses to that item will return the last updated value. It "
            "trades off strong consistency for higher availability and "
            "lower latency, particularly across geographically distributed "
            "deployments. The CAP theorem (Brewer, 2000) formalizes the "
            "tradeoff: in the presence of a network partition, a system "
            "must choose between consistency and availability. Eventually "
            "consistent systems include Amazon Dynamo, Apache Cassandra, "
            "and Riak. Conflict resolution mechanisms vary: last-writer-wins, "
            "vector clocks, CRDTs, or application-level reconciliation."
        ),
    ),
    Document(
        id="chandy-lamport-snapshot",
        text=(
            "The Chandy-Lamport algorithm (1985) records a consistent global "
            "snapshot of a distributed system without halting computation. "
            "An initiator node records its own state and sends a marker on "
            "each outgoing channel. When a node first receives a marker on "
            "any channel, it records its own state, records that channel as "
            "empty, and propagates markers on its outgoing channels. For "
            "subsequent markers, it records the channel state as the messages "
            "received between recording its own state and receiving the "
            "marker. The resulting global state is causally consistent — "
            "useful for checkpointing, deadlock detection, and termination "
            "detection."
        ),
    ),
]


def get_corpus_subset(doc_ids: list[str]) -> list[Document]:
    """Return the subset of corpus documents matching the given ids."""
    by_id = {d.id: d for d in CORPUS}
    return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]
