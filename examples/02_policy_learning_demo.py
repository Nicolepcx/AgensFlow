"""
Layer 1 policy-learning demo.

Runs the same task K times in sequence, threading a single PolicyGraph through
all runs. After each run, prints how the graph grew and how the value
estimates accumulated.

This demonstrates the *learning substrate* of AgensFlow:
  - Each run reconstructs its (signature, action) path from the trace.
  - The reward is computed and backpropagated through the visited nodes.
  - Subsequent runs see the accumulated value estimates and *could* use them
    to inform routing (active influence is chunk 4.5; chunk 4 ships the
    substrate).

What this demo deliberately does NOT yet show:
  - The graph's recommendations overriding the rule-based plan at routing
    time. That integration ships next.
  - Value estimates *changing the answer* of a future run. The same answer
    is produced each time; what changes is the graph's confidence in
    routing decisions.

How to run:
  python examples/02_policy_learning_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agensflow import (
    Document,
    PolicyGraph,
    TaskFeatures,
    load_policy_graph,
    run,
    save_policy_graph,
)


DEFAULT_PERSIST_PATH = Path(__file__).parent / "policy_graph.demo.pkl"


def make_task() -> tuple[str, list[Document], TaskFeatures]:
    user_task = (
        "Using only the provided documents, summarise the differences between "
        "TCP and UDP and give one example of when each is appropriate."
    )
    documents = [
        Document(
            id="rfc793-summary",
            text=(
                "TCP is a connection-oriented protocol. It establishes a "
                "connection via a three-way handshake before data is "
                "exchanged. TCP guarantees that bytes are delivered in order, "
                "retransmits lost segments, and applies flow and congestion "
                "control."
            ),
        ),
        Document(
            id="rfc768-summary",
            text=(
                "UDP is connectionless. It sends datagrams without "
                "establishing a connection and without delivery, ordering, "
                "or duplicate-protection guarantees."
            ),
        ),
        Document(
            id="usage-patterns",
            text=(
                "Applications that require reliable, ordered delivery such "
                "as HTTP, SMTP, and SSH use TCP. Applications that "
                "prioritise low latency over reliability such as DNS, "
                "real-time voice and video, and online gaming use UDP."
            ),
        ),
    ]
    features = TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.2,
        contradiction_risk=0.1,
        evidence_availability=0.9,
        verification_need=0.7,
        novelty_level=0.3,
    )
    return user_task, documents, features


def _print_graph_state(graph: PolicyGraph, run_index: int) -> None:
    s = graph.stats()
    print(
        f"  graph after run {run_index + 1}: "
        f"nodes={s['n_nodes']:>3d}  "
        f"visits={s['total_visits']:>3d}  "
        f"edges={s['total_edges']:>3d}  "
        f"confident_nodes={s['confident_nodes']:>3d}"
    )


def _print_top_actions_at_initial_signature(graph: PolicyGraph) -> None:
    """
    Find the most-visited node and print its per-action statistics.

    Useful for seeing how value estimates evolve. The "initial signature" is
    typically the most-visited because every run starts there.
    """
    if not graph.nodes:
        return
    top_node = max(graph.nodes.values(), key=lambda n: n.visits)
    if top_node.visits == 0:
        return
    print(f"  most-visited node ({top_node.visits} visits, "
          f"mean reward={top_node.value:+.3f}):")
    for action, n_visits in sorted(top_node.action_visits.items(),
                                   key=lambda x: -x[1]):
        v = top_node.action_value(action)
        print(f"    action={action:<12s}  visits={n_visits:>2d}  "
              f"mean_reward={v:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the same task K times to demonstrate policy graph learning."
    )
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of consecutive runs (default: 3).")
    parser.add_argument("--persist", action="store_true",
                        help="Persist the graph to disk between runs.")
    parser.add_argument("--persist-path", type=str, default=str(DEFAULT_PERSIST_PATH),
                        help="Where to save/load the graph.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the persisted graph before starting.")
    args = parser.parse_args()

    persist_path = Path(args.persist_path)

    if args.reset and persist_path.exists():
        persist_path.unlink()
        print(f"Reset: deleted {persist_path}")

    if args.persist and persist_path.exists():
        graph = load_policy_graph(persist_path)
        print(f"Loaded existing graph from {persist_path} ({len(graph)} nodes)")
    else:
        graph = PolicyGraph()
        print("Starting with an empty policy graph")

    user_task, documents, features = make_task()

    print(f"\nRunning the same task {args.runs} time(s) with shared policy graph.\n")

    rewards: list[float] = []
    for i in range(args.runs):
        print(f"=== Run {i + 1}/{args.runs} ===")
        result = run(
            user_task=user_task,
            features=features,
            documents=documents,
            policy_graph=graph,
        )
        print(f"  regime={result.plan.regime.label}  "
              f"done={result.done}  "
              f"reward={result.reward:+.3f}  "
              f"tokens={result.total_tokens}  "
              f"calls={len(result.trace.events)}")
        if result.policy_path:
            print(f"  path: {[a for _, a in result.policy_path]}")
        _print_graph_state(graph, i)
        if result.reward is not None:
            rewards.append(result.reward)
        print()

    print("=" * 60)
    print("Summary across runs:")
    if rewards:
        print(f"  rewards: {[f'{r:+.3f}' for r in rewards]}")
        print(f"  mean reward: {sum(rewards) / len(rewards):+.3f}")
    print()
    _print_top_actions_at_initial_signature(graph)

    if args.persist:
        save_policy_graph(graph, persist_path)
        print(f"\nSaved graph to {persist_path} ({len(graph)} nodes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
