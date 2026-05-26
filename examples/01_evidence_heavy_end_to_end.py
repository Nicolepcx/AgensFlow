"""
End-to-end evidence_heavy run.

What this demonstrates:
  1. Task features describe the task as evidence-heavy.
  2. The activation planner produces the matching coalition
     (planner -> memory -> solver -> verifier -> evaluator) with verifier_gate
     as the merge strategy.
  3. The runtime builds a LangGraph from the plan and executes it with real
     LLM calls via OpenRouter.
  4. Each agent updates only its portion of the structured Handoff and records
     which upstream fields it consumed.
  5. The evaluator decides whether the run is complete and produces the
     user-facing answer.
  6. The trace is printed at the end, showing per-agent token cost and
     latency.

How to run:
  1. Put OPENROUTER_API_KEY=sk-or-... in a .env file at the repo root.
  2. pip install -e .  (from the repo root)
  3. python examples/01_evidence_heavy_end_to_end.py
"""

from __future__ import annotations

from agensflow import (
    Document,
    TaskFeatures,
    run,
)


def main() -> None:
    user_task = (
        "What is the difference between TCP and UDP, and when should each be used? "
        "Answer using only the provided documents."
    )

    documents = [
        Document(
            id="rfc793-summary",
            text=(
                "TCP (Transmission Control Protocol) is a connection-oriented "
                "protocol. It establishes a connection via a three-way handshake "
                "before data is exchanged. TCP guarantees that bytes are delivered "
                "in order, retransmits lost segments, and applies flow and "
                "congestion control. These guarantees add per-message overhead "
                "and latency."
            ),
        ),
        Document(
            id="rfc768-summary",
            text=(
                "UDP (User Datagram Protocol) is a connectionless protocol. It "
                "sends datagrams without establishing a connection and without "
                "delivery, ordering, or duplicate-protection guarantees. UDP "
                "headers are 8 bytes; TCP headers are at least 20 bytes."
            ),
        ),
        Document(
            id="usage-patterns",
            text=(
                "Applications that require reliable, ordered delivery such as "
                "HTTP, SMTP, and SSH use TCP. Applications that prioritise low "
                "latency over reliability such as DNS queries, real-time voice "
                "and video, online gaming, and many telemetry protocols use UDP "
                "and handle loss at the application layer if needed."
            ),
        ),
        Document(
            id="performance-notes",
            text=(
                "Because UDP avoids handshakes and acknowledgements, it has lower "
                "per-message overhead and lower tail latency than TCP. However, "
                "TCP's congestion control is essential for fair bandwidth sharing "
                "on shared networks; UDP-heavy workloads must implement their "
                "own congestion control to coexist with TCP traffic."
            ),
        ),
    ]

    # Features describing the task. evidence_availability and verification_need
    # are both high, which puts this in the evidence_heavy regime.
    features = TaskFeatures(
        requires_factual_grounding=True,
        ambiguity_level=0.2,
        contradiction_risk=0.1,
        evidence_availability=0.9,
        verification_need=0.8,
        novelty_level=0.2,
    )

    print(f"User task:\n  {user_task}\n")
    print(f"Documents provided: {len(documents)}\n")

    result = run(
        user_task=user_task,
        features=features,
        documents=documents,
    )

    print("=" * 72)
    print(f"Regime detected:  {result.plan.regime.label} "
          f"(confidence {result.plan.regime.confidence:.2f})")
    print(f"Selected skills:  {result.plan.selected_skills}")
    print(f"Merge strategy:   {result.plan.merge_strategy}")
    print()

    print("Final answer:")
    print("-" * 72)
    print(result.final_answer)
    print("-" * 72)
    print()

    print(f"Done:             {result.done}")
    print(f"Reasoning:        {result.evaluator_reasoning}")
    print()

    print(f"Verification (raw):")
    print(f"  {result.final_state.verification}")
    print()

    print("Trace:")
    print(result.trace.summary())


if __name__ == "__main__":
    main()
