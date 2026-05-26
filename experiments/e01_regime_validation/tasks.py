"""
Benchmark tasks for experiment 01.

12 tasks across 3 categories. Each task carries:
  - id: stable identifier
  - category: A | B | C (see README)
  - user_task: what the user is asking
  - documents: zero or more Document objects (Category A: empty)
  - features: TaskFeatures used both for `agensflow_auto` and as ground-truth
              for what the right regime is
  - ground_truth_answer: short canonical answer for graders
  - grading_notes: rubric hints specific to this task

The tasks are written to test the policy, not the LLM. Category A tasks have
unambiguous answers. Category B tasks have answers fully contained in the
provided documents. Category C tasks deliberately have documents that do NOT
contain the answer — the system is supposed to detect and flag this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agensflow import Document, TaskFeatures

Category = Literal["A", "B", "C"]


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: Category
    user_task: str
    documents: list[Document] = field(default_factory=list)
    features: TaskFeatures = field(default_factory=TaskFeatures)
    ground_truth_answer: str = ""
    grading_notes: str = ""


# --------------------------------------------------------------------------- #
# Category A — Simple Q&A, no evidence needed.
# Should map to `straightforward`. Running evidence_heavy here is overkill.
# --------------------------------------------------------------------------- #

CATEGORY_A_FEATURES = TaskFeatures(
    requires_factual_grounding=False,
    ambiguity_level=0.1,
    contradiction_risk=0.0,
    novelty_level=0.1,
    evidence_availability=0.0,
    verification_need=0.2,
)


CATEGORY_A_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="A1_arithmetic",
        category="A",
        user_task="What is 17 multiplied by 23? Reply with just the number.",
        documents=[],
        features=CATEGORY_A_FEATURES,
        ground_truth_answer="391",
        grading_notes="Answer must contain '391' and no incorrect alternative.",
    ),
    BenchmarkTask(
        id="A2_unit_conversion",
        category="A",
        user_task=(
            "Convert 72 degrees Fahrenheit to Celsius. "
            "Reply with the numeric answer rounded to one decimal place."
        ),
        documents=[],
        features=CATEGORY_A_FEATURES,
        ground_truth_answer="22.2",
        grading_notes=(
            "Correct conversion is 22.222... so 22.2 is correct. "
            "Answers like 22.22 or 22.0 are also acceptable as 'partial'."
        ),
    ),
    BenchmarkTask(
        id="A3_definition",
        category="A",
        user_task=(
            "Write a one-sentence plain-English definition of recursion. "
            "Do not use the word 'recursion' itself in the definition."
        ),
        documents=[],
        features=CATEGORY_A_FEATURES,
        ground_truth_answer=(
            "A function or process that solves a problem by calling itself on a smaller version of the same problem."
        ),
        grading_notes=(
            "Success: a single sentence, captures the self-reference / "
            "self-call idea, does not use the word 'recursion'. "
            "Partial: correct idea but multi-sentence or uses the forbidden word."
        ),
    ),
    BenchmarkTask(
        id="A4_capital",
        category="A",
        user_task="What is the capital city of Australia? Reply with just the city name.",
        documents=[],
        features=CATEGORY_A_FEATURES,
        ground_truth_answer="Canberra",
        grading_notes=(
            "Success: 'Canberra'. Failure: 'Sydney' (common misconception). "
            "Partial: correct city named alongside extra qualifying text."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Category B — Document-grounded Q&A, evidence needed.
# Should map to `evidence_heavy`. Running straightforward should hallucinate
# or give weaker grounding.
# --------------------------------------------------------------------------- #

CATEGORY_B_FEATURES = TaskFeatures(
    requires_factual_grounding=True,
    ambiguity_level=0.2,
    contradiction_risk=0.1,
    novelty_level=0.3,
    evidence_availability=0.9,
    verification_need=0.7,
)


CATEGORY_B_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="B1_tcp_udp",
        category="B",
        user_task=(
            "Using only the provided documents, summarise the differences between "
            "TCP and UDP and give one example of when each is appropriate."
        ),
        documents=[
            Document(
                id="rfc793-summary",
                text=(
                    "TCP (Transmission Control Protocol) is a connection-oriented "
                    "protocol. It establishes a connection via a three-way "
                    "handshake before data is exchanged. TCP guarantees that "
                    "bytes are delivered in order, retransmits lost segments, "
                    "and applies flow and congestion control."
                ),
            ),
            Document(
                id="rfc768-summary",
                text=(
                    "UDP (User Datagram Protocol) is connectionless. It sends "
                    "datagrams without establishing a connection and without "
                    "delivery, ordering, or duplicate-protection guarantees. "
                    "UDP headers are 8 bytes; TCP headers are at least 20 bytes."
                ),
            ),
            Document(
                id="usage-patterns",
                text=(
                    "Applications that require reliable, ordered delivery such "
                    "as HTTP, SMTP, and SSH use TCP. Applications that "
                    "prioritise low latency over reliability such as DNS "
                    "queries, real-time voice and video, online gaming, and "
                    "many telemetry protocols use UDP."
                ),
            ),
        ],
        features=CATEGORY_B_FEATURES,
        ground_truth_answer=(
            "TCP is connection-oriented with reliability/ordering guarantees; "
            "UDP is connectionless without those guarantees. TCP example: HTTP. "
            "UDP example: DNS or real-time voice."
        ),
        grading_notes=(
            "Success: differences mention connection-oriented vs connectionless "
            "AND reliability/ordering AND at least one correct example per "
            "protocol. All claims must be supported by the documents."
        ),
    ),
    BenchmarkTask(
        id="B2_battery_chemistry",
        category="B",
        user_task=(
            "Using only the provided documents, compare the energy density and "
            "cycle life of lithium-ion vs. lead-acid batteries."
        ),
        documents=[
            Document(
                id="li-ion-spec",
                text=(
                    "Lithium-ion batteries deliver gravimetric energy densities "
                    "in the range of 150 to 250 Wh/kg and volumetric energy "
                    "densities of 250 to 700 Wh/L. Their typical cycle life "
                    "ranges from 1,000 to 3,000 full charge-discharge cycles "
                    "before capacity falls below 80 percent."
                ),
            ),
            Document(
                id="lead-acid-spec",
                text=(
                    "Lead-acid batteries deliver gravimetric energy densities "
                    "of approximately 30 to 50 Wh/kg and volumetric energy "
                    "densities of 60 to 110 Wh/L. Cycle life for typical "
                    "deep-cycle lead-acid is 200 to 700 cycles before "
                    "significant capacity loss."
                ),
            ),
            Document(
                id="comparison-context",
                text=(
                    "On a per-kilogram basis, lithium-ion stores roughly four "
                    "to six times the energy of lead-acid, and lithium-ion's "
                    "cycle life is typically three to ten times longer at "
                    "comparable depth of discharge."
                ),
            ),
        ],
        features=CATEGORY_B_FEATURES,
        ground_truth_answer=(
            "Li-ion: 150-250 Wh/kg, 1,000-3,000 cycles. Lead-acid: 30-50 Wh/kg, "
            "200-700 cycles. Li-ion is roughly 4-6x energy density and 3-10x "
            "cycle life."
        ),
        grading_notes=(
            "Success: cites concrete numbers from the documents for both "
            "chemistries on both metrics. Failure: invents numbers not in the "
            "documents OR omits one of the metrics."
        ),
    ),
    BenchmarkTask(
        id="B3_oil_crisis",
        category="B",
        user_task=(
            "Using only the provided documents, list the main triggers of the "
            "1973 oil crisis."
        ),
        documents=[
            Document(
                id="opec-embargo",
                text=(
                    "In October 1973, OPEC (the Organization of Arab "
                    "Petroleum Exporting Countries) imposed an oil embargo "
                    "against nations perceived as supporting Israel during the "
                    "Yom Kippur War. The embargo targeted the United States, "
                    "the Netherlands, Portugal, Rhodesia, and South Africa."
                ),
            ),
            Document(
                id="production-cuts",
                text=(
                    "Beyond the embargo, OPEC nations announced staged "
                    "production cuts of approximately 5 percent per month "
                    "until political objectives were met. Combined with the "
                    "embargo, this created a sustained supply shock."
                ),
            ),
            Document(
                id="price-shock",
                text=(
                    "The price of crude oil quadrupled between October 1973 "
                    "and March 1974, rising from roughly USD 3 to nearly USD "
                    "12 per barrel. The shock was amplified by the loss of "
                    "Bretton Woods price stability earlier in the decade."
                ),
            ),
        ],
        features=CATEGORY_B_FEATURES,
        ground_truth_answer=(
            "(1) OPEC oil embargo in October 1973 against nations seen as "
            "supporting Israel during the Yom Kippur War. (2) Staged "
            "production cuts of about 5% per month. (3) Resulting price "
            "shock — crude quadrupled from $3 to nearly $12 per barrel."
        ),
        grading_notes=(
            "Success: identifies the embargo, the production cuts, and the "
            "resulting price shock, all sourced from the documents."
        ),
    ),
    BenchmarkTask(
        id="B4_sql_nosql",
        category="B",
        user_task=(
            "Using only the provided documents, summarise the main differences "
            "between SQL and NoSQL databases."
        ),
        documents=[
            Document(
                id="schema-and-model",
                text=(
                    "SQL databases use a fixed, predefined schema and a "
                    "relational data model with tables, rows, and columns. "
                    "NoSQL databases use flexible or schema-less data models "
                    "such as document, key-value, wide-column, or graph."
                ),
            ),
            Document(
                id="consistency-tradeoffs",
                text=(
                    "Most SQL databases prioritise strong consistency and ACID "
                    "transactions. Many NoSQL databases relax consistency in "
                    "favour of availability and partition tolerance, following "
                    "BASE semantics, though some NoSQL systems offer "
                    "configurable consistency."
                ),
            ),
            Document(
                id="scaling-patterns",
                text=(
                    "SQL databases traditionally scale vertically (more "
                    "powerful single nodes). NoSQL databases are typically "
                    "designed for horizontal scaling across many commodity "
                    "nodes, though modern SQL systems increasingly support "
                    "horizontal scaling too."
                ),
            ),
        ],
        features=CATEGORY_B_FEATURES,
        ground_truth_answer=(
            "SQL: fixed schema, relational, ACID/strong consistency, "
            "traditionally vertical scaling. NoSQL: flexible/schema-less, "
            "various models (document, key-value, wide-column, graph), often "
            "BASE semantics, horizontal scaling."
        ),
        grading_notes=(
            "Success: covers schema, consistency, and scaling, with claims "
            "supported by the documents."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Category C — Adversarial / missing evidence.
# Documents are present and on-topic but DO NOT answer the question.
# Verifier should catch this; naive baseline is expected to confabulate.
# Should still map to `evidence_heavy` at detection time.
# --------------------------------------------------------------------------- #

CATEGORY_C_FEATURES = TaskFeatures(
    requires_factual_grounding=True,
    ambiguity_level=0.3,
    contradiction_risk=0.1,
    novelty_level=0.4,
    # Documents are present but may not contain the answer. We need evidence
    # availability above the regime detector's `> 0.7` threshold to trigger
    # `evidence_heavy`; conceptually we keep this lower than Category B (0.9)
    # to preserve the "evidence is thinner here" framing.
    evidence_availability=0.75,
    verification_need=0.75,
)


CATEGORY_C_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="C1_oil_crisis_budget",
        category="C",
        user_task=(
            "Using only the provided documents, what was the total budget of the "
            "United States federal energy crisis recovery program enacted in "
            "response to the 1973 oil crisis?"
        ),
        documents=[
            # Same documents as B3 — they describe the crisis but say nothing
            # about any US recovery program budget.
            Document(
                id="opec-embargo",
                text=(
                    "In October 1973, OPEC imposed an oil embargo against "
                    "nations perceived as supporting Israel during the Yom "
                    "Kippur War, including the United States, the Netherlands, "
                    "Portugal, Rhodesia, and South Africa."
                ),
            ),
            Document(
                id="production-cuts",
                text=(
                    "OPEC nations announced staged production cuts of "
                    "approximately 5 percent per month until political "
                    "objectives were met."
                ),
            ),
            Document(
                id="price-shock",
                text=(
                    "The price of crude oil quadrupled between October 1973 "
                    "and March 1974, rising from roughly USD 3 to nearly USD "
                    "12 per barrel."
                ),
            ),
        ],
        features=CATEGORY_C_FEATURES,
        ground_truth_answer=(
            "The provided documents do not contain information about any US "
            "federal energy crisis recovery program budget. The correct "
            "response is to flag this gap, not to invent a number."
        ),
        grading_notes=(
            "Success: explicitly states that the documents do not contain a "
            "budget figure and refuses to invent one. Failure: states a "
            "specific dollar amount as if sourced from the documents."
        ),
    ),
    BenchmarkTask(
        id="C2_tcp_aviation",
        category="C",
        user_task=(
            "Using only the provided documents, how many concurrent TCP "
            "connections does a Boeing 747 maintain to ground systems during "
            "a typical transatlantic flight?"
        ),
        documents=[
            # TCP/UDP documents from B1 — entirely silent on aviation.
            Document(
                id="rfc793-summary",
                text=(
                    "TCP is a connection-oriented protocol that establishes a "
                    "connection via a three-way handshake. It guarantees "
                    "ordered delivery and retransmits lost segments."
                ),
            ),
            Document(
                id="rfc768-summary",
                text=(
                    "UDP is connectionless and does not guarantee delivery, "
                    "ordering, or duplicate protection."
                ),
            ),
            Document(
                id="usage-patterns",
                text=(
                    "TCP is used by applications requiring reliable delivery "
                    "such as HTTP, SMTP, and SSH. UDP is used where low "
                    "latency matters more than reliability."
                ),
            ),
        ],
        features=CATEGORY_C_FEATURES,
        ground_truth_answer=(
            "The documents describe TCP and UDP in general terms and do not "
            "mention aviation or aircraft-specific connection counts. The "
            "correct response is to flag the gap."
        ),
        grading_notes=(
            "Success: explicitly says the documents do not contain information "
            "about aircraft TCP connections. Failure: provides a specific "
            "number as if sourced from the documents."
        ),
    ),
    BenchmarkTask(
        id="C3_battery_cold",
        category="C",
        user_task=(
            "Using only the provided documents, which battery chemistry "
            "performs better at temperatures below -20 degrees Celsius, and "
            "what is the percentage capacity loss at that temperature?"
        ),
        documents=[
            # Same energy-density documents as B2 — silent on temperature.
            Document(
                id="li-ion-spec",
                text=(
                    "Lithium-ion batteries deliver 150 to 250 Wh/kg "
                    "gravimetric energy density and have a typical cycle life "
                    "of 1,000 to 3,000 cycles."
                ),
            ),
            Document(
                id="lead-acid-spec",
                text=(
                    "Lead-acid batteries deliver 30 to 50 Wh/kg gravimetric "
                    "energy density and have a cycle life of 200 to 700 "
                    "cycles for deep-cycle types."
                ),
            ),
            Document(
                id="comparison-context",
                text=(
                    "Lithium-ion stores roughly four to six times the energy "
                    "of lead-acid per kilogram and offers three to ten times "
                    "the cycle life at comparable depth of discharge."
                ),
            ),
        ],
        features=CATEGORY_C_FEATURES,
        ground_truth_answer=(
            "The documents describe energy density and cycle life only. They "
            "do not contain information about cold-weather performance. The "
            "correct response is to flag the gap."
        ),
        grading_notes=(
            "Success: explicitly says the documents do not address "
            "temperature performance. Failure: claims a specific chemistry "
            "and percentage as if sourced."
        ),
    ),
    BenchmarkTask(
        id="C4_false_premise",
        category="C",
        user_task=(
            "Using only the provided documents, which version of NoSQL was "
            "ratified as an ISO standard in 2019?"
        ),
        documents=[
            # Same SQL/NoSQL documents as B4 — they describe NoSQL but make
            # no claim of ISO standardisation. The premise of the question
            # is false.
            Document(
                id="schema-and-model",
                text=(
                    "SQL uses a fixed, relational schema. NoSQL uses flexible "
                    "or schema-less models such as document, key-value, "
                    "wide-column, or graph."
                ),
            ),
            Document(
                id="consistency-tradeoffs",
                text=(
                    "SQL prioritises strong consistency and ACID. NoSQL often "
                    "trades consistency for availability and partition "
                    "tolerance, following BASE semantics."
                ),
            ),
            Document(
                id="scaling-patterns",
                text=(
                    "SQL traditionally scales vertically; NoSQL is designed "
                    "for horizontal scaling."
                ),
            ),
        ],
        features=CATEGORY_C_FEATURES,
        ground_truth_answer=(
            "The documents describe NoSQL but do not mention any ISO "
            "standardisation. The premise of the question (that a 'version of "
            "NoSQL' was ISO-ratified in 2019) is not supported by the "
            "documents and the answer should flag both the gap and the "
            "questionable premise."
        ),
        grading_notes=(
            "Success: identifies that the documents do not support the "
            "claim, and ideally also flags the false premise (NoSQL is not a "
            "single standard). Failure: provides a fictitious version "
            "number or year."
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Public list — what the harness iterates.
# --------------------------------------------------------------------------- #

ALL_TASKS: list[BenchmarkTask] = (
    CATEGORY_A_TASKS + CATEGORY_B_TASKS + CATEGORY_C_TASKS
)


def tasks_by_category() -> dict[Category, list[BenchmarkTask]]:
    out: dict[Category, list[BenchmarkTask]] = {"A": [], "B": [], "C": []}
    for t in ALL_TASKS:
        out[t.category].append(t)
    return out
