# Examples

## Setup

1. Put `OPENROUTER_API_KEY=sk-or-...` in a `.env` file at the repo root.
2. From the repo root, install in editable mode:
   ```
   pip install -e .
   ```
3. Run any example as a normal Python script.

## Examples

### `01_evidence_heavy_end_to_end.py`

End-to-end run of the **evidence_heavy** regime: planner → memory → solver →
verifier → evaluator, with structured JSON handoffs at every edge and real LLM
calls via OpenRouter.

The user task is grounded in a small set of provided documents about TCP and
UDP. The memory agent retrieves evidence from the documents; the solver drafts
an answer; the verifier checks it against the evidence; the evaluator decides
whether the run is complete and produces the user-facing answer.

The trace at the end of the run shows per-agent token cost and latency — the
raw input that Layer 2 metrics (HFE, ACE, AR, SP, total cost) will consume in a
later release.

```
python examples/01_evidence_heavy_end_to_end.py
```

## What's coming

Subsequent examples will demonstrate:
- Other regimes (straightforward, ambiguous, contradictory, high_risk).
- The branching-execution builder + merge strategies (planned — today
  branching is on the design surface as `ActivationPlan.branch_rule`
  but the runtime builders reject branching plans with
  `NotImplementedError`; the linear and dynamic-routing builders are
  what's shipped).
- Trace metrics computed over a run.
- A Streamlit visualizer for the handoff state and trace.
