"""
Chunk 8.5 — cross-condition quality evaluation.

Tests whether chunk-8 (skip-on) trajectories are *genuinely* competitive
with chunk-7 (no-skip) trajectories, or whether the +0.02 RULER lift in
the headline is an artifact of RULER ranking each condition against its
own peer group rather than against the other condition.

For each task in the 59-task pool, we pull the last-2-epoch trajectories
from both chunk 7 and chunk 8, build a 4-trajectory comparison group,
and run RULER scoring with three judges in parallel:

  - anthropic/claude-haiku-4.5 (the original judge — same-family check)
  - openai/gpt-5.4              (cross-family, OpenAI)
  - google/gemini-3.1-pro-preview (cross-family, Google — chunk-11
    note: this judge returns null score entries under TOOLS mode;
    chunk-11's per-judge config infrastructure can route it via
    Mode.JSON, but gemini-pro-preview specifically has additional
    null-field issues that remain unresolved — chunk-12 follow-up)

The output: per-task win/loss data per judge, plus inter-judge agreement.
This is what tells us whether the chunk-8 quality preservation claim
survives independent scrutiny across model families.
"""
