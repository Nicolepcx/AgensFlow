"""
Chunk 9 — declarative skill-definition variants (skill cards × model bindings).

Builds on the chunk-8 substrate (sustained learning + topology learning via
inline skip-X) by enriching the solver action space from "model variants of
one hardcoded prompt" to "(skill card × model) cross product."

The systems-perspective hypothesis being tested:

  - Different (skill spec, model binding) combinations have non-trivial
    reliability profiles that aren't observable from either axis alone.
  - The framework discovers per-(signature, domain) combinations that are
    cost-efficient and reliable, where the substrate's tracked metrics
    (retry rate, failure rate, reward variance, token variance) make the
    interaction surface visible.
  - Skill specs act as runtime constraints on model behavior — not as
    "better prompts," but as behavioral envelopes that change the model's
    output distribution toward more parseable / more reliable patterns.

Outputs:
  - results_agensflow.jsonl    one line per task-run, with `epoch` field
  - policy_graph_epoch_NN.pkl  per-epoch graph snapshots
  - policy_graph.pkl           final graph
  - RESULTS.md                 auto-generated aggregations + manual framing
"""
