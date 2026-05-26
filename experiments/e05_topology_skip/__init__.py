"""
Chunk 8 — inline `skip:X` topology learning.

Same task pool as chunks 6/7 (59 tasks, C7.1 excluded), but with the
router's `enable_skip` mechanism turned on. The action space at every
routing step expands to include `skip:X` for each legal X, making
coalition membership a learnable coordination decision rather than a
planner decision.

The empirical question: when the framework can learn to *exclude* skills
from the topology — not just choose model bindings or order — does it
discover non-trivial skip patterns per scenario class, and does the
resulting cost-quality curve improve over chunk 7's main run?

Outputs:
  - results_agensflow.jsonl    (one line per task-run, with `epoch`)
  - policy_graph_epoch_NN.pkl  (per-epoch graph snapshots)
  - policy_graph.pkl           (final graph)
  - RESULTS.md                 (auto-generated aggregations)
"""
