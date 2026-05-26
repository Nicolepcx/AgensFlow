"""
Chunk 7 — sustained-traffic experiment.

Reuses the chunk-6 task pool (minus C7.1, the recursion-limit edge case) and
the chunk-6 harness, but runs N epochs (default 8) over the pool with a
warm-started policy graph (loaded from chunk 6.5). The headline question:
*does the system get more reliable with use?*

Outputs:
  - results_agensflow.jsonl    (one line per task-run, with `epoch` field)
  - policy_graph_epoch_NN.pkl  (per-epoch graph snapshots)
  - policy_graph.pkl           (final graph)
  - RESULTS.md                 (auto-generated aggregations)
"""
