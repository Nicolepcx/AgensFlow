# `agensflow.learning.persistence`

Pickle-based save/load for the policy graph. The mechanism that makes
learning compound across runs — without it, each Python process starts
fresh and AgensFlow behaves identically to the rule-based prior.

## Purpose

The whole point of the policy graph is value estimates that
**accumulate**. Each run contributes its outcome via
`PolicyGraph.backup`, future runs consult it via
`PolicyGraph.best_action`. That flywheel needs serialization to span
process boundaries — that's this module.

The persistence layer is intentionally minimal:

- **Pickle, not JSON.** The graph is plain-Python data
  (dataclasses + dicts + tuples), and pickle handles the schema
  evolution we actually care about (forward-compat back-fill of new
  GraphNode fields). JSON would force schema-versioning machinery for
  no real gain.
- **Snapshot the nodes, not the PolicyGraph instance.** Future schema
  changes to the wrapper class don't break old snapshots; only
  GraphNode field additions matter.
- **No auto-persist.** The framework never writes anywhere
  unprompted. Callers (harnesses, experiment runners) decide when to
  snapshot.

## Architecture

```
save_policy_graph(graph, path, config=None)
  └─ ensure parent dir exists (if config.auto_create_parent_dir)
  └─ pickle.dump(graph.nodes, file, protocol=config.pickle_protocol)

load_policy_graph(path)
  └─ if not exists: return empty PolicyGraph
  └─ pickle.load(file)
  └─ back-fill any GraphNode fields the pickle predates
  └─ return PolicyGraph(nodes=loaded)
```

## Configuration knobs

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `pickle_protocol` | `pickle.HIGHEST_PROTOCOL` | pickle format version | cross-Python-version interop with very old runtimes — lower to 4 |
| `auto_create_parent_dir` | true | whether `save_policy_graph` creates missing parent dirs | strict belt-and-suspenders typo detection — set false |
| `default_snapshot_path` | `""` | fallback path when callers don't supply one | harness wants a consistent snapshot location — set to `"./graphs/policy_graph.pkl"` |

Defaults ship in `agensflow/configs/defaults/persistence.yaml`.

## Usage

### Default (no config):

```python
from agensflow.learning.persistence import (
    load_policy_graph, save_policy_graph,
)

graph = load_policy_graph("./graphs/policy_graph.pkl")  # empty if missing
# ... runs accumulate ...
save_policy_graph(graph, "./graphs/policy_graph.pkl")
```

### With config (e.g. pinned pickle protocol for cross-version interop):

```python
from agensflow.config import load_config
from agensflow.learning.persistence import save_policy_graph
cfg = load_config("my-config.yaml")
save_policy_graph(graph, path, config=cfg.persistence)
```

## Design notes

- **`PersistenceConfig` is mutable by mechanism, immutable by
  convention.** Same OmegaConf trade-off as every other module
  config — see e.g. `web_search/README.md` for the rationale.

- **Forward-compat back-fill on load.** When loading a pickle written
  before a new `GraphNode` field existed, the missing attribute is
  back-filled with the dataclass default. This is what lets a
  chunk-6.5 graph (pre-`action_failure_count`) be warm-started into
  chunk-9 (post-Welford-variance) without a migration script.

- **Pickle is intentional.** We accept the standard pickle caveats
  (don't load untrusted files, format isn't human-readable) in
  exchange for zero-friction schema evolution. The format is internal
  to AgensFlow runs; users who need cross-tool serialization should
  add a JSON exporter on top of `to_dict`-style methods on PolicyGraph
  / GraphNode rather than fight the pickle.

## Caveats

- **No locking.** Concurrent save calls to the same path will race.
  If you parallelize experiment runs, snapshot to per-run paths and
  merge offline — don't have N processes write the same file.

- **No incremental save.** Every snapshot is the full graph. For
  current AgensFlow scale (hundreds of nodes, thousands of visits)
  this is fine; if the graph gets enormous, an append-only changelog
  format would be the next iteration.

- **Pickle protocol coupling.** A graph saved with protocol 5 can't
  load on Python <3.8. Pin via `pickle_protocol` if you need
  portability.

## Tests

`tests/test_persistence.py` covers round-trip save/load, forward-compat
back-fill, missing-file behavior, parent-dir creation.
