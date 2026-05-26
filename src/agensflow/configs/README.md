# `agensflow.config` — YAML configuration system

How to configure an AgensFlow run end-to-end without touching code.

## TL;DR

```python
from agensflow.config import load_config

cfg = load_config("my-config.yaml")
print(cfg.web_search.exa_max_retries)
print(cfg.policy_graph.confidence_threshold)
print(cfg.reward.ruler_weight)
```

`my-config.yaml` overrides only the keys you care about; everything else
falls through to the framework defaults. Unknown keys raise
`UnknownKeyError` in strict mode (default) so typos are caught at load
time, not silently ignored.

## Why this design

Three problems we wanted to avoid:

1. **30 command-line flags.** Long-running experiments shouldn't be
   configured via `--exa_max_retries=4 --policy_graph_ucb_c=1.4 ...`.
2. **Config scattered across modules.** Knobs hidden as inline
   constants (`UCB_C = 1.4`) at module top are invisible to anyone who
   isn't reading the source. 
3. **Hand-rolled config plumbing per module.** Every module having
   its own config-loading code means inconsistent override semantics
   and inconsistent error messages.

The OmegaConf-based loader fixes all three:

- One YAML file per run, edited like any other config artifact.
- Every module exposes its knobs as a typed dataclass (`config.py`)
  with a YAML defaults file shipped in the package.
- The loader composes module configs into a single `AgensflowConfig`
  schema; OmegaConf does the merge + validation.

## Architecture

```
agensflow.config.load_config(*user_paths, strict=True, extra=None)
  └─ schema = OmegaConf.structured(AgensflowConfig)
       (every module's Config dataclass registered as a field)

  └─ for each YAML in agensflow/configs/defaults/:
       wrap under <stem> (e.g. web_search.yaml → web_search:)
       merge into schema

  └─ for each user_path:
       merge with strictness check
       (strict=True raises UnknownKeyError on typos;
        strict=False logs warning + drops unknown keys)

  └─ OmegaConf.to_object(schema)  → typed AgensflowConfig instance
```

The composition root is `AgensflowConfig`:

```python
@dataclass
class AgensflowConfig:
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    governance: GovernancePolicy = field(default_factory=GovernancePolicy)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    policy_graph: PolicyGraphConfig = field(default_factory=PolicyGraphConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ruler: RulerConfig = field(default_factory=RulerConfig)
```

Add a new module's config by:
1. Creating `<module>/config.py` with a typed dataclass
2. Shipping `agensflow/configs/defaults/<module>.yaml` (filename stem
   == loader field name, by convention)
3. Registering as a field on `AgensflowConfig`

That's it — the loader picks up the YAML automatically.

## File layout

```
src/agensflow/
├── config/
│   ├── __init__.py             ← public load_config / write_default_config
│   └── loader.py               ← AgensflowConfig + OmegaConf glue
└── configs/
    ├── __init__.py
    ├── README.md               ← this file
    └── defaults/
        ├── client.yaml
        ├── governance.yaml
        ├── graph.yaml
        ├── models.yaml
        ├── persistence.yaml
        ├── policy_graph.yaml
        ├── preflight.yaml
        ├── report.yaml
        ├── reward.yaml
        ├── router.yaml
        ├── ruler.yaml
        └── web_search.yaml
```

## Usage

### Load defaults only:

```python
from agensflow.config import load_config

cfg = load_config()  # all defaults
print(cfg.web_search)
print(cfg.policy_graph.ucb_c)  # 1.4
```

### Override with a user YAML:

```yaml
# my-experiment.yaml
web_search:
  exa_max_results: 5

policy_graph:
  confidence_threshold: 10
  reliability_weight: 1.0

reward:
  ruler_weight: 1.5
```

```python
cfg = load_config("my-experiment.yaml")
print(cfg.web_search.exa_max_results)        # 5
print(cfg.policy_graph.confidence_threshold) # 10
print(cfg.web_search.exa_max_retries)        # 4 (default — not overridden)
```

### Compose multiple YAMLs (later overrides earlier):

```python
cfg = load_config("base.yaml", "overrides.yaml", "experiment.yaml")
```

### Programmatic last-mile override:

```python
cfg = load_config("base.yaml", extra={"router": {"max_steps": 20}})
```

### Permissive mode (forward-compat with older configs):

```python
cfg = load_config("legacy.yaml", strict=False)
# unknown keys log a warning and are dropped, instead of raising
```

### Dump defaults to a starter file:

```python
from agensflow import write_default_config
write_default_config("agensflow.yaml")
# now edit agensflow.yaml as your starting point
```

## Strict vs. permissive

| mode | what happens on unknown key |
|---|---|
| `strict=True` (default) | `UnknownKeyError` naming the offending key + source file |
| `strict=False` | warning logged, key is dropped before merge |

Strict in development (catches typos cheaply); permissive in
production where library upgrades may rename or remove knobs the user
hasn't migrated yet.

## Override semantics

OmegaConf's dict-merge:

- **Adds** keys present in the overlay but not in the base.
- **Replaces** values for keys present in both.
- **Recurses** into nested dicts.

So:

```yaml
# base
models:
  default_assignment:
    solver: "anthropic/claude-haiku-4.5"
    planner: "openai/gpt-5.4-nano"
```

```yaml
# overlay
models:
  default_assignment:
    solver: "openai/gpt-5.4-mini"   # REPLACE
    critic: "anthropic/claude-opus-4"  # ADD
```

Result: `solver` becomes `gpt-5.4-mini`, `critic` is added,
`planner` stays `gpt-5.4-nano` (not in overlay, retained from base).

**There is no "delete this key" operator.** To remove a default entry,
build the dict explicitly in your YAML rather than overlaying.

## The `frozen=True` design choice

Every module-level config dataclass deliberately omits `frozen=True`.
This is documented in every module README's "Design notes" section.

**Why:** OmegaConf's structured-config merge requires mutable nested
dataclasses. `frozen=True` causes `ReadonlyConfigError` during YAML
overlay merging. Since the entire point of bringing each module into
`agensflow.config` is the YAML flow, we accept the trade-off.

**What this means for you:** treat config instances as immutable in
application code. Construct once at startup via `load_config(...)`,
pass into runtime, never mutate. If you need per-run variation, build
different `<Module>Config` instances rather than mutating one.

The framework follows this convention internally; your code should
too. There's no enforcement mechanism — only convention. Module
READMEs flag this in the Design notes section so it's not mistaken for
an oversight.

Runtime artifacts (`RunReport`, `CheckResult`, `GovernanceViolation`,
…) ARE marked `frozen=True` — they're produced by runtime code, not
config-merged from YAML, so the safer default applies.

## When to use `load_config` vs. direct construction

Both are fine. Pick based on context:

| use `load_config(...)` when | use `XConfig(...)` directly when |
|---|---|
| you have a YAML file (production runs, experiments) | you're writing a unit test for a single module |
| you want consistent defaults across all modules | you want to override one knob without touching disk |
| you want strict-mode typo detection | the test wants to construct an explicit instance |

The two approaches compose: `load_config(...)` returns
`AgensflowConfig`; you can then pass `cfg.web_search` to a factory or
construct a fresh `WebSearchConfig(...)` for a specific test path.

## Adding a new module's config

Follow the canonical pattern from
[`runtime/web_search/`](../runtime/web_search/README.md):

1. **`<module>/config.py`** — typed dataclass with field defaults.
   Don't mark `frozen=True` (see above).

2. **`<module>/__init__.py`** — re-export the config dataclass and the
   public surface from `core.py`:
   ```python
   from agensflow.<area>.<module>.config import <Module>Config
   from agensflow.<area>.<module>.core import (...)
   __all__ = [...]
   ```

3. **`agensflow/configs/defaults/<module>.yaml`** — defaults that
   mirror the dataclass field-by-field, with comments per knob
   explaining what they do and when to tune them. Filename stem
   MUST match the loader field name.

4. **`<module>/README.md`** — Purpose / Architecture / Knobs table /
   Usage / Required env / Design notes (incl. frozen-by-convention) /
   Caveats / Tests.

5. **Register in `config/loader.py`**:
   ```python
   from agensflow.<area>.<module>.config import <Module>Config

   @dataclass
   class AgensflowConfig:
       ...
       <module>: <Module>Config = field(default_factory=<Module>Config)
   ```

6. **Run the test suite** — the existing `test_config_loader.py` will
   pick up the new section automatically and verify it loads cleanly.

## Caveats

- **No environment-variable interpolation.** The OmegaConf
  `${oc.env:VAR}` resolver isn't enabled. Secrets stay in env vars
  (read by the modules that need them — e.g. `OPENROUTER_API_KEY`).
  YAML is for non-secret tuning.

- **Lists replace, not merge.** OmegaConf's dict-merge is
  position-aware for dicts but list-replace for lists. So
  `default_checks: ["openrouter"]` in your overlay REPLACES the full
  default `[openrouter, exa, tavily]` rather than appending. Build
  the full list when overriding.

- **Tuples become lists.** OmegaConf serializes tuples as lists. The
  `models.variant_bindings` dict uses `list[str]` of length 2 rather
  than `tuple[str, str]` for this reason.

- **`null` overrides defaults.** Setting a YAML field to `null`
  explicitly nulls the value (not "use default"). To keep a default,
  omit the key entirely.

## See also

- [`docs/architecture.md`](../../../docs/architecture.md) — overall
  framework architecture and layer breakdown.
- Per-module READMEs (linked from `architecture.md`) — knob-by-knob
  guidance for each subsystem.
- `tests/test_config_loader.py` — exhaustive coverage of strict/
  permissive modes, override composition, unknown-key detection.
