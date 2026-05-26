# `agensflow.runtime.models`

Skill → model bindings, plus the **variant pattern** that makes
AgensFlow's online model-routing claim a real thing.

## Purpose

Two layers of indirection:

1. **`default_assignment`** — base skill (planner, memory, solver,
   critic, verifier, evaluator, synthesizer) → default model. Used
   when the activation plan invokes the base skill name directly.

2. **`variant_bindings`** — variant skill name (e.g. `solver_fast`,
   `solver_qwen_flash`, `solver_concise_haiku`) → `(base_skill,
   model_id)`. Variants share the base skill's agent factory but route
   to a specific model. The policy graph learns per-signature which
   variant wins on which class of task.

3. **`variant_cards`** — variant skill name → skill_card name. When set,
   the card's `instructions` replace the hardcoded system prompt. This
   is what makes (skill_card × model) the unit of action-space variation
   in chunk-9 — same model, different behavioral spec; the policy learns
   which spec × which model is the cost/quality/reliability winner.

## Architecture

```
get_model_for_skill(skill, override=None)
  └─ if override and skill in override: return override[skill]
  └─ if skill in SKILL_VARIANT_BINDINGS: return variant's model
  └─ if skill in DEFAULT_MODEL_ASSIGNMENT: return base model
  └─ raise KeyError

get_card_for_skill(skill) → str | None
  └─ SKILL_VARIANT_CARDS.get(skill)

is_variant(skill) → bool
  └─ skill in SKILL_VARIANT_BINDINGS

get_base_skill(skill) → str
  └─ if variant: return base; else: return skill itself
```

The variant pattern is what enables AgensFlow's online model-routing:
drop a new variant in the bindings map, the policy explores it on a
fraction of traffic, the reward signal judges it, and the policy
converges to using it on signatures where it wins.

## Configuration knobs

The bindings ARE the config. All three tables are YAML-overridable via
`ModelsConfig` (see `config.py`); defaults ship in
`agensflow/configs/defaults/models.yaml`.

| knob | type | what it controls |
|---|---|---|
| `default_assignment` | `dict[str, str]` | base skill → model id |
| `variant_bindings` | `dict[str, list[str]]` (length-2) | variant → [base_skill, model_id] |
| `variant_cards` | `dict[str, str]` | variant → skill_card name |

YAML override merges: dict-merge **adds** keys + **replaces** on
collision. To **remove** a default variant, build the dict explicitly
in your YAML — OmegaConf has no "delete key" operator.

## Usage

### Default (no setup):

```python
from agensflow.runtime.models import get_model_for_skill, is_variant

print(get_model_for_skill("solver"))               # base
print(get_model_for_skill("solver_qwen_flash"))    # variant
print(is_variant("solver_qwen_flash"))             # True
```

### Override via YAML — add a custom variant:

```yaml
# my-config.yaml
models:
  variant_bindings:
    solver_my_custom: ["solver", "openai/gpt-5.4-mini"]
```

```python
from agensflow.config import load_config
cfg = load_config("my-config.yaml")
print(cfg.models.variant_bindings["solver_my_custom"])
```

### Override via YAML — swap default model for a base skill:

```yaml
# my-config.yaml
models:
  default_assignment:
    solver: "openai/gpt-5.4-mini"  # cheaper than haiku for my workload
```

### Override programmatically (e.g. for tests):

```python
from agensflow.runtime.models import ModelsConfig
cfg = ModelsConfig()
cfg.variant_bindings["solver_test"] = ["solver", "test/echo"]
```

## Design notes

- **`ModelsConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module config — see
  `web_search/README.md` for the rationale.

- **Variant bindings stored as `list[str]` not `tuple[str, str]`.**
  OmegaConf's structured-config schema doesn't handle fixed-length
  tuples cleanly. The list-of-two-strings form serializes naturally to
  YAML and the runtime accessor adapts to a tuple at read time. The
  in-code `SKILL_VARIANT_BINDINGS` constant uses tuples (preserves the
  old API); the YAML uses lists.

- **Module-level constants kept for backward compat.** Code that
  imports `DEFAULT_MODEL_ASSIGNMENT` / `SKILL_VARIANT_BINDINGS` /
  `SKILL_VARIANT_CARDS` directly from `agensflow.runtime.models` keeps
  working — these constants stay populated with the in-code defaults.
  Code that wants YAML-driven bindings goes through
  `agensflow.config.load_config(...).models` instead.

## Caveats

- **Variant card resolution depends on the registry.** A variant whose
  `variant_cards` entry points at a skill-card that isn't registered
  in `default_registry` falls back to the hardcoded prompt. No error.
  This is by design (variants without cards work fine), but it means
  typos in card names silently degrade behavior — verify your card
  bindings with `default_registry.has_card(...)` if you're adding new
  ones.

- **No model-availability check.** The bindings reference model IDs by
  string; if you bind `solver` to a model OpenRouter doesn't expose
  on your tier, you'll get a runtime 4xx from OpenRouter (caught by
  Layer 0 / governance). Pre-flight could grow a model-availability
  probe; not in scope today.

## Tests

`tests/test_models.py` covers `get_model_for_skill` lookup order,
`is_variant`, `get_base_skill`, `get_card_for_skill` lookup +
fallback-to-None.
