"""
Central config loader — defaults + user-overrides + structured validation.

The loader is OmegaConf-based but kept narrow on purpose: a single
`load_config()` entry point, a single `AgensflowConfig` schema dataclass,
and explicit exceptions for the two failure modes (file missing, schema
mismatch) so users can debug their YAML without wading through OmegaConf
internals.

Flow:

  1. Library default YAMLs (one per module) are loaded from
     `agensflow.configs.defaults` package data.
  2. User-supplied YAMLs (zero or more paths) are merged on top, in
     order — later ones override earlier ones.
  3. The merged config is validated against the structured
     `AgensflowConfig` schema. Unknown keys raise `UnknownKeyError` in
     strict mode (default), or log a warning + are dropped in permissive
     mode.
  4. The validated config is converted to a typed `AgensflowConfig`
     dataclass instance for the runtime to consume.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

# Per-module config schemas — imported here for composition into the
# top-level AgensflowConfig. Each module exposes its Config dataclass
# from `<module>/config.py`. As more modules are converted to the
# OmegaConf flow, their imports go here.
from agensflow.learning.persistence.config import PersistenceConfig
from agensflow.learning.policy_graph.config import PolicyGraphConfig
from agensflow.learning.reward.config import RewardConfig
from agensflow.learning.router.config import RouterConfig
from agensflow.learning.relative_judge.config import RelativeJudgeConfig
from agensflow.runtime.client.config import ClientConfig
from agensflow.runtime.governance.config import GovernancePolicy
from agensflow.runtime.graph.config import GraphConfig
from agensflow.runtime.models.config import ModelsConfig
from agensflow.runtime.preflight.config import PreflightConfig
from agensflow.runtime.report.config import ReportConfig
from agensflow.runtime.web_search.config import WebSearchConfig

logger = logging.getLogger("agensflow.config")


class ConfigError(Exception):
    """Base class for configuration errors."""


class UnknownKeyError(ConfigError):
    """Raised in strict mode when the user YAML contains a key that
    isn't part of the structured `AgensflowConfig` schema. Usually a
    typo or a knob that was renamed/removed in a library version
    upgrade. The error message names the offending key so users can fix
    their YAML directly."""


# --------------------------------------------------------------------------- #
# Top-level schema — composes every per-module config dataclass
# --------------------------------------------------------------------------- #
#
# As each module is converted to the OmegaConf flow, its Config class is
# imported here and added as a field on AgensflowConfig. The field name
# determines the YAML key (`web_search:` block in the YAML maps to
# `cfg.web_search` in code).
#
# This file stays small — the actual configuration lives in each
# module's config.py. AgensflowConfig is just the composition root.


@dataclass
class AgensflowConfig:
    """Top-level config — composes every per-module config.

    Mutable (not frozen) so OmegaConf can construct + merge into it.
    Per-module configs ARE frozen (see each module's config.py) — once
    a Config instance is constructed by the loader, it shouldn't be
    mutated by runtime code.

    Each field name MUST match the corresponding default-YAML stem in
    `configs/defaults/` — that's how the loader connects YAMLs to
    schema fields. e.g. `web_search.yaml` → `cfg.web_search`.
    """

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
    relative_judge: RelativeJudgeConfig = field(default_factory=RelativeJudgeConfig)


# --------------------------------------------------------------------------- #
# Default-config discovery
# --------------------------------------------------------------------------- #


def _list_default_yaml_files() -> list[Path]:
    """Return paths to every YAML file shipped under
    `agensflow.configs.defaults` (in install order).

    Uses importlib.resources so it works regardless of whether the
    package is installed editable, wheel, or zipped.
    """
    paths: list[Path] = []
    try:
        defaults_pkg = importlib_resources.files("agensflow.configs.defaults")
    except (ModuleNotFoundError, FileNotFoundError):
        # No defaults shipped yet (we're at the start of the refactor).
        return paths
    for entry in defaults_pkg.iterdir():
        # `entry` is a Traversable; convert to Path when possible.
        name = getattr(entry, "name", "")
        if not name.endswith(".yaml"):
            continue
        try:
            paths.append(Path(str(entry)))
        except Exception:
            continue
    paths.sort(key=lambda p: p.name)
    return paths


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_config(
    *user_paths: str | Path,
    strict: bool = True,
    extra: dict[str, Any] | DictConfig | None = None,
) -> AgensflowConfig:
    """Load + merge config from defaults + user files + optional extra dict.

    Args:
        *user_paths: zero or more paths to user YAML config files. Merged
            in order on top of the library defaults; later files override
            earlier ones. Pass nothing to use library defaults only.
        strict: when True (default), unknown keys in the user config
            raise `UnknownKeyError` — catches typos. When False, unknown
            keys log a warning and are dropped (forward-compat with
            older configs against newer libraries).
        extra: optional dict (or DictConfig) merged on top of everything
            else. Useful for programmatic overrides (e.g. test setups
            that want to bump a single knob without writing a YAML).

    Returns:
        A typed `AgensflowConfig` instance with all per-module configs
        populated from defaults + overrides.

    Raises:
        FileNotFoundError: a `user_paths` entry doesn't exist.
        UnknownKeyError: strict mode and the merged config contains a
            key not in the schema. Message names the key.
        ConfigError: schema validation otherwise failed (wrong type,
            missing required field, etc.).
    """
    schema = OmegaConf.structured(AgensflowConfig)

    # Load + merge defaults from package resources.
    for default_path in _list_default_yaml_files():
        defaults = OmegaConf.load(default_path)
        # Wrap loose top-level keys under the appropriate module name.
        # Convention: filename without extension == module config field name.
        # e.g. `web_search.yaml` → `web_search:` block in merged config.
        module_key = default_path.stem
        wrapped = OmegaConf.create({module_key: defaults})
        schema = OmegaConf.merge(schema, wrapped)

    # Merge user paths on top.
    for p in user_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        user_cfg = OmegaConf.load(path)
        schema = _merge_with_strictness(schema, user_cfg, strict=strict, source=str(path))

    # Final extra dict / DictConfig (programmatic overrides).
    if extra is not None:
        if not isinstance(extra, DictConfig):
            extra = OmegaConf.create(dict(extra))
        schema = _merge_with_strictness(schema, extra, strict=strict, source="<extra>")

    # Convert back to a typed instance. OmegaConf.to_object validates
    # against the structured schema — wrong types / missing required
    # fields raise here.
    try:
        return OmegaConf.to_object(schema)  # type: ignore[return-value]
    except (ConfigAttributeError, ConfigKeyError) as exc:
        raise ConfigError(f"Schema validation failed: {exc}") from exc


def _merge_with_strictness(
    base: DictConfig,
    overlay: DictConfig | Any,
    *,
    strict: bool,
    source: str,
) -> DictConfig:
    """Merge `overlay` onto `base`. In strict mode, unknown overlay keys
    raise `UnknownKeyError` naming the key + the source file. In permissive
    mode, unknown keys are stripped from the overlay (and a warning is
    logged) so the structured base merge doesn't reject them."""
    if not isinstance(overlay, DictConfig):
        overlay = OmegaConf.create(overlay)
    unknown = _find_unknown_keys(base, overlay)
    if unknown:
        if strict:
            keys_str = ", ".join(sorted(unknown))
            raise UnknownKeyError(
                f"Unknown config key(s) in {source}: {keys_str}. "
                f"Did you typo a knob name? Run with strict=False to "
                f"drop unknown keys with a warning instead."
            )
        # Permissive: strip unknown keys from the overlay before merge,
        # so OmegaConf's structured-merge rejection doesn't fire.
        for dotted in sorted(unknown):
            logger.warning(
                "config: ignoring unknown key %s from %s", dotted, source
            )
            _delete_dotted_key(overlay, dotted)
    return OmegaConf.merge(base, overlay)


def _delete_dotted_key(cfg: DictConfig, dotted: str) -> None:
    """Delete a `parent.child.leaf` path from a DictConfig in place.
    Used by permissive mode to strip unknown overlay keys before merge."""
    parts = dotted.split(".")
    node: Any = cfg
    for p in parts[:-1]:
        if not isinstance(node, DictConfig) or p not in node:
            return
        node = node[p]
    if isinstance(node, DictConfig) and parts[-1] in node:
        del node[parts[-1]]


def _find_unknown_keys(
    base: DictConfig | Any,
    overlay: DictConfig | Any,
    prefix: str = "",
) -> set[str]:
    """Recursively diff overlay vs base; return dotted keys present in
    overlay but absent from base.

    Open-dict heuristic: when `base` is a DictConfig with no schema-
    declared keys (i.e. it's a `dict[str, X]` field with an empty
    default — the WHOLE POINT being user-supplied content), we don't
    treat its contents as "unknown." This prevents the loader from
    rejecting valid per-user maps like `cross_judge_modes:
    {"qwen/qwen3.6-flash": "json"}` where the model id contains `/`
    and `.` that OmegaConf would otherwise mistake for path separators.

    The trade-off: typos under such fields can't be caught by strict
    mode. That's acceptable because the field is open-by-design — there
    is no schema for "valid contents" to validate against.
    """
    if not isinstance(overlay, DictConfig) or not isinstance(base, DictConfig):
        return set()
    # Open-dict: base has no schema-declared keys → accept any overlay
    # content as valid user input. See docstring above.
    if len(base) == 0:
        return set()
    unknown: set[str] = set()
    for key in overlay:
        full = f"{prefix}.{key}" if prefix else str(key)
        if key not in base:
            unknown.add(full)
            continue
        # Recurse if both sides are dicts.
        unknown.update(_find_unknown_keys(base[key], overlay[key], full))
    return unknown


def write_default_config(path: str | Path) -> Path:
    """Write the library's default config to `path` as a YAML file.

    Useful starting point for users who want to copy + edit the defaults
    rather than write a config from scratch:

        python -c "from agensflow import write_default_config; write_default_config('agensflow.yaml')"

    Returns the path written to.
    """
    cfg = load_config(strict=True)  # all defaults
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), out)
    return out
