"""
ClientConfig — typed configuration for the OpenRouter client.

The default `OpenRouterClient(...)` constructor takes positional kwargs
matching every field here, so existing call sites keep working
unchanged. The config exists so users running through `load_config(...)`
can drive the client from YAML alongside everything else.

See `README.md` in this directory for the per-knob explanation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClientConfig:
    """Configuration for `OpenRouterClient`.

    Not frozen because OmegaConf's structured-config merge requires
    mutable fields. Treat as immutable by convention: construct once at
    startup via `agensflow.config.load_config(...)`, never mutate.
    """

    # ----- transport ----- #
    # OpenRouter's OpenAI-compatible base URL. Override for self-hosted
    # gateways or for testing against a recorded-fixture server.
    base_url: str = "https://openrouter.ai/api/v1"

    # Per-request HTTP timeout (seconds). 60s is a safe upper bound —
    # frontier-model calls (Claude Opus, GPT-5 Pro) can take 30-45s on
    # complex prompts. Lower for fast-model-only experiments.
    timeout_seconds: float = 60.0

    # The OpenAI SDK's built-in transport-level retry budget. Stacks
    # ABOVE Instructor's validation retries — keep low (≤2) so we don't
    # accidentally build the multi-layer retry stack the framework
    # explicitly criticizes.
    max_transport_retries: int = 2

    # ----- attribution headers (sent to OpenRouter) ----- #
    # Shown in OpenRouter's dashboard so users can attribute spend to a
    # specific application / site.
    app_name: str = "AgensFlow"
    site_url: str = "https://agensflow.ai"

    # ----- Instructor mode ----- #
    # Instructor schema-enforcement mode. "tools" is the most provider-
    # portable + gives strongest schema enforcement; "json" is faster
    # but provider-dependent. Use the string here; the runtime maps to
    # the `instructor.Mode` enum on construction.
    instructor_mode: str = "tools"

    # ----- per-call defaults (used when caller doesn't override) ----- #
    # Default `max_retries` for `complete_typed`. 2 = first attempt + at
    # most one corrective retry — exactly the bounded-recovery
    # discipline the framework requires. Increase only if you have a
    # reason to tolerate more validation failures.
    default_max_retries: int = 2

    # Default sampling temperature for typed completions. 0.2 keeps
    # outputs near-deterministic for schema compliance; raise to
    # 0.5-0.8 for creative tasks if you accept the schema-failure rate.
    default_temperature: float = 0.2

    # Default max output tokens per call. 4096 is generous for the
    # short structured outputs AgensFlow agents produce; raise for
    # solver variants that emit long-form answers.
    default_max_tokens: int = 4096
