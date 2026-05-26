# `agensflow.runtime.client`

OpenRouter client wrapped with Instructor for typed I/O. The transport
layer for every agent invocation in AgensFlow.

## Purpose

Three jobs in one module:

1. **Single transport for every provider.** OpenRouter exposes OpenAI,
   Anthropic, Google, Qwen, and dozens more behind one OpenAI-shaped
   API. AgensFlow agents pass `model="provider/model-id"`; the client
   handles the rest.
2. **Instructor-enforced typed responses.** Every agent passes a
   Pydantic `response_model`; Instructor enforces the schema via tools
   mode. Failed validations get **bounded** corrective retries (one,
   not a stack — the framework's discipline) and surface as
   `InvalidAgentOutputError`.
3. **Honest cost accounting for failed attempts.** A registered
   parse-error hook records every failed validation as a `TraceEvent`
   with `error="..."` populated. The framework's "fewer tokens per
   successful task than retry stacks" claim depends on this — if
   recovery events were invisible, the cost story would be unhonest.

## Architecture

```
OpenRouterClient(api_key, config=ClientConfig())
  ├─ OpenAI SDK pointed at config.base_url
  ├─ Instructor wrapper in config.instructor_mode
  └─ hooks: completion:kwargs, completion:response, parse:error
       └─ start timer / capture latency / record failed-attempt TraceEvent

client.complete_typed(model, system_prompt, user_prompt, output_model, ...)
  └─ ContextVar carries (agent_name, trace, state_snapshot) into hooks
  └─ instructor.chat.completions.create_with_completion(...)
       ├─ on success: return CompletionResult(parsed, tokens, latency, ...)
       ├─ on failed attempt: parse:error hook → record TraceEvent(error=...)
       └─ on retries exhausted: raise InvalidAgentOutputError
```

## Configuration knobs

| knob | default | what it controls | tune when |
|---|---:|---|---|
| `base_url` | OpenRouter `/api/v1` | transport endpoint | self-hosted gateway / fixture server |
| `timeout_seconds` | 60.0 | per-request HTTP timeout | fast-model-only experiments — lower to 20-30; frontier models — keep at 60+ |
| `max_transport_retries` | 2 | OpenAI SDK transport retry budget | rarely tune — stacking with Instructor risks the multi-layer retry stack we explicitly criticize |
| `app_name` | "AgensFlow" | OpenRouter X-Title header | per-customer attribution |
| `site_url` | "https://agensflow.ai" | OpenRouter HTTP-Referer header | per-customer attribution |
| `instructor_mode` | "tools" | Instructor schema-enforcement mode | benchmark "json" vs "tools" for your provider mix |
| `default_max_retries` | 2 | per-call validation retry budget when caller doesn't override | rarely tune — bounded recovery is the framework's discipline |
| `default_temperature` | 0.2 | per-call sampling temperature default | creative tasks — raise to 0.5-0.8 (accepting some schema-failure rate) |
| `default_max_tokens` | 4096 | per-call max output tokens default | long-form solver variants — raise to 8192-16384 |

Defaults ship in `agensflow/configs/defaults/client.yaml`.

## Usage

### Default (env var `OPENROUTER_API_KEY` is set):

```python
from agensflow.runtime.client import OpenRouterClient
client = OpenRouterClient()
result = client.complete_typed(
    model="anthropic/claude-haiku-4.5",
    system_prompt="...", user_prompt="...",
    output_model=MyPydanticOutput,
    agent_name="my_agent", trace=trace,
    state_snapshot={"...": "..."},
)
print(result.parsed_output)
```

### From YAML:

```yaml
# my-config.yaml
client:
  timeout_seconds: 30.0
  default_max_tokens: 8192
```

```python
from agensflow.config import load_config
from agensflow.runtime.client import OpenRouterClient
cfg = load_config("my-config.yaml")
client = OpenRouterClient(
    base_url=cfg.client.base_url,
    timeout_seconds=cfg.client.timeout_seconds,
    max_transport_retries=cfg.client.max_transport_retries,
    app_name=cfg.client.app_name,
    site_url=cfg.client.site_url,
)
# Per-call defaults flow through complete_typed kwargs as needed.
```

## Required environment variables

- `OPENROUTER_API_KEY` — required at construction unless `api_key=` is
  passed explicitly. Without it, `OpenRouterClient()` raises
  `RuntimeError` immediately (fail-fast, before any LLM tokens are
  spent). The pre-flight module's `check_openrouter` catches this
  earlier still.

## Design notes

- **`ClientConfig` is mutable by mechanism, immutable by convention.**
  Same OmegaConf trade-off as every other module — see
  `web_search/README.md` for the rationale.

- **Per-call ContextVar.** `_current_call` is a `ContextVar` carrying
  the per-call attribution into the Instructor hooks. This is what
  lets the parse-error hook know which agent / trace to attribute the
  failed attempt to without polluting the synchronous `complete_typed`
  signature with thread/async-state plumbing.

- **No second retry layer here.** OpenAI SDK does transport retries;
  Instructor does validation retries. We deliberately do NOT add a
  third layer in this module — that's the framework's "no retry
  stacks" thesis applied to its own implementation.

- **`instructor_mode` is a string in the config, not an enum.** YAML
  doesn't naturally serialize Python enums; the constructor maps the
  string to `instructor.Mode` at runtime. Add new modes here when
  Instructor adds them upstream.

## Caveats

- **Synchronous client.** All calls block until the response (or
  timeout). LangGraph's runtime is happy with this; if you move to
  fully-async in the future, this needs a sibling async client.

- **No streaming.** AgensFlow needs the full structured response to
  validate against the Pydantic schema, so streaming would buy
  nothing. If you want token-by-token UI updates, add streaming
  support OUTSIDE this client (e.g. for a chat surface) and keep the
  typed transport for agent invocations.

## Tests

`tests/test_client.py` covers construction (env-var resolution,
key-missing raise), `complete_typed` happy path against mocked
Instructor, parse-error trace-event recording, retry-exhaustion →
`InvalidAgentOutputError` mapping.
