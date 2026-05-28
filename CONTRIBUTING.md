You’re right. I over-dissected it. Here is the full `CONTRIBUTING.md` as one file.

# Contributing

AgensFlow is currently in alpha, and contributions are welcome.

Because the framework is designed as a coordination-policy substrate rather than a collection of agent demos, contributions should preserve the core design principles:

* Coordination decisions should remain observable and auditable.
* Runtime behavior should be configurable through the YAML configuration system.
* New modules should expose typed config objects and documented defaults.
* Experiments should include enough information to be reproducible.
* Changes to routing, reward, or evaluation should make their effect on traces and policy updates explicit.
* Integrations should avoid hiding important coordination decisions behind opaque wrappers.

Useful contribution areas include:

* Documentation and examples
* Reproducibility improvements
* Additional experiment harnesses
* Policy-graph extensions
* New skill protocols
* Governance and audit tooling
* RelativeJudge improvements
* Model-provider compatibility probes
* LangGraph and other multi-agent framework integrations
* Visualization and reporting utilities

## Development setup

```bash
git clone https://github.com/Nicolepcx/AgensFlow.git
cd AgensFlow
python -m pip install -e .
```

## Running tests

```bash
pytest
```

## Pull requests

Before submitting large architectural changes, please open an issue first so the design can be discussed in relation to AgensFlow’s layer structure, traceability requirements, and reproducibility goals.

For pull requests, please include:

* A clear summary of the change
* The affected layer or module
* Tests or reproduction steps
* Any new configuration knobs
* Any impact on traces, policy updates, reward calculation, governance, or reproducibility

Large changes that affect routing, reward, governance, persistence, or evaluation should explain why the change preserves AgensFlow’s core design goal: making coordination decisions observable, learnable, and auditable.

## PR template

````markdown
## Summary

What does this PR change?

## Affected layer/module

* [ ] Layer 0: web search wrappers
* [ ] Layer 1: schema, regimes, skill registry
* [ ] Layer 2: agents, transport, trace
* [ ] Layer 3: policy graph, router, LangGraph integration
* [ ] Layer 4: reward, RelativeJudge
* [ ] Layer 5: pre-flight, governance, reports
* [ ] Config/docs/tests/experiments

## Tests run

```bash
pytest
````

## Reproducibility impact

Does this change affect experiment outputs, policy values, reward calculation, traces, reports, or saved artifacts?

## Governance/security impact

Does this change affect tool use, model calls, external APIs, credentials, policy checks, audit behavior, or failure handling?

## Notes

Anything reviewers should pay particular attention to?

