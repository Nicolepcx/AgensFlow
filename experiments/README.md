# Experiments

Empirical validation runs that test specific claims about the AgensFlow
framework. These live outside `src/agensflow/` because they are not part of
the library — they are the evidence *for* the library.

Each subdirectory is a self-contained experiment:
- A `README.md` stating the hypothesis and design.
- The benchmark, baselines, harness, and grader code.
- A `RESULTS.md` that pre-registers predictions BEFORE any run, then records
  the actual outcomes after the run.

## Running an experiment

From the repo root, with the package installed in editable mode and
`OPENROUTER_API_KEY` set in `.env`:

```
python -m experiments.e01_regime_validation.run
```

Each experiment writes its results back into its own `RESULTS.md`.
