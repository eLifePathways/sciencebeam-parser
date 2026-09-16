# Benchmarks

Measures extraction quality against published JATS. A run fetches gold
documents, converts each PDF with the tool under test, scores the result field
by field, and prints one table per corpus comparing every tool in the run.

## Corpora, splits and modes

Configured in [`benchmarks/eval.yml`](../benchmarks/eval.yml).

| split | for |
| --- | --- |
| `train` | reading closely when an extraction failed; the default |
| `validation` | CI, on every labelled PR |
| `test` | numbers meant to be published |

Modes size the sample: `smoke` (10 per corpus), `small`, `medium`, `large`,
`full`. They nest, so generating at the largest mode covers every smaller one.

`plos-manuscripts` is opt-in (`--include-corpus`): those manuscripts are not
redistributable, so reaching for them is a decision.

## Scoring

Gold is JATS; predictions are TEI or JATS. sciencebeam-judge picks its field
mapping from the root element, so both score through the same definitions — the
extension is the only difference. Fields and methods are set in `eval.yml`.

## The predictions store

`sciencebeam-eval-predictions`, a private repo written by CI, keyed
`<tool>/<version>/<profile>/`. A run fetches what it has and generates only
what is missing; the version in the path stops a bumped tool scoring against its
predecessor's predictions. Beside the predictions, `manifest.jsonl` records each
document's outcome and, for a served model, its calls and timings.

It must stay private: it holds predictions from the PLOS manuscripts, which are
close to the full text of the documents they came from.

## Baselines and the report

Every entry under `baselines:` becomes a column plus a delta against the primary
run. `generate: false` means it contributes what the store has and never
generates.

**Overall** covers only corpora every column scored, so a column missing one
drops it from the aggregate for all of them.

## Running it

```sh
make docker-benchmark-with-baselines            # smoke by default
make docker-benchmark BENCHMARK_MODE=small
```

In CI, label a PR `benchmark:smoke` (or `:small`, `:medium`, `:large`, `:full`,
`:plos`). Each run posts a new comment and collapses the previous ones.

## The trained JATS annotation model

Not the [LLM engine](llm_engine.md), which swaps an API model into the parser.
This is a whole-document pipeline — PDF → markdown → three section rollouts →
JATS — scored as its own tool.

Annotating needs a GPU the benchmark runner does not have, so it never generates
during a run. Predictions come from the **Generate LLM predictions** workflow
and are read from the store like any baseline that does not generate:

```sh
gh workflow run "Generate LLM predictions" --ref main
```

The checkpoint defaults to the `version:` `eval.yml` gives the tool, so
generation stores under the version the benchmark reads. What the store already
has is fetched rather than regenerated.

**Check the service is serving the checkpoint you are storing under.** The run
logs what `/health` reports and records it as `served_model`, but cannot refuse
a mismatch — a served-model label and a checkpoint name never match textually.

When it finishes, read the job summary's ok/error counts rather than the green
tick: a failed document is recorded, not raised. Note this column can never
cover `plos-manuscripts`, so on `main` it costs that corpus from Overall.
