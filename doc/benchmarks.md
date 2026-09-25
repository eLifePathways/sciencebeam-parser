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

A document the parser fails on writes no prediction, and scoring iterates
prediction files, so it is left out rather than scored zero. The report states
each run's coverage — how many documents a retry recovered, and how many ended
with no prediction — and calls out a comparison whose columns cover different
documents, since a delta across unequal sets reflects which documents each
column covered as well as how it performed.

## Where the gold does not record a field

Whether a publisher records an acknowledgement or marks its body sections is a
property of the publisher rather than of the document, and nothing in the PDF
says which it is. A score over every document therefore mixes how well a field
is extracted with how often a model abstains.

The report keeps that combined figure and pairs it with a second row, in the
Overall table and in each per-corpus table, for every field where some column
produced a value the gold has none of. A `Docs` column says which documents each
row covers — `all 222` against `gold 79` — so neither is read as the other. The
`Overall` gold row is weighted by the gold documents it covers, so a corpus
recording nothing for the field drops out of it; the combined row is unchanged.

A field a corpus records nothing for shows `—` rather than `0.000`, since there
is nothing there to extract, and gets no second row.

What each column produced on the documents whose gold records nothing is counted
in its own section at the foot of the report, in documents and in values. It is
not an extraction result and carries no delta.

Both figures come from the per-document score files, so
`python -m benchmarks.score --run <dir> --from-scores` re-summarises a run
without scoring it again, which also works where its gold is no longer cached.
It summarises the score files as they stand, including any left by an earlier
scoring of the same directory.

## Running it

```sh
make docker-benchmark-with-baselines            # smoke by default
make docker-benchmark BENCHMARK_MODE=small
make docker-benchmark BENCHMARK_RETRY_PASSES=2  # ask again for what failed
```

`BENCHMARK_RETRY_PASSES` is how many times a run goes over the corpus, asking
again for documents that still have no prediction; `1` never retries. CI uses
`2`, a local run the default, so a failure stays in front of you.

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
