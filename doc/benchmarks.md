# Benchmarks

Measures extraction quality against published JATS, so a change can be shown to
help rather than argued to.

A run fetches gold documents, converts each PDF with the tool under test, scores
the result field by field, and prints one table per corpus comparing every tool
in the run.

## Corpora, splits and modes

Corpora and sampling live in [`benchmarks/eval.yml`](../benchmarks/eval.yml).

| split | for |
| --- | --- |
| `train` | reading closely when working out why an extraction failed; the default |
| `validation` | CI, on every labelled PR |
| `test` | numbers meant to be published; deliberate runs only |

Modes size the sample: `smoke` (10 per corpus), `small`, `medium`, `large`,
`full`. **They nest** — `smoke` is a subset of `small` is a subset of `medium` —
so generating at the largest mode you need also covers every smaller run.

`plos-manuscripts` is opt-in (`--include-corpus`). Those manuscripts are not
redistributable, which is why reaching for them is a decision rather than a
default, and why an LLM profile combined with them is refused outright.

## Scoring

Gold is JATS. Predictions are TEI from the GROBID-compatible tools and JATS from
an annotation model; sciencebeam-judge selects its field mapping from the root
element, so both are scored through the same field definitions. The extension is
the only thing that differs — see
[`benchmarks/prediction_files.py`](../benchmarks/prediction_files.py).

Fields, scoring types and methods are configured under `fields:` and `scoring:`
in `eval.yml`.

## The predictions store

Predictions live in `sciencebeam-eval-predictions`, a private repo written only
by CI, keyed `<tool>/<version>/<profile>/`.

Generating is expensive and, for a fixed tool version, produces the same output
every time — so a run fetches what the store has, generates only what is missing,
and pushes the rest back. The version in the path is what keeps a bumped tool
from silently scoring against its predecessor's predictions.

**It must stay private.** It holds predictions from the PLOS manuscripts, and a
prediction is close to the whole text of the document it came from. Deleting
files would not remove them from git history.

## Baselines and the report

Every entry under `baselines:` becomes a column, plus a delta against the primary
run (the last one). `generate: false` means a baseline contributes whatever the
store has and never generates.

The **Overall** table covers only corpora that *every* column scored — an
aggregate over a corpus one run lacks would differ for composition reasons, which
is exactly what an overall row is read as ruling out. Per-corpus sections still
show everything, flagged where the columns are unequal.

## Running it

Locally, against the containerised parser:

```sh
make docker-benchmark-with-baselines            # BENCHMARK_MODE=smoke by default
make docker-benchmark BENCHMARK_MODE=small
```

In CI, label a PR `benchmark:smoke` (or `:small`, `:medium`, `:large`, `:full`,
`:plos`), or dispatch the **Benchmark** workflow. Each run posts its report as a
new PR comment and collapses the previous ones.

## The trained JATS annotation model

Not to be confused with the [LLM engine](llm_engine.md), which swaps an API model
in for two CRF sequence models *inside* the parser. This is a separate
whole-document pipeline: PDF → markdown → three section rollouts → JATS, scored
as its own tool rather than as a parser profile.

Annotating needs a GPU, which the benchmark runner does not have, so it never
generates during a benchmark run. Predictions are generated once by the
**Generate LLM predictions** workflow — the model is served over HTTP, so that
job needs no GPU either — and pushed to the store; the benchmark then reads them
like any other baseline that does not generate.

```sh
gh workflow run "Generate LLM predictions" --ref main
```

The checkpoint defaults to the `version:` that `eval.yml` gives the tool, so
generation stores under the version the benchmark reads. Point `endpoint` at a
different service to measure a different deployment.

Two things to check when it finishes, rather than trusting the green tick: the
job summary's ok/error counts, since a failed document is recorded rather than
raised; and whether any corpus ended with no successes, which would drop that
corpus from the Overall table for every column.
