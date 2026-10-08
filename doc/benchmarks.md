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

### Author–affiliation linking

`affiliation_text` compares a document's affiliations as one flat list: it says
how well they were extracted, and scores the same whether or not each is attached
to the right author. `affiliation_linked` scores the attachment. Each gold author
is paired with a predicted author by normalised surname, the forename initial
breaking a tie, and the affiliations attached to the two are compared one to one.
Precision, recall and F1 are over author–affiliation links, so an affiliation
shared by five authors is five links.

It follows GROBID's metric of the same name, and like it scores only an author
whose gold link is explicit: an `xref` to the affiliation inside the `contrib`, or
an `aff` nested in it. Publisher JATS often records the link by position alone,
which says nothing a parser could be held to, so such an author is left out
rather than counted as missed, and so is the predicted author paired with them.
A document without an explicit link therefore adds nothing to the figure, in
either direction, unless the prediction names an author the gold does not have.
How many documents have one is `n_gold` under `gold_presence` in `summary.json`.

Affiliations a parser extracted but attached to nobody are links it missed, not
links it got wrong. A named author the gold does not have, carrying affiliations,
is links it got wrong.

The judge's mapping has no way to select a link, so the field is scored in
[`benchmarks/affiliation_linking.py`](../benchmarks/affiliation_linking.py), by
the same methods as the other fields.

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

## Compute cost

Three numbers, beside the scores, for the variants that record them:

- **latency** — what one document waited, as a median and a p90 over the
  documents that got a prediction. A request that timed out took the client
  timeout rather than that long to answer, so it is left out.
- **throughput** — documents an hour, at the concurrency they were generated
  at. `--concurrency 0` resolves to the core count of the machine running the
  client, so the resolved value is recorded with the number.
- **CPU-seconds per document** — the machine's busy time while generating
  them, divided by the documents produced.

Both rates are over what a run produced rather than what it attempted, and
where the two differ the attempts are stated beside them: the wall clock covers
the failures too. A run that produced nothing — one that reached no parser, say,
and failed every document in a tenth of a second — states no rate at all rather
than the fastest ever recorded.

The CPU figure is the whole machine (`/proc/stat`), not the parser alone: the
work is spread across a persistent wapiti process, in-process torch threads and
subprocesses, and no measure taken inside the parser sees all of it. So it
includes the benchmark client, and anything else the host was doing, and it is
recorded only where the parser ran on the same machine — a run against a remote
`--parser-url` records none. Where only part of a set was measured that way, the
rate is stated over the part that was.

All three cover every run that generated any of the predictions, not only the
run that scored them. A run fetches what the predictions store has and asks the
parser only for what is missing, so a set is usually assembled over several
invocations, and may be assembled on several machines. Each invocation appends
its own record to `manifest.jsonl` — what it processed, how long it took, at
what concurrency, on what machine — beside the documents it produced, and each
document entry is stamped with the invocation that wrote it. The store carries
the manifest, so those records arrive with the predictions. Every other reader
of the manifest selects on `status` or on a document's keys, so a line with
neither is ignored by all of them.

The report names what the set was measured on, and warns when the runs behind
one column, or the columns being compared, differ in hardware or concurrency,
since a timing delta is then partly a property of the measurement.
## A comparison of your own

A comparison is a file under `benchmarks/comparisons/`, naming the variants to
put side by side, the rows to keep and what each chart shows. It is a view over
summaries that already exist, so adding one costs no run and moves no figure.

```yaml
# benchmarks/comparisons/reference-models.yml
variants:
  - {label: grobid, tool: grobid, version: 0.9.1-crf, profile: default}
  - {label: main, tool: sciencebeam-parser, version: main, profile: grobid_crf}
  - {label: this run, current: true}      # the last is the primary; deltas are to it

corpora: [biorxiv, pkp, scielo_br]        # optional; every corpus otherwise

rows:
  - {field: reference_title, method: levenshtein, type: partial_list}
  - {field: acknowledgement, methods: [levenshtein, edit_sim], scope: gold}

charts:
  - row: {field: reference_title, method: levenshtein}
    title: Reference titles by corpus
    corpora: [biorxiv, pkp]               # optional; the table's corpora otherwise
  - rows:                                 # several rows, fields along the axis
      - {field: title, method: levenshtein}
      - {field: reference_title, method: levenshtein}
    title: Key fields
  - {compute: cpu_seconds_per_doc}        # what it spent, rather than what it scored
  - {compute: estimated_cost_per_1k, cpu_usd_per_hour: 0.03}
```

```sh
# Fetch each named variant's predictions from the store, score them, and render:
make dev-comparison-with-baselines COMPARISON=reference-models

# Or, where every variant has already been scored, just render:
make dev-comparison COMPARISON=reference-models
```

The first is the one to reach for. Predictions are the expensive part of a benchmark
and the store keeps them, so a question asked after the fact costs only the scoring --
no parser, no docker and nothing generated. The second skips even that, and is for
iterating on rows and charts once the summaries exist.

`BENCHMARK_PREDICTIONS_REPO` points at a checked-out `sciencebeam-eval-predictions`, so
a comparison reads the variants CI sees rather than only those predicted on this machine.
That repo holds `validation` for the GROBID and `sciencebeam-parser` baselines, which is
the split CI runs; `train` is there only where a run pushed it. Comparing what CI compares
therefore means `BENCHMARK_SPLIT=validation`, which is a deliberate choice rather than a
default — the point of leaving `validation` alone is what makes its numbers worth quoting.

`stored-baselines` is the one that needs nothing of its own: it names only variants the
store holds, so it compares what CI compares without a parser or a benchmark run.
`reference-models` adds the run under test, so it needs one — `COMPARISON_CURRENT_RUN`
says where its summary is. `make dev-comparisons-list` names them. `BENCHMARK_DATA` and
`BENCHMARK_RUNS` say where the gold and the runs are, which a git worktree needs since
neither is in one: both are gitignored and stay in the checkout that produced them. The
comparison itself is written to `COMPARISON_OUT_DIR`, which stays in the tree being worked
in rather than following `BENCHMARK_RUNS` — the runs are an input and may be read from
elsewhere.
`COMPARISON_CURRENT_RUN` points a `current: true` variant at a run directory. A comparison compares runs
that have already been scored, so it fails until they have been — naming each variant
it could not find and listing the baselines that *are* there. A git worktree has no
`benchmarks/runs` of its own, since it is gitignored and stays in the checkout that
produced it, so comparing from one means `BENCHMARK_RUNS=<that checkout>/benchmarks/runs`. The target resolves the
comparison's variants against `benchmarks/runs` for `BENCHMARK_SPLIT`, writes
`comparison-<name>.md` into `BENCHMARK_RUN` and prints where the charts went; the
underlying command is `python -m benchmarks.report --comparison <name>`, which takes
`--runs`, `--split` and `--current-run` directly.

Charts name a corpus, a field and a scoring method the way a reader would — `SciELO
Preprints` rather than `scielo_preprints-jats`, `Authors` rather than
`author_full_names`, `edit similarity` rather than `levenshtein`. The identifiers stay as they are in the tables and everywhere a
name has to match `eval.yml` or the predictions store; a corpus with no friendlier name
charts under its own.

`label` is optional. Left out, a column is named for its tool, version and profile —
`sciencebeam-parser main (llm_all)` — which is what tells two profiles of one version
apart. Set it where that runs long, since it is the column heading and the chart's legend
entry; keep the profile in it.

The order of `variants:` is the order the columns and bars appear in. `primary: true`
says which column the deltas measure against — without it the last one, as `--summary`
has always worked — so moving a variant for the sake of reading moves nothing else.

A variant is **named rather than pointed at** — by `tool`, `version` and `profile`,
the way the predictions store holds it — so a checked-in file carries no run id and
resolves against whichever run is at hand. `current: true` is the run under test, and
`summary: <path>` takes a file directly, for something ad hoc. A variant that cannot
be resolved is an error saying which one; nothing is generated to satisfy a comparison.

A **row** is a field, a scoring method and a scope. Omit `method` for every method the
field carries, and `scope` for whichever rows it earns — `gold` exists only where some
variant produced a value the gold has none of. `type` is **asserted, not selected**: a
summary gives a field exactly one scoring type, so naming it catches a run that re-typed
the field instead of comparing across the change.

Charts are drawn in the order the file declares them, whatever their kind, and written to `charts/` beside the report. They are drawn in Source Sans Pro, which ships as a dependency rather than being looked for on the machine, so a chart drawn in CI matches one drawn on a laptop; without the package they fall back to matplotlib's own font.

A **score chart** names one row with `row:` and draws it as a grouped bar chart, the variants as series and the corpora along the axis. Its caption says how the field was scored — `exact match`, `edit similarity`, or `edit similarity, ignoring punctuation` for `edit_sim`, which strips punctuation and whitespace from both sides before measuring — so a `title:` of your own cannot hide it. It reads the same cells the table does, so a variant that scored nothing for a corpus leaves a gap there rather than a bar at zero, and nothing is drawn below two corpora. `rows:` instead of `row:` draws several rows side by side with the fields along the axis, which says which fields a difference reaches rather than where it lives; it needs two rows or more, and having no corpus axis it needs no two corpora either.

A **compute chart** says what a run spent rather than what it scored. `compute:` takes `cpu_seconds_per_doc`, `latency_median`, `latency_p90`, `docs_per_hour` or `estimated_cost_per_1k`, reading the same record the Compute cost section states as text, and draws one bar per variant. Nothing is drawn until two variants recorded the figure — one bar is a number with a rectangle around it, and a run that predates the measurement records none — and the axis says how many did where some did not.

The `latency_` metrics are named for the record they read. What they measure is a whole document being converted, so the charts call it time per document and name the unit as wall-clock seconds.

`estimated_cost_per_1k` prices the CPU at `cpu_usd_per_hour` and adds what an LLM provider charged, each over the documents it was measured over, and stacks the two so the bar says which part it is. Colour there says which part rather than which variant, since that is what the segments differ by; every other chart keeps a variant's colour the same throughout. The default rate is roughly what a small general-purpose instance costs on demand per vCPU-hour — an AWS `t4g`/`c7g` or a GCP `e2`, excluding free tiers and anything with a usage limit. It is a sense of scale rather than a quote: rates move, differ by region and fall with commitment, so check current pricing before quoting any of it, and nothing here is billed at any rate anyway, since CI's CPU costs us nothing. The axis says which rate it used.

The report opens with a collapsed block naming what each column is — the tool, version
and profile behind the label, and which column the deltas measure against — because a
label says what distinguishes a column rather than what produced it, and the comparison
file that knows is somewhere the reader is not.

Anything named that no summary can answer for — a field, method, corpus, scope, variant
or asserted type — is an error that says so, rather than an empty column.

### Ad-hoc narrowing

For a one-off, the same selection is available as flags over `--summary` pairs:

```sh
python -m benchmarks.report \
  --summary "grobid=benchmarks/runs/<run>/summary.json" \
  --summary "head=benchmarks/runs/<run>/validation/summary.json" \
  --field acknowledgement --method levenshtein \
  --corpus biorxiv --corpus pkp \
  --chart acknowledgement --out comparison.md
```

`--field`, `--method` and `--corpus` are repeatable and render in the order given. They
select what is displayed and never what is computed, so a narrowed view shows the same
numbers as the full one, and each is checked against every summary rather than only the
primary. `--chart <field>` draws every method and scope of that field; `--chart-method`
narrows which images without touching the tables.

### Where the images are linked from

The report links charts by relative path, which renders in an editor preview and in the
repository's own view of the file. `--chart-base-url` links them somewhere public
instead, for a surface that cannot render a local file — the files still have to be
published there — and `--chart-prefix` keeps runs published together from overwriting
each other.

### In CI

A `comparison:<name>` label on a PR renders that comparison in CI. On its own it runs
`benchmark-comparison.yml`, which fetches the gold, scores what the predictions store
holds and posts the result — minutes, with no parser and nothing generated, so it is
cheap to re-run whenever the comparison file changes. Alongside a `benchmark:` label it
runs instead as part of `benchmark.yml`, beside the report that always posts; that is
also the only route for a comparison naming `current: true`, since only that run produces
the column. Either way the images are uploaded to the `benchmark-charts` pre-release and
linked from the comment.

The report CI always posts is unchanged and never carries charts. A chart cannot be asked
for by flag in CI — a comparison file is how a run asks for one, and `--chart` on
`benchmarks.report` covers the ad-hoc case locally.

A comparison names variants of its own, which is most of why it exists — they do not
have to be among `eval.yml`'s `baselines:`. Any it names that `eval.yml` does not already
run are fetched from the predictions store and scored before it is rendered. That is not
generating: a variant whose predictions were never pushed to the store cannot be
compared, and fails saying so. In practice that means a profile becomes comparable once
some run has pushed it, which `--push-current` does on `main`.

Those extra variants stay out of the report CI always posts: its columns are still the
`baselines:` entries plus the one profile the run under test uses, which `profile:<name>`
chooses.

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
separately, in documents and in values: a collapsed block under each corpus's
own table, and one under **Overall** covering the run. It is not an extraction
result and carries no delta.

Both figures come from the per-document score files, so
`python -m benchmarks.score --run <dir> --from-scores` re-summarises a run
without scoring it again, which also works where its gold is no longer cached.
It summarises the score files as they stand, including any left by an earlier
scoring of the same directory.

## Where a document carries the abstract in more than one language

A JATS document may hold its abstract in several languages, as repeated
`<abstract>` elements or as `<abstract>` plus `<trans-abstract>`. Two fields read
that, and both are scored:

| field | gold | prediction |
| --- | --- | --- |
| `abstract` | the article's own abstract, always one value | the abstract it filed as the article's own |
| `abstract_anywhere` | the article's own abstract, always one value | any abstract it carries |
| `abstract_any_language` | every language the document carries it in, the article's own first | any abstract it carries |
| `abstract_all_languages` | the same | every abstract it carries, paired with the gold |
| `abstract_language` | the language each abstract is declared to be in | the language it declared for the abstract that matches |

`abstract` is the headline and is literal: the article's own abstract on both
sides. `abstract_anywhere` credits a tool that finds the article's own abstract
but files it as the translation, which a prediction can only do once it carries
more than one abstract — until then the two rows are the same number, and after
that the gap between them is how often a tool filed it under the wrong element.

The article's own abstract is the first `<abstract>` in document order. A
`<trans-abstract>` is a translation whatever it declares; a later `<abstract>` is
one only if it carries the same `abstract-type` and declares a different
`@xml:lang`. Anything else — a second abstract in the same language, a
`plain-language-summary` — is not another language and is left out of both fields.

`abstract_any_language` credits any language, so the gap between it and the rows
above is what not requiring the article's own is worth.

`abstract_all_languages` asks the opposite question — did the prediction reproduce
every language the document carries — and is the only abstract row that charges
for an abstract the gold has none of. A format that can hold one abstract cannot
score well on it, which is why it reads low on the multilingual corpora today. Its
denominator counts values rather than documents, so a paper carrying four
languages weighs four times one carrying a single abstract, and its figure is not
comparable with the rows above it. Below each corpus table, and in the
comparison report, a block counts how often the credited one was a translation,
per corpus: a run that reads the translation of every multilingual paper scores
like one that reads the article's own, and that block is where it shows.

`abstract_language` scores the declaration rather than the text, and is empty
unless the run's profile sets `abstracts_mode` to `variants` or
`merged_by_language`: the shipped default declares no language. The
`grobid_crf_abstract_variants` and `grobid_crf_abstract_merged_by_language`
profiles are the default with that one setting changed, so a run against either
and a run against the default differ in nothing else. The two sides are paired
by the abstract's text, since neither the order nor the count can be relied on,
and a gold abstract declaring no language is left out of the
comparison rather than counted as a miss: `biorxiv`, `ore` and `pkp` declare
none at all, and `scielo_br` and `scielo_preprints-jats` declare one on
`<trans-abstract>` and not on the `<abstract>` beside it. Its denominator is
pairs rather than documents, so a document carrying three declared languages
weighs three times one carrying a single one, and the figure is not comparable
with the rows above.

`abstract_legacy` is the measure these replaced: every `<abstract>` element
joined into one string, with `<trans-abstract>` unread. It is scored so that a
figure published before the change can be reproduced by a current run — re-score
the stored predictions and read that row — rather than only restated. A
comparison whose runs scored a field differently says so above its tables.

A second block counts predictions that returned several languages as one value,
which a per-language score can only half match and which reads as a poor
extraction rather than as the unsegmented one it is.

**On `scielo_mx` the article's own abstract is a reading of document order, not
of anything the document declares.** Its 59 multilingual documents repeat
`<abstract>` — so no tag says which is the translation — and none declares an
article language, so the first one printed is taken as the article's own: Spanish
on 50 of them and English on 9. `scielo_br` and `scielo_preprints-jats` are not
affected, since `<trans-abstract>` names the translation outright, and `ore`
declares an article language. Read that corpus's translation count as a signal
rather than a verdict.

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

Or from GitHub: **Actions** → **Generate LLM predictions** → **Run workflow**,
on `main`. Leave the checkpoint blank to use the `eval.yml` version; `mode` and
`split` pick the sample.

The checkpoint defaults to the `version:` `eval.yml` gives the tool, so
generation stores under the version the benchmark reads. What the store already
has is fetched rather than regenerated.

**Check the service is serving the checkpoint you are storing under.** The run
logs what `/health` reports and records it as `served_model`, but cannot refuse
a mismatch — a served-model label and a checkpoint name never match textually.

When it finishes, read the job summary's ok/error counts rather than the green
tick: a failed document is recorded, not raised. Note this column can never
cover `plos-manuscripts`, so on `main` it costs that corpus from Overall.
