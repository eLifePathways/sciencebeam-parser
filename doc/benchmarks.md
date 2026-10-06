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

The checkpoint defaults to the `version:` `eval.yml` gives the tool, so
generation stores under the version the benchmark reads. What the store already
has is fetched rather than regenerated.

**Check the service is serving the checkpoint you are storing under.** The run
logs what `/health` reports and records it as `served_model`, but cannot refuse
a mismatch — a served-model label and a checkpoint name never match textually.

When it finishes, read the job summary's ok/error counts rather than the green
tick: a failed document is recorded, not raised. Note this column can never
cover `plos-manuscripts`, so on `main` it costs that corpus from Overall.
