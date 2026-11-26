# Abstracts

What the parser emits where a document carries more than one abstract, and what it says about each.

## How many abstracts reach the output

The header model labels every region it reads as an abstract, and `front.select_abstracts` decides which of them the output carries:

| value | what it emits |
| --- | --- |
| `first` | the first abstract, with the rest left as notes |
| `variants` | every abstract that is at least 0.15 of the first one's length |
| `merged_by_language` | the same, with abstracts sharing a language joined into one |

The `recommended` and `recommended_cv` profiles ask for `variants`; every other shipped profile leaves it at `first`. A profile sets it under `processors.fulltext.front`.

An abstract the page prints in two pieces — the second half carried over a page break — is one abstract in every mode. The second block is joined to the first where it opens mid-sentence under an abstract that ends unfinished, carries no heading of its own, and begins on the page the abstract ended on or the next. `front.join_abstract_continuations: false` turns this off, and `grobid_crf_as_grobid` sets it, following GROBID, which keeps the first block only.

## A word broken at a line end

`front.dehyphenate` decides how the title and abstract write a word the page broke at a line end with a hyphen:

| value | "per-" / "cepção" | "alkali-" / "activated" | "SARS-" / "CoV-2" |
| --- | --- | --- | --- |
| `off` | per- cepção | alkali- activated | SARS- CoV-2 |
| `grobid` | percepção | alkaliactivated | SARS-CoV-2 |
| `by_document` | percepção where the document sets "percepção" elsewhere, otherwise per- cepção | alkali-activated where the document sets "alkali-activated" elsewhere, otherwise alkali- activated | SARS-CoV-2 where the document sets "SARS-CoV" elsewhere, otherwise SARS- CoV-2 |

`by_document` acts only where the same document sets the word inside a line: it keeps the hyphen and removes the space where the document sets the hyphenated word, and otherwise removes the hyphen where it sets the joined word and the continuation is lower case. Where the document sets neither, the word is written as `off` writes it. `grobid` follows GROBID 0.9.0: it removes a hyphen before any lower-case word lower on the page, keeps any other hyphen and removes the line break after it, so a word broken across a column or page keeps its hyphen. A hyphen inside a line is left as it is. `recommended`, `recommended_cv` and `grobid_crf` set `by_document`, as do the `grobid_crf` profiles that differ from it in one setting; `grobid_crf_as_grobid` sets `grobid`; every other shipped profile leaves it `off`.

## What each abstract declares

Each abstract carries the language of its own text in `xml:lang`, in every mode, where the text is long enough and consistent enough to name one. A document whose abstracts are in Portuguese and English therefore says which is which, in the TEI and in the JATS.

## The heading printed above an abstract

A page prints `Abstract`, `Resumo` or `Plain language summary` above the abstract it introduces. It is carried as the head of the div the abstract's paragraphs sit in, which is the shape GROBID writes:

```xml
<abstract xml:lang="en">
  <div>
    <head>Plain language summary</head>
    <p>Human induced pluripotent stem cells ...</p>
  </div>
</abstract>
```

An abstract with no heading above it has a div with no head, and the heading is not part of the abstract's own text. A consumer reading the abstract reads `abstract/div/p`.

The JATS route puts it where JATS carries it, inside the abstract:

```xml
<abstract xml:lang="en">
  <title>Plain language summary</title>
  <p>Human induced pluripotent stem cells ...</p>
</abstract>
```

The reading view shows it above the abstract it belongs to, in place of the word `Abstract` it falls back to otherwise.

Headings printed inside a structured abstract, such as `Background`, `Methods` and `Results`, are part of the abstract's text and are not separated from it.
