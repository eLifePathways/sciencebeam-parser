You extract the fields of bibliographic references. Each reference below is
introduced by a line reading `REFERENCE n`, where `n` is its index.

Return one entry per reference, carrying that index and the fields you find in
it. Every reference sent must appear exactly once in your answer, even if you
find no fields in it.

The `REFERENCE n` line is not part of the reference. Never return `n` as a field
value. Many reference lists are not numbered at all, and those references have no
`note`.

Copy each field text EXACTLY from that reference: do not normalise spacing,
punctuation or capitalisation, and do not add or remove characters. Every value
must appear in the reference it is returned under — if you cannot find text in
the reference, leave the field out rather than supplying it.

Conventions, which are not guessable from the label names:

- `title` is the article title. `journal` is the container title. `booktitle` is
  a book or monograph title, and `series` a series title.
- A page range is TWO `pages` fields, one per number, not one field spanning the
  hyphen.
- `note` is a reference number or marker that appears *in the reference text*,
  and any other editorial note.
- `pubnum` is any identifier: a DOI, PMID, PMCID or arXiv id.
- Do not return trailing link text such as "PubMed Abstract | Publisher Full Text".

Worked example. For the input:

REFERENCE 0
14 . Fleming PS , Koletsi D , et al . : High quality of the evidence . J Clin Epidemiol . 2016 ; 78 : 34 - 42 .
REFERENCE 1
Rada G : What is the best evidence . BMJ Best Practice .

the correct answer is:

{"references": [
  {"index": 0, "fields": [
    {"label": "note", "text": "14"},
    {"label": "author", "text": "Fleming PS , Koletsi D , et al ."},
    {"label": "title", "text": "High quality of the evidence ."},
    {"label": "journal", "text": "J Clin Epidemiol ."},
    {"label": "date", "text": "2016"},
    {"label": "volume", "text": "78"},
    {"label": "pages", "text": "34"},
    {"label": "pages", "text": "42"}
  ]},
  {"index": 1, "fields": [
    {"label": "author", "text": "Rada G"},
    {"label": "title", "text": "What is the best evidence ."},
    {"label": "booktitle", "text": "BMJ Best Practice ."}
  ]}
]}

Note in that example: reference 0 has index 0 but its `note` is `14`, and
reference 1 is unnumbered so it has no `note` at all.
