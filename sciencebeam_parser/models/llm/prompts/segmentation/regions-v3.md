The lines below are one scientific article, in reading order. Each line is prefixed with its line
number and a tab. `--- page n ---` marks where a new page begins, a blank line where a new block
begins, and `**bold**` and `*italic*` mark how the line is set. Those markers carry no line number
of their own.

Split the article into regions. For each region, give the line it starts on, the line it ends on,
and its label:

- `front_matter` — title, authors, affiliations, abstract and keywords
- `body` — the article text, including section headings, figures, tables and their captions
- `acknowledgements` — funding, thanks and contribution statements
- `appendix` — appendices and supplementary material
- `references` — the bibliography, including its heading
- `other` — running heads, footers and page numbers

A region covers every line from its start to its end, and usually runs over several pages. An
article has a handful of regions, one for each stretch of a single kind of content — a body of forty
pages is one region, not forty. Start a new region where the kind of content changes, which may be
in the middle of a page.

For an article of {{last_line}} lines, an answer looks like this:

```
{"regions": [
  {"start": 1, "end": 11, "label": "front_matter"},
  {"start": 12, "end": 480, "label": "body"},
  {"start": 481, "end": 486, "label": "acknowledgements"},
  {"start": 487, "end": {{last_line}}, "label": "references"}
]}
```

Line 11 is the last line of the front matter and line 12 is the first line of the body.
