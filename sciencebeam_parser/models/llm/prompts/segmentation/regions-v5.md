The lines below are one scientific article, in reading order. Each line is prefixed with its line
number and a tab, and a blank line separates one block of text from the next. Blank lines carry no
line number of their own.

Split the article into regions. For each region, give the line it starts on, the line it ends on,
and its label:

- `front_matter` — title, authors, affiliations, abstract and keywords
- `body` — the article text, including section headings, figures, tables and their captions
- `acknowledgements` — funding, thanks and contribution statements
- `appendix` — appendices and supplementary material
- `references` — the bibliography, including its heading
- `other` — running heads, footers and page numbers

A region covers every line from its start to its end, and may span many blocks. The shape of an
answer, with numbers chosen only to show how regions meet:

```
{"regions": [
  {"start": 1, "end": 3, "label": "front_matter"},
  {"start": 4, "end": 9, "label": "body"}
]}
```

Line 3 is the last line of the front matter and line 4 is the first line of the body.
