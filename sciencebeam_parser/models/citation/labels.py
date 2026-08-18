from typing import AbstractSet, FrozenSet


# The label set the citation model is trained and served against, matching GROBID's
# citation corpus (grobid-trainer/resources/dataset/citation).
#
# Every identifier carries IDENTIFIER_LABEL, whatever kind it is; the kind is detected
# from the text at extraction, so no label depends on a regex having matched.
IDENTIFIER_LABEL = '<pubnum>'

OTHER_LABEL = '<other>'

CITATION_LABELS: FrozenSet[str] = frozenset({
    '<author>',
    '<booktitle>',
    '<collaboration>',
    '<date>',
    '<editor>',
    '<institution>',
    '<issue>',
    '<journal>',
    '<location>',
    '<note>',
    OTHER_LABEL,
    '<pages>',
    '<publisher>',
    '<series>',
    '<tech>',
    '<title>',
    '<volume>',
    '<web>',
    IDENTIFIER_LABEL
})

# Labels with no semantic counterpart, which CitationSemanticExtractor keeps as notes.
# Anything else in CITATION_LABELS has to map to semantic content.
NOTE_CITATION_LABELS: AbstractSet[str] = frozenset({
    '<booktitle>',
    '<collaboration>',
    '<institution>',
    '<note>',
    '<series>',
    '<tech>'
})
