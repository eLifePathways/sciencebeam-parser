from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL

_REFERENCE_MODEL_LABELS: Dict[str, frozenset] = {
    'segmentation':        frozenset({'<references>'}),
    'reference-segmenter': frozenset({'<reference>'}),
}

# A reference reaches the output only if the citation model gives it one of these: without
# one, CitationSemanticExtractor yields SemanticInvalidReference, which no TEI or JATS factory
# renders.  So the citation model decides which references exist, and with them which one is
# first -- which is why first_reference_text moves when only the citation model changes.
REFERENCE_PRESENCE_LABELS: FrozenSet[str] = frozenset({
    '<author>',
    '<editor>',
    '<journal>',
    '<title>',
    '<web>',
    IDENTIFIER_LABEL,
})

_HEADER_MODEL_LABELS: Dict[str, frozenset] = {
    'segmentation': frozenset({'<header>'}),
}

MODEL_RELEVANT_LABELS: Dict[str, Dict[str, frozenset]] = {
    'reference_doi':        {**_REFERENCE_MODEL_LABELS,
                             'citation': frozenset({IDENTIFIER_LABEL, '<web>'})},
    'reference_title':      {**_REFERENCE_MODEL_LABELS, 'citation': frozenset({'<title>'})},
    'first_reference_text': {**_REFERENCE_MODEL_LABELS,
                             'citation': REFERENCE_PRESENCE_LABELS},
    'title':                {**_HEADER_MODEL_LABELS, 'header': frozenset({'<title>'})},
    'abstract':             {**_HEADER_MODEL_LABELS, 'header': frozenset({'<abstract>'})},
    'keywords':             {**_HEADER_MODEL_LABELS, 'header': frozenset({'<keyword>'})},
    'author_full_names':    {**_HEADER_MODEL_LABELS, 'header': frozenset({'<author>'}),
                             'name-header': frozenset({'<forenames>', '<surname>'})},
    'affiliation_text':     {**_HEADER_MODEL_LABELS, 'header': frozenset({'<affiliation>'})},
    'body_section_titles':  {'segmentation': frozenset({'<body>'}),
                             'fulltext': frozenset({'<section>'})},
    'acknowledgement':      {'segmentation': frozenset({'<acknowledgement>'}),
                             'fulltext': frozenset({'<acknowledgement>'})},
}

FIELD_MODEL: Dict[str, str] = {
    'title':                'header',
    'abstract':             'header',
    'author_full_names':    'name-header',
    'affiliation_text':     'affiliation-address',
    'keywords':             'header',
    'body_section_titles':  'fulltext',
    'acknowledgement':      'fulltext',
    'first_reference_text': 'citation',
    'reference_title':      'citation',
    'reference_doi':        'citation',
}

MODEL_PARENT: Dict[str, str] = {
    'header':              'segmentation',
    'name-header':         'header',
    'affiliation-address': 'header',
    'fulltext':            'segmentation',
    'reference-segmenter': 'segmentation',
    'citation':            'reference-segmenter',
}


def _get_model_chain(analysis_field: str) -> List[str]:
    chain = []
    model: Optional[str] = FIELD_MODEL[analysis_field]
    while model is not None:
        chain.append(model)
        model = MODEL_PARENT.get(model)
    return list(reversed(chain))
