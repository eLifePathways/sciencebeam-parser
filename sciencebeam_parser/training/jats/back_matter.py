"""Which kind of back-matter section a JATS element is.

An article states its availability, its funding, who contributed and what it
declares in sections the JATS marks in several different ways, or does not mark
at all.  This is where those are recognised, and where a section whose heading
alone cannot decide it is read against where it sits.
"""
import re
import unicodedata
from typing import FrozenSet, NamedTuple, Optional, Set, Tuple

from lxml import etree

from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames


def element_text(el: etree._Element) -> str:
    return ' '.join(' '.join(el.itertext()).split())


_AVAILABILITY_SEC = '@sec-type="data-availability"'

# Publishers that do not set `sec-type` still give the section one of a handful
# of headings, and only ever as a top-level section.
_AVAILABILITY_TITLES = frozenset({
    'data availability',
    'software availability',
    'code availability',
    'data and software availability',
    'availability of data',
    'availability of data and material',
    'availability of data and materials',
})


def _fold(text: str) -> str:
    """Lower-cased and stripped of accents, so one spelling stands for all."""
    stripped = unicodedata.normalize('NFKD', text.lower().replace('\u2019', "'"))
    return ''.join(c for c in stripped if not unicodedata.combining(c))


def _normalised_section_title(el: etree._Element) -> str:
    title_el = el.find('title')
    if title_el is None:
        return ''
    return re.sub(r'\s+statement$', '', _fold(element_text(title_el)).strip(' .:'))


_CONTRIBUTION_SEC = (
    '@sec-type="author-contributions"'
    ' or @sec-type="author-contribution"'
    ' or @sec-type="contributions"'
)

_CONFLICT_SEC = (
    '@sec-type="conflict"'
    ' or @sec-type="COI-statement"'
    ' or @sec-type="coi-statement"'
)

_CONFLICT_TITLES = frozenset({
    'conflict of interest',
    'conflicts of interest',
    'competing interests',
    'declaration of competing interest',
    'conflito de interesse',
    'conflitos de interesse',
    'conflicto de intereses',
    'conflictos de interes',
})

_ETHICS_SEC = '@sec-type="ethics"' + " or @sec-type='ethics-statement'"

# GROBID has no label of its own for these.  Its corpus files one of the
# headings below under `<div type="annex">` 61 times and as `<body>` 39, so the
# heading alone does not decide: what separates them is where the section sits.
# `_get_back_matter_ethics_sections` is what applies that.
_ETHICS_TITLES = frozenset({
    'ethics',
    'ethics and consent',
    'ethics approval',
    'ethics approval and consent to participate',
    'ethical approval',
    'ethical considerations',
    'ethical statement',
    'ethics statement',
    'consent',
    'consent to participate',
    'consent to publish',
    'consent for publication',
    'patient consent for publication',
    'institutional review board',
    'aspectos eticos',
    'consideracoes eticas',
    'consideraciones eticas',
    'comite de etica',
    'aprovacao do cep',
})

_FUNDING_SEC = '@sec-type="funding"'

_FUNDING_TITLES = frozenset({
    'funding',
    'funding information',
    'grant information',
    'financial support',
    'financial disclosure',
    'suporte financeiro',
    'apoio financeiro',
    'fonte de financiamento',
    'financiamiento',
    'apoyo financiero',
})

# A footnote group is the other place the publisher puts these.  `fn-type` is
# meant to say which, but is only sometimes set and is sometimes set wrong --
# one preprint files its conflict-of-interest declaration under "con" -- so a
# heading set in bold decides first, in this order.  Only a bold heading: a
# numbered page footnote opening "1 This project has received funding from"
# names no statement, it just mentions one.
_FN_KIND_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('conflict', ('conflict', 'conflito', 'conflicto', 'competing interest')),
    # One stem for financial / financiamiento / financiacion / financeiro;
    # conflict is matched first, so "financial competing interests" is still a
    # conflict.
    ('funding', ('funding', 'financ', 'grant', 'apoyo financiero')),
    ('contribution', ('contribution', 'contribuicao', 'contribucion',
                      'colabora', 'authorship')),
    ('acknowledgement', ('acknowledg', 'agradecimento', 'agradecimiento')),
    # The same statement the `data-availability` sec-type marks, filed as a
    # footnote instead -- which is where a preprint server puts it.
    ('availability', ('data availability', 'disponibilidad de datos',
                      'disponibilidade dos dados', 'disponibilidade de dados',
                      'availability of data')),
    ('ethics', ('ethic', 'etico', 'etica', 'consent', 'comite de etica',
                'institutional review')),
)

_FN_KIND_BY_TYPE = {
    'con': 'contribution',
    'conflict': 'conflict',
    'coi-statement': 'conflict',
    'COI-statement': 'conflict',
    'financial-disclosure': 'funding',
    'supported-by': 'funding',
}

_FN_HEADING_LENGTH = 80


def match_fn_keywords(heading: str) -> Optional[str]:
    folded = _fold(heading)[:_FN_HEADING_LENGTH]
    for kind, keywords in _FN_KIND_KEYWORDS:
        if any(keyword in folded for keyword in keywords):
            return kind
    return None


def _get_fn_heading(el: etree._Element) -> Optional[str]:
    """What the footnote calls itself, wherever the publisher wrote it.

    Usually a bold line opening the note.  Sometimes the whole group carries
    one title and each note under it is bare.  Failing both, the note's own
    opening words serve -- unless it opens with its number, which is what a
    numbered page note does and a statement never does: of the notes that
    carry neither a bold heading nor an `fn-type`, 94 of 101 open with a digit.
    """
    bold = el.find('.//bold')
    if bold is not None:
        return element_text(bold)
    group = el.getparent()
    group_title = group.find('title') if group is not None else None
    if (
        group_title is not None
        and element_text(group_title)
        and group.find('fn') is el
    ):
        # The title heads the note that follows it, not every note in the group:
        # one group titled for its conflict statement also holds the citation.
        return element_text(group_title)
    text = element_text(el)
    return None if re.match(r'^\d', text) else text


def get_fn_kind(el: etree._Element) -> Optional[str]:
    """Which statement a footnote makes, by its heading first and its type second."""
    heading = _get_fn_heading(el)
    by_heading = match_fn_keywords(heading) if heading else None
    return by_heading or _FN_KIND_BY_TYPE.get(el.get('fn-type') or '')


_CONTRIBUTION_TITLES = frozenset({
    'author contributions',
    "authors' contributions",
    'author contribution',
    "author's contribution",
    'contributions of authors',
    'credit authorship contribution',
    'colaboradores',
    'colaboracao',
    'colaboracoes',
    'contribuicao dos autores',
    'contribuicoes dos autores',
    'contribucion de los autores',
    'contribuciones de los autores',
    # The heading an article gives the section when it names who took part
    # rather than what they did.
    'contributors',
})


def _get_sections(
    root: etree._Element,
    sec_type_predicate: str,
    titles: FrozenSet[str],
    fn_kind: Optional[str] = None,
    any_depth: bool = False,
) -> Set[etree._Element]:
    """Sections of one kind, however the JATS happens to say so.

    A section nested inside another is claimed only for a kind that goes on to
    test where it sits: "Author contributions" under Methods is the article
    describing its method, while an article can nest its declarations inside
    the last section it argues, which is where they print rather than what they
    are.
    """
    sections = set(root.xpath(f'//sec[{sec_type_predicate}]'))
    step = '//' if any_depth else '/'
    for el in root.xpath(f'(body|back){step}sec[not(@sec-type)]'):
        if _normalised_section_title(el) in titles:
            sections.add(el)
    if fn_kind is not None:
        for el in root.xpath('back//fn'):
            if get_fn_kind(el) == fn_kind:
                sections.add(el)
    return sections


def _get_availability_sections(root: etree._Element) -> Set[etree._Element]:
    return _get_sections(
        root, _AVAILABILITY_SEC, _AVAILABILITY_TITLES, 'availability'
    )


def _get_contribution_sections(root: etree._Element) -> Set[etree._Element]:
    return _get_sections(
        root, _CONTRIBUTION_SEC, _CONTRIBUTION_TITLES, 'contribution'
    )


def _get_conflict_sections(root: etree._Element) -> Set[etree._Element]:
    return _get_sections(root, _CONFLICT_SEC, _CONFLICT_TITLES, 'conflict')


def _get_funding_sections(root: etree._Element) -> Set[etree._Element]:
    return _get_sections(root, _FUNDING_SEC, _FUNDING_TITLES, 'funding')


def _get_ethics_sections(root: etree._Element) -> Set[etree._Element]:
    return _get_sections(root, _ETHICS_SEC, _ETHICS_TITLES, 'ethics', any_depth=True)


# A named declaration GROBID files under `<div type="annex">`.
_NOTICE_TITLES = frozenset({
    'disclaimer',
    'aviso legal',
    'nota legal',
    "publisher's note",
    'publishers note',
    'patents',
    'patentes',
    'abbreviations',
    'abreviaturas',
    'abreviacoes',
    'declarations',
    'declaraciones',
    'declaracoes',
})


def _get_notice_sections(root: etree._Element) -> Set[etree._Element]:
    return {
        el for el in root.xpath('(body|back)//sec[not(@sec-type)]')
        if _normalised_section_title(el) in _NOTICE_TITLES
    }


def _get_back_matter_ethics_sections(
    root: etree._Element,
    already_placed: Set[etree._Element],
) -> Set[etree._Element]:
    """Ethics and notice sections that sit in the back matter, not the argument.

    An article states its ethics either way: as a section of its methods, with
    results still to come, or among the declarations at the end.  The heading
    reads the same in both, so what follows the section decides.  One under
    `back` is already past the argument; one under `body` counts only where
    nothing but other back-matter sections follows it.

    `already_placed` is what the clearer kinds have taken, which is why they are
    matched first: they are how this one recognises the run it is in.
    """
    sections = _get_ethics_sections(root) | _get_notice_sections(root)
    back_matter = already_placed | sections
    ordered = root.xpath('body//sec')
    position_by_section = {el: index for index, el in enumerate(ordered)}

    def _is_back_matter(el: etree._Element, candidate: etree._Element) -> bool:
        """A section of the candidate's own, or one a back-matter section holds."""
        if candidate in back_matter or candidate in el.iterdescendants('sec'):
            return True
        return any(
            ancestor in back_matter for ancestor in candidate.iterancestors('sec')
        )

    return {
        el for el in sections
        if el not in position_by_section
        or all(
            _is_back_matter(el, following)
            for following in ordered[position_by_section[el] + 1:]
        )
    }


def is_bold_heading(el: etree._Element) -> bool:
    """Whether a paragraph is nothing but a bold heading.

    A footnote states its kind in bold, sometimes as a paragraph of its own and
    sometimes opening the paragraph it belongs to.  Only the first is a title;
    the second is the text, heading and all.
    """
    paragraph = el.getparent() if el.tag == 'bold' else el
    if paragraph is None:
        return False
    bold = paragraph.find('.//bold')
    text = element_text(paragraph)
    return bool(text) and bold is not None and text == element_text(bold)


def _is_within(el: etree._Element, elements: Set[etree._Element]) -> bool:
    return any(ancestor in elements for ancestor in el.iterancestors()) or el in elements


def statement_text(el: etree._Element) -> str:
    """A footnote's declaration without the words introducing it.

    A publisher writes the heading inside the note, in bold, and the page prints
    it as a lead-in.  GROBID labels the lead-in `<other>` and only what follows
    it as the declaration, on 110 of its 835 header documents.
    """
    text = element_text(el)
    bold = el.find('.//bold')
    heading = element_text(bold) if bold is not None else ''
    if heading and text.startswith(heading):
        return text[len(heading):].strip()
    return text


class LabelledSections(NamedTuple):
    """The sections of the body or back matter the model has its own label for."""
    supplementary: Set[etree._Element]
    availability: Set[etree._Element]
    contribution: Set[etree._Element]
    conflict: Set[etree._Element]
    funding: Set[etree._Element]
    acknowledgement: Set[etree._Element]
    ethics: Set[etree._Element]
    footnote: Set[etree._Element]

    @staticmethod
    def of(root: etree._Element) -> 'LabelledSections':
        availability = _get_availability_sections(root)
        contribution = _get_contribution_sections(root)
        conflict = _get_conflict_sections(root)
        funding = _get_funding_sections(root)
        acknowledgement = {
            el for el in root.xpath('back//fn')
            if get_fn_kind(el) == 'acknowledgement'
        }
        return LabelledSections(
            supplementary=set(
                root.xpath('//sec[@sec-type="supplementary-material"]')
            ),
            availability=availability,
            contribution=contribution,
            conflict=conflict,
            funding=funding,
            acknowledgement=acknowledgement,
            ethics=_get_back_matter_ethics_sections(
                root,
                already_placed=(
                    availability | contribution | conflict | funding
                    | acknowledgement
                ),
            ),
            footnote=set(root.xpath('back//fn')),
        )

    def _kind(self, el: etree._Element) -> Optional[str]:
        for kind in (
            'supplementary', 'availability', 'contribution', 'conflict',
            'funding', 'acknowledgement', 'ethics', 'footnote',
        ):
            if _is_within(el, getattr(self, kind)):
                return kind
        return None

    def title_field(self, el: etree._Element, default: str) -> str:
        kind = self._kind(el)
        return default if kind is None else TITLE_FIELD_BY_KIND[kind]

    def paragraph_field(self, el: etree._Element, default: str) -> str:
        kind = self._kind(el)
        return default if kind is None else PARAGRAPH_FIELD_BY_KIND[kind]

    def claims(self, el: etree._Element) -> bool:
        return self._kind(el) is not None


TITLE_FIELD_BY_KIND = {
    # An appendix the JATS files under `body` rather than under `app`.
    'supplementary': JatsFieldNames.APPENDIX_GROUP_TITLE,
    'availability': JatsFieldNames.AVAILABILITY_SECTION_TITLE,
    'contribution': JatsFieldNames.CONTRIBUTION_SECTION_TITLE,
    'conflict': JatsFieldNames.CONFLICT_SECTION_TITLE,
    'funding': JatsFieldNames.FUNDING_SECTION_TITLE,
    'acknowledgement': JatsFieldNames.ACK_SECTION_TITLE,
    'ethics': JatsFieldNames.BACK_SECTION_TITLE,
    # A footnote that names no statement is a numbered page footnote, and the
    # number it opens with is not a heading.
    'footnote': JatsFieldNames.BACK_FOOTNOTE,
}

# Where a declaration filed among the author notes goes instead.  The front
# matter is where the JATS puts it and where the page prints it, so it is the
# header's to label rather than a region of its own.  Only these two kinds: the
# header model has a label for a conflict and a contribution and for none of the
# others.
NOTE_PARAGRAPH_FIELD_BY_KIND = {
    'contribution': JatsFieldNames.CONTRIBUTION_NOTE,
    'conflict': JatsFieldNames.CONFLICT_NOTE,
}


PARAGRAPH_FIELD_BY_KIND = {
    'supplementary': JatsFieldNames.APPENDIX,
    'availability': JatsFieldNames.AVAILABILITY_SECTION_PARAGRAPH,
    'contribution': JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
    'conflict': JatsFieldNames.CONFLICT_SECTION_PARAGRAPH,
    'funding': JatsFieldNames.FUNDING_SECTION_PARAGRAPH,
    'acknowledgement': JatsFieldNames.ACK_SECTION_PARAGRAPH,
    'ethics': JatsFieldNames.BACK_SECTION_PARAGRAPH,
    'footnote': JatsFieldNames.BACK_FOOTNOTE,
}
