import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union
)

from lxml import etree

from sciencebeam_parser.utils.xml_writer import TagExpression
from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressLine,
    SemanticAuthor,
    SemanticContentWrapper,
    SemanticCountry,
    SemanticDepartment,
    SemanticExternalIdentifier,
    SemanticExternalUrl,
    SemanticFigure,
    SemanticFigureCitation,
    SemanticGivenName,
    SemanticGraphic,
    SemanticHeading,
    SemanticInstitution,
    SemanticBookTitle,
    SemanticIssue,
    SemanticJournal,
    SemanticLabel,
    SemanticLaboratory,
    SemanticLocation,
    SemanticMarker,
    SemanticMiddleName,
    SemanticMixedNote,
    SemanticNameSuffix,
    SemanticNameTitle,
    SemanticNote,
    SemanticOptionalValueSemanticMixedContentWrapper,
    SemanticPageRange,
    SemanticParagraph,
    SemanticPostBox,
    SemanticPostCode,
    SemanticPublisher,
    SemanticRawEditors,
    SemanticRawEquation,
    SemanticReferenceCitation,
    SemanticRegion,
    SemanticSection,
    SemanticSettlement,
    SemanticSurname,
    SemanticTable,
    SemanticTableCitation,
    SemanticTextContentWrapper,
    SemanticTitle,
    SemanticVolume
)
from sciencebeam_parser.document.tei.common import (
    TEI_E,
    create_tei_note_element,
    extend_element,
    get_default_attributes_for_semantic_content,
    iter_layout_block_tei_children,
    parse_tag_expression
)
from sciencebeam_parser.document.tei.factory import (
    T_ElementChildrenList,
    TeiElementFactory,
    TeiElementFactoryContext
)
from sciencebeam_parser.document.tei.misc import (
    SemanticMixedNoteTeiElementFactory
)
from sciencebeam_parser.document.tei.citation import (
    CitationTeiElementFactory
)
from sciencebeam_parser.document.tei.external_identifiers import (
    ExternalIdentifierTeiElementFactory
)
from sciencebeam_parser.document.tei.equation import (
    RawEquationTeiElementFactory
)
from sciencebeam_parser.document.tei.figure_table import (
    FigureTeiElementFactory,
    TableTeiElementFactory
)
from sciencebeam_parser.document.tei.graphic import (
    GraphicTeiElementFactory
)
from sciencebeam_parser.document.tei.page_range import (
    TeiBiblScopeForPageRangeElementFactory
)
from sciencebeam_parser.document.tei.section import (
    HeadingTeiElementFactory,
    ParagraphTeiElementFactory,
    SectionTeiElementFactory
)


LOGGER = logging.getLogger(__name__)


SIMPLE_TAG_EXPRESSION_BY_SEMANTIC_CONTENT_CLASS = {
    SemanticNameTitle: 'roleName',
    SemanticGivenName: 'forename[@type="first"]',
    SemanticMiddleName: 'forename[@type="middle"]',
    SemanticSurname: 'surname',
    SemanticNameSuffix: 'genName',
    SemanticRawEditors: 'editor',
    SemanticLabel: 'label',
    SemanticMarker: 'note[@type="marker"]',
    SemanticInstitution: 'orgName[@type="institution"]',
    SemanticDepartment: 'orgName[@type="department"]',
    SemanticLaboratory: 'orgName[@type="laboratory"]',
    SemanticAddressLine: 'addrLine',
    SemanticPostCode: 'postCode',
    SemanticPostBox: 'postBox',
    SemanticSettlement: 'settlement',
    SemanticRegion: 'region',
    SemanticCountry: 'country',
    SemanticJournal: 'title[@level="j"]',
    SemanticBookTitle: 'title[@level="m"]',
    SemanticVolume: 'biblScope[@unit="volume"]',
    SemanticIssue: 'biblScope[@unit="issue"]',
    SemanticPublisher: 'publisher',
    SemanticLocation: 'addrLine',
    SemanticExternalUrl: 'ref[@type="url"]'
}


PARSED_TAG_EXPRESSION_BY_SEMANTIC_CONTENT_CLASS: Dict[type, TagExpression] = {
    key: parse_tag_expression(value)
    for key, value in SIMPLE_TAG_EXPRESSION_BY_SEMANTIC_CONTENT_CLASS.items()
}


PARENT_PATH_BY_SEMANTIC_CONTENT_CLASS = {
    SemanticTitle: ['analytic'],
    SemanticAuthor: ['analytic'],
    SemanticRawEditors: ['monogr'],
    SemanticExternalIdentifier: ['analytic'],
    SemanticJournal: ['monogr'],
    SemanticBookTitle: ['monogr'],
    SemanticVolume: ['monogr', 'imprint'],
    SemanticIssue: ['monogr', 'imprint'],
    SemanticPageRange: ['monogr', 'imprint'],
    SemanticPublisher: ['monogr', 'imprint'],
    SemanticLocation: ['monogr', 'meeting', 'address']
}


ELEMENT_FACTORY_BY_SEMANTIC_CONTENT_CLASS: Mapping[
    Type[Any],
    TeiElementFactory
] = {
    SemanticMixedNote: SemanticMixedNoteTeiElementFactory(),
    SemanticPageRange: TeiBiblScopeForPageRangeElementFactory(),
    SemanticExternalIdentifier: ExternalIdentifierTeiElementFactory(),
    SemanticFigure: FigureTeiElementFactory(),
    SemanticTable: TableTeiElementFactory(),
    SemanticFigureCitation: CitationTeiElementFactory(),
    SemanticTableCitation: CitationTeiElementFactory(),
    SemanticReferenceCitation: CitationTeiElementFactory(),
    SemanticHeading: HeadingTeiElementFactory(),
    SemanticParagraph: ParagraphTeiElementFactory(),
    SemanticSection: SectionTeiElementFactory(),
    SemanticRawEquation: RawEquationTeiElementFactory(),
    SemanticGraphic: GraphicTeiElementFactory()
}


def get_tei_note_for_semantic_content(
    semantic_content: SemanticContentWrapper
) -> etree.ElementBase:
    note_type = 'other'
    if isinstance(semantic_content, SemanticNote):
        note_type = semantic_content.note_type
    else:
        note_type = 'other:' + type(semantic_content).__name__
    return create_tei_note_element(note_type, semantic_content.merged_block)


class DefaultTeiElementFactoryContext(TeiElementFactoryContext):
    def get_default_attributes_for_semantic_content(
        self,
        semantic_content: SemanticContentWrapper,
        **kwargs
    ) -> Dict[str, str]:
        return get_default_attributes_for_semantic_content(semantic_content, **kwargs)

    def iter_layout_block_tei_children(
        self,
        layout_block: LayoutBlock,
        enable_coordinates: bool = True
    ) -> Iterable[Union[str, etree.ElementBase]]:
        return iter_layout_block_tei_children(
            layout_block,
            enable_coordinates=enable_coordinates
        )

    def get_tei_child_elements_for_semantic_content(
        self,
        semantic_content: SemanticContentWrapper
    ) -> List[etree.ElementBase]:
        return get_tei_child_elements_for_semantic_content(
            semantic_content,
            context=self
        )

    def append_tei_children_list_and_get_whitespace(
        self,
        children: T_ElementChildrenList,
        semantic_content: SemanticContentWrapper,
        pending_whitespace: str
    ) -> str:
        return append_tei_children_list_and_get_whitespace(
            children,
            semantic_content=semantic_content,
            pending_whitespace=pending_whitespace,
            context=self
        )

    def get_parent_path_for_semantic_content(
        self,
        semantic_content: SemanticContentWrapper
    ) -> Optional[List[str]]:
        return PARENT_PATH_BY_SEMANTIC_CONTENT_CLASS.get(
            type(semantic_content)
        )


DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT = DefaultTeiElementFactoryContext()


def get_tei_child_elements_for_semantic_content(
    semantic_content: SemanticContentWrapper,
    context: TeiElementFactoryContext
) -> List[etree.ElementBase]:
    semantic_type = type(semantic_content)
    element_factory = ELEMENT_FACTORY_BY_SEMANTIC_CONTENT_CLASS.get(
        semantic_type
    )
    if element_factory is not None:
        return element_factory.get_tei_children_for_semantic_content(
            semantic_content,
            context=context
        )
    parsed_tag_expression = PARSED_TAG_EXPRESSION_BY_SEMANTIC_CONTENT_CLASS.get(
        semantic_type
    )
    if parsed_tag_expression:
        if (
            isinstance(semantic_content, SemanticOptionalValueSemanticMixedContentWrapper)
            and semantic_content.value is not None
        ):
            return [parsed_tag_expression.create_node(
                get_default_attributes_for_semantic_content(semantic_content),
                semantic_content.value,
                element_maker=TEI_E
            )]
        return [parsed_tag_expression.create_node(
            *iter_layout_block_tei_children(semantic_content.merged_block),
            element_maker=TEI_E
        )]
    return [get_tei_note_for_semantic_content(semantic_content)]


def get_tei_children_and_whitespace_for_semantic_content(
    semantic_content: SemanticContentWrapper,
    context: TeiElementFactoryContext
) -> Tuple[List[Union[dict, str, etree.ElementBase]], str]:
    layout_block = semantic_content.merged_block
    if isinstance(semantic_content, SemanticTextContentWrapper):
        return (
            list(iter_layout_block_tei_children(layout_block)),
            layout_block.whitespace
        )
    return (
        get_tei_child_elements_for_semantic_content(
            semantic_content,
            context=context
        ),
        layout_block.whitespace
    )


def append_tei_children_and_get_whitespace(
    parent: etree.ElementBase,
    semantic_content: SemanticContentWrapper,
    context: TeiElementFactoryContext
) -> str:
    children, whitespace = get_tei_children_and_whitespace_for_semantic_content(
        semantic_content,
        context=context
    )
    extend_element(parent, children)
    return whitespace


def append_tei_children_list_and_get_whitespace(
    children: T_ElementChildrenList,
    semantic_content: SemanticContentWrapper,
    pending_whitespace: str,
    context: TeiElementFactoryContext
) -> str:
    tail_children, tail_whitespace = get_tei_children_and_whitespace_for_semantic_content(
        semantic_content,
        context=context
    )
    if not tail_children:
        return pending_whitespace
    if pending_whitespace:
        children.append(pending_whitespace)
    children.extend(tail_children)
    return tail_whitespace
