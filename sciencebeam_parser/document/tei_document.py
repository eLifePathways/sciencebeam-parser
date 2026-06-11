import logging
from typing import (
    Dict,
    List,
    Optional,
)

from sciencebeam_parser.utils.stop_watch import StopWatchRecorder
from sciencebeam_parser.document.semantic_document import (
    SemanticAbstract,
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticDocument,
    SemanticFigure,
    SemanticMarker,
    SemanticNote,
    SemanticReferenceList,
    SemanticSection,
    SemanticSectionTypes,
    SemanticTable,
    SemanticTitle
)
from sciencebeam_parser.document.tei.common import (
    TeiElementBuilder
)
from sciencebeam_parser.document.tei.document import (
    TeiDocument
)
from sciencebeam_parser.document.tei.factories import (
    DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT,
    TeiElementFactoryContext,
    get_tei_note_for_semantic_content
)
from sciencebeam_parser.document.tei.author import (
    get_dummy_tei_author_for_semantic_affiliations_element,
    get_orphan_affiliations,
    get_tei_author_for_semantic_author_element
)
from sciencebeam_parser.document.tei.references import (
    get_tei_raw_reference_list_element
)


LOGGER = logging.getLogger(__name__)


def get_tei_for_semantic_document(  # pylint: disable=too-many-branches, too-many-statements
    semantic_document: SemanticDocument,
    context: Optional[TeiElementFactoryContext] = None
) -> TeiDocument:
    if context is None:
        context = DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
    stop_watch_recorder = StopWatchRecorder()
    LOGGER.info('generating tei document')
    LOGGER.debug('semantic_document: %s', semantic_document)
    tei_document = TeiDocument()

    stop_watch_recorder.start('front')
    LOGGER.info('generating tei document: front')
    semantic_title = next(semantic_document.front.iter_by_type(SemanticTitle), None)
    if semantic_title is not None and semantic_title.merged_block:
        tei_document.set_title_layout_block(
            semantic_title.merged_block,
            trailing_text=semantic_title.trailing_text
        )

    abstract_block = semantic_document.front.view_by_type(SemanticAbstract).merged_block
    if abstract_block:
        tei_document.set_abstract_layout_block(abstract_block)

    affiliations_by_marker: Dict[str, List[SemanticAffiliationAddress]] = {}
    for semantic_content in semantic_document.front:
        if isinstance(semantic_content, SemanticAffiliationAddress):
            marker_text = semantic_content.get_text_by_type(SemanticMarker)
            affiliations_by_marker.setdefault(marker_text, []).append(semantic_content)
    LOGGER.debug('affiliations_by_marker: %r', affiliations_by_marker)

    semantic_authors: List[SemanticAuthor] = []
    for semantic_content in semantic_document.front:
        if isinstance(semantic_content, SemanticAuthor):
            semantic_authors.append(semantic_content)
            tei_author_element = get_tei_author_for_semantic_author_element(
                semantic_content,
                context=context,
                affiliations_by_marker=affiliations_by_marker
            )
            tei_document.get_or_create_element_at(
                ['teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic']
            ).append(tei_author_element)
            continue
        if isinstance(semantic_content, SemanticTitle):
            continue
        if isinstance(semantic_content, SemanticAbstract):
            continue
        if isinstance(semantic_content, SemanticAffiliationAddress):
            continue
        tei_document.get_or_create_element_at(
            ['teiHeader']
        ).append(get_tei_note_for_semantic_content(
            semantic_content
        ))
    orphan_affiliations = get_orphan_affiliations(
        affiliations_by_marker=affiliations_by_marker,
        authors=semantic_authors
    )
    if orphan_affiliations:
        dummy_tei_author_element = get_dummy_tei_author_for_semantic_affiliations_element(
            orphan_affiliations,
            context=context
        )
        tei_document.get_or_create_element_at(
            ['teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic']
        ).append(dummy_tei_author_element)

    LOGGER.info('generating tei document: body')
    stop_watch_recorder.start('body')
    for semantic_content in semantic_document.body_section:
        if isinstance(semantic_content, SemanticNote):
            tei_document.get_body().add_note(
                semantic_content.note_type,
                semantic_content.merged_block
            )
            continue
        TeiElementBuilder(tei_document.get_body_element()).extend(
            context.get_tei_child_elements_for_semantic_content(semantic_content)
        )
    for semantic_content in semantic_document.body_section.iter_by_types_recursively(
        (SemanticFigure, SemanticTable,)
    ):
        TeiElementBuilder(tei_document.get_body_element()).extend(
            context.get_tei_child_elements_for_semantic_content(semantic_content)
        )

    LOGGER.info('generating tei document: back section')
    stop_watch_recorder.start('back')
    for semantic_content in semantic_document.back_section:
        if isinstance(semantic_content, SemanticSection):
            if semantic_content.section_type == SemanticSectionTypes.ACKNOWLEDGEMENT:
                _parent = TeiElementBuilder(tei_document.get_back_element())
            else:
                _parent = TeiElementBuilder(tei_document.get_back_annex_element())
            TeiElementBuilder(_parent).extend(
                context.get_tei_child_elements_for_semantic_content(semantic_content)
            )
            continue
        if isinstance(semantic_content, SemanticNote):
            tei_document.get_back_annex().add_note(
                semantic_content.note_type,
                semantic_content.merged_block
            )
            continue
        if isinstance(semantic_content, SemanticReferenceList):
            tei_document.get_references().element.append(
                get_tei_raw_reference_list_element(semantic_content, context=context)
            )
            continue
        TeiElementBuilder(tei_document.get_back_annex_element()).extend(
            context.get_tei_child_elements_for_semantic_content(semantic_content)
        )
    for semantic_figure in semantic_document.back_section.iter_by_types_recursively(
        (SemanticFigure, SemanticTable,)
    ):
        TeiElementBuilder(tei_document.get_back_annex_element()).extend(
            context.get_tei_child_elements_for_semantic_content(semantic_figure)
        )
    stop_watch_recorder.stop()
    LOGGER.info('generating tei document done, took: %s', stop_watch_recorder)
    return tei_document
