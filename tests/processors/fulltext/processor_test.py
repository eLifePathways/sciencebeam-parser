# pylint: disable=too-many-lines
import logging
import os
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutGraphic,
    LayoutLineMeta,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta
)
from sciencebeam_parser.document.semantic_document import (
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticCaption,
    SemanticCountry,
    SemanticEditor,
    SemanticFigure,
    SemanticFigureCitation,
    SemanticGivenName,
    SemanticGraphic,
    SemanticInstitution,
    SemanticInvalidReference,
    SemanticLabel,
    SemanticMarker,
    SemanticNote,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticReference,
    SemanticReferenceCitation,
    SemanticReferenceList,
    SemanticSectionTypes,
    SemanticSurname,
    SemanticTable,
    SemanticTableCitation,
    SemanticTitle,
    iter_by_semantic_type_recursively
)
from sciencebeam_parser.processors.fulltext.processor import (
    _authors_match,
    _is_initial
)
from sciencebeam_parser.processors.fulltext import processor as processor_module
from sciencebeam_parser.processors.fulltext.processor import (
    FullTextProcessor,
    FullTextProcessorConfig,
    FullTextProcessorDocumentContext
)
from tests.processors.fulltext.model_mocks import MockFullTextModels
from tests.test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)


PAGE_META_1 = LayoutPageMeta.for_coordinates(LayoutPageCoordinates(
    x=0,
    y=0,
    width=400,
    height=600,
    page_number=1
))


COORDINATES_1 = LayoutPageCoordinates(
    x=10,
    y=100,
    width=200,
    height=100,
    page_number=1
)


LINE_META_1 = LayoutLineMeta(
    line_id=1,
    page_meta=PAGE_META_1
)


@pytest.fixture(name='fulltext_models_mock')
def _fulltext_models() -> MockFullTextModels:
    return MockFullTextModels()


@pytest.fixture(name='get_cv_document_graphic_provider_mock')
def _get_cv_document_graphic_provider_mock() -> Iterator[MagicMock]:
    with patch.object(processor_module, 'get_cv_document_graphic_provider') as mock:
        yield mock


def _get_layout_block_for_text(
    text: str,
    coordinates: LayoutPageCoordinates
):
    page_meta = PAGE_META_1._replace(page_number=coordinates.page_number)
    line_meta = LINE_META_1._replace(page_meta=page_meta)
    return LayoutBlock.for_text(text, coordinates=coordinates, line_meta=line_meta)


# pylint: disable=too-many-locals
@log_on_exception
class TestFullTextProcessor:
    def test_should_not_fail_with_empty_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        layout_document = LayoutDocument(pages=[])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None

    def test_should_extract_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        header_block = LayoutBlock.for_text('This is the header')
        body_block = LayoutBlock.for_text('This is the body')
        acknowledgment_block = LayoutBlock.for_text('Some acknowledgement')
        back_block = LayoutBlock.for_text('This is the back')

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        header_model_mock = fulltext_models_mock.header_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            header_block, '<header>'
        )
        segmentation_model_mock.update_label_by_layout_block(
            body_block, '<body>'
        )
        segmentation_model_mock.update_label_by_layout_block(
            acknowledgment_block, '<acknowledgement>'
        )
        segmentation_model_mock.update_label_by_layout_block(
            back_block, '<annex>'
        )

        header_model_mock.update_label_by_layout_block(
            header_block, '<title>'
        )

        fulltext_model_mock.update_label_by_layout_block(
            body_block, '<paragraph>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            acknowledgment_block, '<paragraph>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            back_block, '<paragraph>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            header_block,
            body_block,
            acknowledgment_block,
            back_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        assert semantic_document.front.get_text() == header_block.text
        assert semantic_document.front.get_text_by_type(SemanticTitle) == header_block.text
        assert semantic_document.body_section.get_text() == body_block.text
        assert semantic_document.back_section.view_by_section_type(
            SemanticSectionTypes.OTHER
        ).get_text() == back_block.text
        assert semantic_document.back_section.view_by_section_type(
            SemanticSectionTypes.ACKNOWLEDGEMENT
        ).get_text() == acknowledgment_block.text

    def test_should_extract_acknowledgement_only(
        self, fulltext_models_mock: MockFullTextModels
    ):
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        acknowledgment_block = LayoutBlock.for_text('Some acknowledgement')

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            acknowledgment_block, '<acknowledgement>'
        )

        fulltext_model_mock.update_label_by_layout_block(
            acknowledgment_block, '<paragraph>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            acknowledgment_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        assert semantic_document.back_section.view_by_section_type(
            SemanticSectionTypes.ACKNOWLEDGEMENT
        ).get_text() == acknowledgment_block.text

    def test_should_extract_author_names_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given_name_block = LayoutBlock.for_text('Given name')
        surname_block = LayoutBlock.for_text('Surname')
        authors_block = LayoutBlock.merge_blocks([given_name_block, surname_block])
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        header_block = authors_block

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        header_model_mock = fulltext_models_mock.header_model_mock
        name_header_model_mock = fulltext_models_mock.name_header_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            header_block, '<header>'
        )

        header_model_mock.update_label_by_layout_block(
            authors_block, '<author>'
        )

        name_header_model_mock.update_label_by_layout_block(
            given_name_block, '<forename>'
        )
        name_header_model_mock.update_label_by_layout_block(
            surname_block, '<surname>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            header_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        assert semantic_document.front.get_text() == authors_block.text
        assert (
            semantic_document.front
            .view_by_type(SemanticAuthor).get_text()
        ) == authors_block.text
        authors = semantic_document.front.authors
        assert len(authors) == 1
        assert authors[0].given_name_text == given_name_block.text
        assert authors[0].surname_text == surname_block.text

    def test_should_extract_author_names_separated_by_another_tag(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given_name_block = LayoutBlock.for_text('Given name')
        surname_block = LayoutBlock.for_text('Surname')
        other_block = LayoutBlock.for_text('Other')
        authors_block = LayoutBlock.merge_blocks([
            given_name_block, other_block, surname_block
        ])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            config=FullTextProcessorConfig(merge_raw_authors=True)
        )
        header_block = authors_block

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        header_model_mock = fulltext_models_mock.header_model_mock
        name_header_model_mock = fulltext_models_mock.name_header_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            header_block, '<header>'
        )

        header_model_mock.update_label_by_layout_block(
            given_name_block, '<author>'
        )
        header_model_mock.update_label_by_layout_block(
            surname_block, '<author>'
        )

        name_header_model_mock.update_label_by_layout_block(
            given_name_block, '<forename>'
        )
        name_header_model_mock.update_label_by_layout_block(
            surname_block, '<surname>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            header_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        authors = semantic_document.front.authors
        assert len(authors) == 1
        assert authors[0].given_name_text == given_name_block.text
        assert authors[0].surname_text == surname_block.text

    def _setup_author_blocks(
        self,
        fulltext_models_mock: MockFullTextModels,
        given_name_block: LayoutBlock,
        surname_block: LayoutBlock
    ):
        fulltext_models_mock.name_header_model_mock.update_label_by_layout_block(
            given_name_block, '<forename>'
        )
        fulltext_models_mock.name_header_model_mock.update_label_by_layout_block(
            surname_block, '<surname>'
        )

    def test_should_deduplicate_exact_author_from_second_raw_authors_span(
        self, fulltext_models_mock: MockFullTextModels
    ):
        # Main author list: "Richard Coward"
        given1 = LayoutBlock.for_text('Richard')
        sur1 = LayoutBlock.for_text('Coward')
        authors_block = LayoutBlock.merge_blocks([given1, sur1])
        # Correspondence section: same "Richard Coward" (separate tokens)
        given2 = LayoutBlock.for_text('Richard')
        sur2 = LayoutBlock.for_text('Coward')
        corresp_block = LayoutBlock.merge_blocks([given2, sur2])

        fulltext_models_mock.segmentation_model_mock.update_label_by_layout_block(
            LayoutBlock.merge_blocks([authors_block, corresp_block]), '<header>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            authors_block, '<author>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            corresp_block, '<author>'
        )
        self._setup_author_blocks(fulltext_models_mock, given1, sur1)
        self._setup_author_blocks(fulltext_models_mock, given2, sur2)

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock.merge_blocks([authors_block, corresp_block])
        ])])
        semantic_document = FullTextProcessor(
            fulltext_models_mock
        ).get_semantic_document_for_layout_document(layout_document=layout_document)
        authors = semantic_document.front.authors
        assert len(authors) == 1
        assert authors[0].given_name_text == 'Richard'
        assert authors[0].surname_text == 'Coward'
        notes = list(semantic_document.front.iter_by_type(SemanticNote))
        assert len(notes) == 1
        assert notes[0].note_type == 'raw_authors'

    def test_should_keep_second_span_when_not_a_duplicate(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given1 = LayoutBlock.for_text('Alice')
        sur1 = LayoutBlock.for_text('Smith')
        authors_block = LayoutBlock.merge_blocks([given1, sur1])
        given2 = LayoutBlock.for_text('Bob')
        sur2 = LayoutBlock.for_text('Jones')
        corresp_block = LayoutBlock.merge_blocks([given2, sur2])

        fulltext_models_mock.segmentation_model_mock.update_label_by_layout_block(
            LayoutBlock.merge_blocks([authors_block, corresp_block]), '<header>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            authors_block, '<author>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            corresp_block, '<author>'
        )
        self._setup_author_blocks(fulltext_models_mock, given1, sur1)
        self._setup_author_blocks(fulltext_models_mock, given2, sur2)

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock.merge_blocks([authors_block, corresp_block])
        ])])
        semantic_document = FullTextProcessor(
            fulltext_models_mock
        ).get_semantic_document_for_layout_document(layout_document=layout_document)
        authors = semantic_document.front.authors
        assert len(authors) == 2
        notes = list(semantic_document.front.iter_by_type(SemanticNote))
        assert len(notes) == 0

    def test_should_not_deduplicate_when_disabled(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given1 = LayoutBlock.for_text('Richard')
        sur1 = LayoutBlock.for_text('Coward')
        authors_block = LayoutBlock.merge_blocks([given1, sur1])
        given2 = LayoutBlock.for_text('Richard')
        sur2 = LayoutBlock.for_text('Coward')
        corresp_block = LayoutBlock.merge_blocks([given2, sur2])

        fulltext_models_mock.segmentation_model_mock.update_label_by_layout_block(
            LayoutBlock.merge_blocks([authors_block, corresp_block]), '<header>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            authors_block, '<author>'
        )
        fulltext_models_mock.header_model_mock.update_label_by_layout_block(
            corresp_block, '<author>'
        )
        self._setup_author_blocks(fulltext_models_mock, given1, sur1)
        self._setup_author_blocks(fulltext_models_mock, given2, sur2)

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock.merge_blocks([authors_block, corresp_block])
        ])])
        semantic_document = FullTextProcessor(
            fulltext_models_mock,
            config=FullTextProcessorConfig(deduplicate_raw_authors=False)
        ).get_semantic_document_for_layout_document(layout_document=layout_document)
        authors = semantic_document.front.authors
        assert len(authors) == 2


class TestIsInitial:
    def test_single_letter(self):
        assert _is_initial('R') is True

    def test_single_letter_with_dot(self):
        assert _is_initial('R.') is True

    def test_two_letters(self):
        assert _is_initial('JM') is True

    def test_full_name(self):
        assert _is_initial('Richard') is False

    def test_short_name_three_chars(self):
        assert _is_initial('Jan') is False


class TestAuthorsMatch:
    def _make_author(self, given_name: str, surname: str) -> SemanticAuthor:
        a = SemanticAuthor()
        if given_name:
            a.add_content(SemanticGivenName(layout_block=LayoutBlock.for_text(given_name)))
        if surname:
            a.add_content(SemanticSurname(layout_block=LayoutBlock.for_text(surname)))
        return a

    def test_exact_full_name_match(self):
        a = self._make_author('Richard', 'Coward')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=True) is True

    def test_different_surnames_no_match(self):
        a = self._make_author('Richard', 'Coward')
        b = self._make_author('Richard', 'Smith')
        assert _authors_match(a, b, use_initial_fallback=True) is False

    def test_different_full_first_names_no_match(self):
        a = self._make_author('James', 'Smith')
        b = self._make_author('John', 'Smith')
        assert _authors_match(a, b, use_initial_fallback=True) is False

    def test_initial_vs_full_name_with_fallback(self):
        a = self._make_author('R', 'Coward')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=True) is True

    def test_initial_vs_full_name_without_fallback(self):
        a = self._make_author('R', 'Coward')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=False) is False

    def test_case_insensitive(self):
        a = self._make_author('richard', 'coward')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=True) is True

    def test_missing_given_name_no_match(self):
        a = self._make_author('', 'Coward')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=True) is False

    def test_missing_surname_no_match(self):
        a = self._make_author('Richard', '')
        b = self._make_author('Richard', 'Coward')
        assert _authors_match(a, b, use_initial_fallback=True) is False


class TestFullTextProcessorAffiliation:
    def test_should_extract_affiliation_address_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        marker_block = LayoutBlock.for_text('1')
        institution_block = LayoutBlock.for_text('Institution1')
        country_block = LayoutBlock.for_text('Country1')
        aff_block = LayoutBlock.merge_blocks([marker_block, institution_block])
        address_block = LayoutBlock.merge_blocks([country_block])
        aff_address_block = LayoutBlock.merge_blocks([aff_block, address_block])
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        header_block = aff_address_block

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        header_model_mock = fulltext_models_mock.header_model_mock
        affiliation_address_model_mock = fulltext_models_mock.affiliation_address_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            header_block, '<header>'
        )

        header_model_mock.update_label_by_layout_block(
            aff_block, '<affiliation>'
        )
        header_model_mock.update_label_by_layout_block(
            address_block, '<address>'
        )

        affiliation_address_model_mock.update_label_by_layout_block(
            marker_block, '<marker>'
        )
        affiliation_address_model_mock.update_label_by_layout_block(
            institution_block, '<institution>'
        )
        affiliation_address_model_mock.update_label_by_layout_block(
            country_block, '<country>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            header_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        assert semantic_document.front.get_text() == aff_address_block.text
        assert (
            semantic_document.front
            .view_by_type(SemanticAffiliationAddress).get_text()
        ) == aff_address_block.text
        affiliations = list(semantic_document.front.iter_by_type(SemanticAffiliationAddress))
        assert len(affiliations) == 1
        assert affiliations[0].get_text_by_type(SemanticMarker) == marker_block.text
        assert affiliations[0].get_text_by_type(SemanticInstitution) == institution_block.text
        assert affiliations[0].get_text_by_type(SemanticCountry) == country_block.text
        assert affiliations[0].content_id == 'aff0'

    def test_should_not_merge_separate_raw_affiliations(
        self, fulltext_models_mock: MockFullTextModels
    ):
        aff_suffix_texts = ['1', '2']
        institution_blocks = [
            LayoutBlock.for_text(f'Institution{t}')
            for t in aff_suffix_texts
        ]
        aff_blocks = institution_blocks
        aff_address_blocks = aff_blocks
        fulltext_processor = FullTextProcessor(fulltext_models_mock)
        header_block = LayoutBlock.merge_blocks(aff_address_blocks)

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        header_model_mock = fulltext_models_mock.header_model_mock
        affiliation_address_model_mock = fulltext_models_mock.affiliation_address_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            header_block, '<header>'
        )

        for aff_block in aff_blocks:
            header_model_mock.update_label_by_layout_block(
                aff_block, '<affiliation>'
            )

        for institution_block in institution_blocks:
            affiliation_address_model_mock.update_label_by_layout_block(
                institution_block, '<institution>'
            )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            header_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        assert semantic_document is not None
        affiliations = list(semantic_document.front.iter_by_type(SemanticAffiliationAddress))
        LOGGER.debug('affiliations: %r', affiliations)
        assert (
            [aff.get_text() for aff in affiliations]
            == [aff_block.text for aff_block in aff_blocks]
        )
        assert (
            [aff.content_id for aff in affiliations]
            == ['aff0', 'aff1']
        )

    def test_should_extract_raw_references_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        label_block = LayoutBlock.for_text('1')
        ref_text_block = LayoutBlock.for_text('Reference 1')
        ref_block = LayoutBlock.merge_blocks([label_block, ref_text_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(extract_citation_fields=False)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            ref_block, '<references>'
        )

        reference_segmenter_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        reference_segmenter_model_mock.update_label_by_layout_block(
            ref_text_block, '<reference>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            ref_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        assert semantic_document.back_section.get_text() == ref_block.text
        reference_list = list(semantic_document.back_section.iter_by_type(SemanticReferenceList))
        assert len(reference_list) == 1
        references = list(reference_list[0].iter_by_type(SemanticRawReference))
        assert len(references) == 1
        ref = references[0]
        assert ref.get_text_by_type(SemanticLabel) == label_block.text
        assert ref.get_text_by_type(SemanticRawReferenceText) == ref_text_block.text
        assert ref.content_id == 'b0'

    def test_should_extract_references_fields_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        other_body = LayoutBlock.for_text('the body')
        citation_block = LayoutBlock.for_text('1')
        body_block = LayoutBlock.merge_blocks([other_body, citation_block])
        label_block = LayoutBlock.for_text('1')
        ref_title_block = LayoutBlock.for_text('Reference Title 1')
        ref_text_block = LayoutBlock.merge_blocks([
            ref_title_block
        ])
        ref_block = LayoutBlock.merge_blocks([label_block, ref_text_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(extract_citation_fields=True)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock
        citation_model_mock = fulltext_models_mock.citation_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            body_block, '<body>'
        )
        segmentation_model_mock.update_label_by_layout_block(
            ref_block, '<references>'
        )

        fulltext_model_mock.update_label_by_layout_block(
            other_body, '<section>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            citation_block, '<citation_marker>'
        )

        reference_segmenter_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        reference_segmenter_model_mock.update_label_by_layout_block(
            ref_text_block, '<reference>'
        )

        citation_model_mock.update_label_by_layout_block(
            ref_title_block, '<title>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            body_block,
            ref_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        reference_list = list(semantic_document.back_section.iter_by_type(SemanticReferenceList))
        assert len(reference_list) == 1
        references = list(reference_list[0].iter_by_type(SemanticReference))
        assert len(references) == 1
        ref = references[0]
        assert ref.get_text_by_type(SemanticTitle) == ref_title_block.text
        assert ref.get_text_by_type(SemanticLabel) == label_block.text
        assert ref.get_text_by_type(SemanticRawReferenceText) == ref_text_block.text
        assert ref.content_id == 'b0'
        ref_citations = list(
            semantic_document.iter_by_type_recursively(SemanticReferenceCitation)
        )
        assert len(ref_citations) == 1
        assert ref_citations[0].target_content_id == 'b0'

    def test_should_extract_invalid_reference_from_document(
        self, fulltext_models_mock: MockFullTextModels
    ):
        other_body = LayoutBlock.for_text('the body')
        citation_block = LayoutBlock.for_text('1')
        body_block = LayoutBlock.merge_blocks([other_body, citation_block])
        invalid_reference_block = LayoutBlock.for_text('This is an invalid reference 1')
        ref_text_block = invalid_reference_block
        ref_block = ref_text_block
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(extract_citation_fields=True)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock
        citation_model_mock = fulltext_models_mock.citation_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            body_block, '<body>'
        )
        segmentation_model_mock.update_label_by_layout_block(
            ref_block, '<references>'
        )

        fulltext_model_mock.update_label_by_layout_block(
            other_body, '<section>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            citation_block, '<citation_marker>'
        )

        reference_segmenter_model_mock.update_label_by_layout_block(
            ref_text_block, '<reference>'
        )

        citation_model_mock.update_label_by_layout_block(
            invalid_reference_block, 'O'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            body_block,
            ref_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        reference_list = list(semantic_document.back_section.iter_by_type(SemanticReferenceList))
        assert len(reference_list) == 1
        references = list(reference_list[0].iter_by_type(SemanticInvalidReference))
        assert len(references) == 1
        assert references[0].get_text() == invalid_reference_block.text

    def test_should_extract_author_names_from_references_fields(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given_name_block = LayoutBlock.for_text('Given name')
        surname_block = LayoutBlock.for_text('Surname')
        other_block = LayoutBlock.for_text('Other')
        authors_block = LayoutBlock.merge_blocks([
            given_name_block, other_block, surname_block
        ])
        ref_text_block = LayoutBlock.merge_blocks([
            authors_block
        ])
        ref_block = LayoutBlock.merge_blocks([ref_text_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(extract_citation_fields=True)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock
        citation_model_mock = fulltext_models_mock.citation_model_mock
        name_citation_model_mock = fulltext_models_mock.name_citation_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            ref_block, '<references>'
        )

        reference_segmenter_model_mock.update_label_by_layout_block(
            ref_text_block, '<reference>'
        )

        citation_model_mock.update_label_by_layout_block(
            authors_block, '<author>'
        )

        name_citation_model_mock.update_label_by_layout_block(
            given_name_block, '<forename>'
        )
        name_citation_model_mock.update_label_by_layout_block(
            surname_block, '<surname>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            ref_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        reference_list = list(semantic_document.back_section.iter_by_type(SemanticReferenceList))
        assert len(reference_list) == 1
        references = list(reference_list[0].iter_by_type(SemanticReference))
        assert len(references) == 1
        ref = references[0]
        authors = list(ref.iter_by_type(SemanticAuthor))
        assert len(authors) == 1
        assert authors[0].given_name_text == given_name_block.text
        assert authors[0].surname_text == surname_block.text

    def test_should_extract_editor_names_from_references_fields(
        self, fulltext_models_mock: MockFullTextModels
    ):
        given_name_block = LayoutBlock.for_text('Given name')
        surname_block = LayoutBlock.for_text('Surname')
        other_block = LayoutBlock.for_text('Other')
        editors_block = LayoutBlock.merge_blocks([
            given_name_block, other_block, surname_block
        ])
        ref_text_block = LayoutBlock.merge_blocks([
            editors_block
        ])
        ref_block = LayoutBlock.merge_blocks([ref_text_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(
                extract_citation_fields=True,
                extract_citation_authors=False,
                extract_citation_editors=True
            )
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock
        citation_model_mock = fulltext_models_mock.citation_model_mock
        name_citation_model_mock = fulltext_models_mock.name_citation_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            ref_block, '<references>'
        )

        reference_segmenter_model_mock.update_label_by_layout_block(
            ref_text_block, '<reference>'
        )

        citation_model_mock.update_label_by_layout_block(
            editors_block, '<editor>'
        )

        name_citation_model_mock.update_label_by_layout_block(
            given_name_block, '<forename>'
        )
        name_citation_model_mock.update_label_by_layout_block(
            surname_block, '<surname>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            ref_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        reference_list = list(semantic_document.back_section.iter_by_type(SemanticReferenceList))
        assert len(reference_list) == 1
        references = list(reference_list[0].iter_by_type(SemanticReference))
        assert len(references) == 1
        ref = references[0]
        editors = list(ref.iter_by_type(SemanticEditor))
        assert len(editors) == 1
        assert editors[0].given_name_text == given_name_block.text
        assert editors[0].surname_text == surname_block.text

    @pytest.mark.parametrize(
        'segmentation_label',
        ['<body>', '<annex>']
    )
    def test_should_extract_figure_label_caption_and_graphic(
        self, fulltext_models_mock: MockFullTextModels,
        segmentation_label: str
    ):
        citation_block = LayoutBlock.for_text('Figure 1')
        _coordinates = COORDINATES_1
        graphic_local_file_path = '/path/to/graphic1.svg'
        graphic = LayoutGraphic(
            coordinates=_coordinates,
            local_file_path=graphic_local_file_path,
            page_meta=PAGE_META_1
        )
        _coordinates = _coordinates.move_by(dy=10)
        label_block = _get_layout_block_for_text('Figure 1', coordinates=_coordinates)
        _coordinates = _coordinates.move_by(dy=10)
        caption_block = _get_layout_block_for_text('Caption 1', coordinates=_coordinates)
        other_block = LayoutBlock.for_text('Other')
        figure_block = LayoutBlock.merge_blocks([
            label_block, other_block, caption_block
        ])
        fulltext_block = LayoutBlock.merge_blocks([citation_block, figure_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(
                extract_figure_fields=True,
                extract_graphic_bounding_boxes=True,
                extract_graphic_assets=True
            )
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        figure_model_mock = fulltext_models_mock.figure_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            fulltext_block, segmentation_label
        )

        fulltext_model_mock.update_label_by_layout_block(
            citation_block, '<figure_marker>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            figure_block, '<figure>'
        )

        figure_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        figure_model_mock.update_label_by_layout_block(
            caption_block, '<figDesc>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(
            blocks=[fulltext_block],
            graphics=[graphic]
        )])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        figure_list = list(iter_by_semantic_type_recursively(
            [semantic_document.body_section, semantic_document.back_section],
            SemanticFigure
        ))
        assert len(figure_list) == 1
        figure = figure_list[0]
        assert figure.get_text_by_type(SemanticLabel) == label_block.text
        assert figure.get_text_by_type(SemanticCaption) == caption_block.text
        assert figure.content_id == 'fig_0'
        figure_citation_list = list(
            semantic_document.iter_by_type_recursively(SemanticFigureCitation)
        )
        assert len(figure_citation_list) == 1
        assert figure_citation_list[0].get_text() == citation_block.text
        assert figure_citation_list[0].target_content_id == 'fig_0'
        semantic_graphic_list = list(figure.iter_by_type(SemanticGraphic))
        assert semantic_graphic_list
        assert semantic_graphic_list[0].layout_graphic == graphic
        assert semantic_graphic_list[0].relative_path == os.path.basename(
            graphic_local_file_path
        )

    @pytest.mark.parametrize(
        'segmentation_label',
        ['<body>', '<annex>']
    )
    def test_should_detect_and_extract_figure_graphic(
        self, fulltext_models_mock: MockFullTextModels,
        tmp_path: Path,
        get_cv_document_graphic_provider_mock: MagicMock,
        segmentation_label: str
    ):
        _coordinates = COORDINATES_1
        graphic_local_file_path = '/path/to/graphic1.svg'
        graphic = LayoutGraphic(
            coordinates=_coordinates,
            local_file_path=graphic_local_file_path,
            page_meta=PAGE_META_1
        )
        iter_semantic_graphic_for_layout_document_mock = (
            get_cv_document_graphic_provider_mock
            .return_value
            .iter_semantic_graphic_for_layout_document
        )
        iter_semantic_graphic_for_layout_document_mock.return_value = [SemanticGraphic(
            layout_graphic=graphic,
            relative_path=os.path.basename(graphic_local_file_path)
        )]
        _coordinates = _coordinates.move_by(dy=10)
        label_block = _get_layout_block_for_text('Figure 1', coordinates=_coordinates)
        _coordinates = _coordinates.move_by(dy=10)
        caption_block = _get_layout_block_for_text('Caption 1', coordinates=_coordinates)
        other_block = LayoutBlock.for_text('Other')
        figure_block = LayoutBlock.merge_blocks([
            label_block, other_block, caption_block
        ])
        fulltext_block = LayoutBlock.merge_blocks([figure_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(
                use_cv_model=True,
                extract_figure_fields=True,
                extract_graphic_bounding_boxes=True,
                extract_graphic_assets=True
            )
        )
        processor_document_context = FullTextProcessorDocumentContext(
            pdf_path=str(tmp_path / 'test.pdf'),
            temp_dir=str(tmp_path)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        figure_model_mock = fulltext_models_mock.figure_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            fulltext_block, segmentation_label
        )

        fulltext_model_mock.update_label_by_layout_block(
            figure_block, '<figure>'
        )

        figure_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        figure_model_mock.update_label_by_layout_block(
            caption_block, '<figDesc>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(
            blocks=[fulltext_block]
        )])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document,
            context=processor_document_context
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        figure_list = list(iter_by_semantic_type_recursively(
            [semantic_document.body_section, semantic_document.back_section],
            SemanticFigure
        ))
        assert len(figure_list) == 1
        figure = figure_list[0]
        assert figure.get_text_by_type(SemanticLabel) == label_block.text
        assert figure.get_text_by_type(SemanticCaption) == caption_block.text
        semantic_graphic_list = list(figure.iter_by_type(SemanticGraphic))
        assert semantic_graphic_list
        assert semantic_graphic_list[0].layout_graphic == graphic
        assert semantic_graphic_list[0].relative_path == os.path.basename(
            graphic_local_file_path
        )
        assert get_cv_document_graphic_provider_mock.call_args[1]['page_numbers'] == [
            _coordinates.page_number
        ]

    @pytest.mark.parametrize(
        'segmentation_label',
        ['<body>', '<annex>']
    )
    def test_should_extract_table_label_caption(
        self, fulltext_models_mock: MockFullTextModels,
        segmentation_label: str
    ):
        citation_block = LayoutBlock.for_text('Table 1')
        label_block = LayoutBlock.for_text('Table 1')
        caption_block = LayoutBlock.for_text('Caption 1')
        other_block = LayoutBlock.for_text('Other')
        figure_block = LayoutBlock.merge_blocks([
            label_block, other_block, caption_block
        ])
        fulltext_block = LayoutBlock.merge_blocks([citation_block, figure_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(extract_table_fields=True)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        table_model_mock = fulltext_models_mock.table_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            fulltext_block, segmentation_label
        )

        fulltext_model_mock.update_label_by_layout_block(
            citation_block, '<table_marker>'
        )
        fulltext_model_mock.update_label_by_layout_block(
            figure_block, '<table>'
        )

        table_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        table_model_mock.update_label_by_layout_block(
            caption_block, '<figDesc>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            fulltext_block
        ])])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        table_list = list(iter_by_semantic_type_recursively(
            [semantic_document.body_section, semantic_document.back_section],
            SemanticTable
        ))
        assert len(table_list) == 1
        table = table_list[0]
        assert table.get_text_by_type(SemanticLabel) == label_block.text
        assert table.get_text_by_type(SemanticCaption) == caption_block.text
        assert table.content_id == 'tab_0'
        table_citation_list = list(
            semantic_document.iter_by_type_recursively(SemanticTableCitation)
        )
        assert len(table_citation_list) == 1
        assert table_citation_list[0].get_text() == citation_block.text
        assert table_citation_list[0].target_content_id == 'tab_0'

    @pytest.mark.parametrize(
        'segmentation_label',
        ['<body>', '<annex>']
    )
    def test_should_detect_and_extract_table_graphic(
        self, fulltext_models_mock: MockFullTextModels,
        tmp_path: Path,
        get_cv_document_graphic_provider_mock: MagicMock,
        segmentation_label: str
    ):
        _coordinates = COORDINATES_1
        graphic_local_file_path = '/path/to/graphic1.svg'
        graphic = LayoutGraphic(
            coordinates=_coordinates,
            local_file_path=graphic_local_file_path,
            page_meta=PAGE_META_1
        )
        iter_semantic_graphic_for_layout_document_mock = (
            get_cv_document_graphic_provider_mock
            .return_value
            .iter_semantic_graphic_for_layout_document
        )
        iter_semantic_graphic_for_layout_document_mock.return_value = [SemanticGraphic(
            layout_graphic=graphic,
            relative_path=os.path.basename(graphic_local_file_path)
        )]
        _coordinates = _coordinates.move_by(dy=10)
        label_block = _get_layout_block_for_text('Table 1', coordinates=_coordinates)
        _coordinates = _coordinates.move_by(dy=10)
        caption_block = _get_layout_block_for_text('Caption 1', coordinates=_coordinates)
        other_block = LayoutBlock.for_text('Other')
        table_block = LayoutBlock.merge_blocks([
            label_block, other_block, caption_block
        ])

        fulltext_block = LayoutBlock.merge_blocks([table_block])
        fulltext_processor = FullTextProcessor(
            fulltext_models_mock,
            FullTextProcessorConfig(
                use_cv_model=True,
                extract_figure_fields=True,
                extract_graphic_bounding_boxes=True,
                extract_graphic_assets=True
            )
        )
        processor_document_context = FullTextProcessorDocumentContext(
            pdf_path=str(tmp_path / 'test.pdf'),
            temp_dir=str(tmp_path)
        )

        segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
        fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
        table_model_mock = fulltext_models_mock.table_model_mock

        segmentation_model_mock.update_label_by_layout_block(
            fulltext_block, segmentation_label
        )

        fulltext_model_mock.update_label_by_layout_block(
            table_block, '<table>'
        )

        table_model_mock.update_label_by_layout_block(
            label_block, '<label>'
        )
        table_model_mock.update_label_by_layout_block(
            caption_block, '<figDesc>'
        )

        layout_document = LayoutDocument(pages=[LayoutPage(
            blocks=[fulltext_block]
        )])
        semantic_document = fulltext_processor.get_semantic_document_for_layout_document(
            layout_document=layout_document,
            context=processor_document_context
        )
        LOGGER.debug('semantic_document: %s', semantic_document)
        assert semantic_document is not None
        table_list = list(iter_by_semantic_type_recursively(
            [semantic_document.body_section, semantic_document.back_section],
            SemanticTable
        ))
        assert len(table_list) == 1
        table = table_list[0]
        assert table.get_text_by_type(SemanticLabel) == label_block.text
        assert table.get_text_by_type(SemanticCaption) == caption_block.text
        semantic_graphic_list = list(table.iter_by_type(SemanticGraphic))
        assert semantic_graphic_list
        assert semantic_graphic_list[0].layout_graphic == graphic
        assert semantic_graphic_list[0].relative_path == os.path.basename(
            graphic_local_file_path
        )
        assert get_cv_document_graphic_provider_mock.call_args[1]['page_numbers'] == [
            _coordinates.page_number
        ]
