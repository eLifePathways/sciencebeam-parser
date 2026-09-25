import logging

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticDate,
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticExternalUrl,
    SemanticInvalidReference,
    SemanticIssue,
    SemanticJournal,
    SemanticLabel,
    SemanticLocation,
    SemanticPageRange,
    SemanticPublisher,
    SemanticRawAuthors,
    SemanticRawEditors,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticReference,
    SemanticTitle,
    SemanticVolume
)
from sciencebeam_parser.models.citation.extract import (
    CitationSemanticExtractor,
    parse_page_range,
    parse_web,
    parse_pubnum,
    parse_date
)


LOGGER = logging.getLogger(__name__)


PII_1 = 'S0123-1234(11)01234-5'


class TestParsePageRange:
    def test_should_parse_page_range_from_to(self):
        page_range = parse_page_range(LayoutBlock.for_text('12-15'))
        assert page_range.from_page == '12'
        assert page_range.to_page == '15'

    def test_should_ignore_spaces(self):
        page_range = parse_page_range(LayoutBlock.for_text(' 12 - 15 '))
        assert page_range.from_page == '12'
        assert page_range.to_page == '15'

    def test_should_complete_end_page(self):
        page_range = parse_page_range(LayoutBlock.for_text('1234-56'))
        assert page_range.from_page == '1234'
        assert page_range.to_page == '1256'

    def test_should_parse_single_page_number(self):
        page_range = parse_page_range(LayoutBlock.for_text('12'))
        assert page_range.from_page is None
        assert page_range.to_page is None


class TestParseWeb:
    def test_should_parse_url(self):
        external_url = parse_web(LayoutBlock.for_text('http://host/path'))
        assert isinstance(external_url, SemanticExternalUrl)
        assert external_url.value == 'http://host/path'

    def test_should_remove_spaces(self):
        external_url = parse_web(LayoutBlock.for_text('http : // host / path'))
        assert external_url.value == 'http://host/path'

    def test_should_detect_doi_url(self):
        external_identifier = parse_web(LayoutBlock.for_text('http://doi.org/10.1234/test'))
        assert isinstance(external_identifier, SemanticExternalIdentifier)
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.DOI
        assert external_identifier.value == '10.1234/test'


class TestParsePubNum:
    def test_should_use_none_type_for_unknown_pattern(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('xyz'))
        assert external_identifier.external_identifier_type is None
        assert external_identifier.value == 'xyz'

    def test_should_detect_doi_with_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('doi: 10.1234/test'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.DOI
        assert external_identifier.value == '10.1234/test'

    def test_should_detect_doi_without_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('10.1234/test'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.DOI
        assert external_identifier.value == '10.1234/test'

    def test_should_detect_pmid_with_pmid_colon_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PMID: 1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMID
        assert external_identifier.value == '1234567'

    def test_should_detect_pmid_with_pmid_label_without_colon_and_space(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PMID1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMID
        assert external_identifier.value == '1234567'

    def test_should_detect_pmid_with_pubmed_colon_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PubMed: 1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMID
        assert external_identifier.value == '1234567'

    def test_should_detect_pmid_with_pubmedid_colon_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PubMedID: 1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMID
        assert external_identifier.value == '1234567'

    def test_should_detect_pmcids_with_pmc_prefix(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PMC1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMCID
        assert external_identifier.value == 'PMC1234567'

    def test_should_detect_pmcids_with_double_pmc_prefix(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PMCPMC1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMCID
        assert external_identifier.value == 'PMC1234567'

    def test_should_detect_pmcids_with_pmc_prefix_and_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('PubMed Central: PMC1234567'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PMCID
        assert external_identifier.value == 'PMC1234567'

    def test_should_parse_arxiv_with_arxiv_colon_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('arXiv: 0706.0001'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.ARXIV
        assert external_identifier.value == '0706.0001'

    def test_should_parse_arxiv_with_old_arxiv_id(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('arXiv: math.GT/0309136'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.ARXIV
        assert external_identifier.value == 'math.GT/0309136'

    def test_should_parse_pii_without_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text(PII_1))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PII
        assert external_identifier.value == PII_1

    def test_should_parse_pii_with_pii_colon_label(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text('pii: ' + PII_1))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PII
        assert external_identifier.value == PII_1

    def test_should_parse_pii_with_pii_square_brackets_suffix(self):
        external_identifier = parse_pubnum(LayoutBlock.for_text(PII_1 + ' [pii]'))
        assert external_identifier.external_identifier_type == SemanticExternalIdentifierTypes.PII
        assert external_identifier.value == PII_1


class TestParseDate:
    def test_should_parse_invalid_value(self):
        semantic_date = parse_date(LayoutBlock.for_text('xyz'))
        assert semantic_date.get_text() == 'xyz'
        assert semantic_date.year is None

    def test_should_parse_four_digit_year(self):
        semantic_date = parse_date(LayoutBlock.for_text('1991'))
        assert semantic_date.get_text() == '1991'
        assert semantic_date.year == 1991

    def test_should_parse_four_digit_year_in_iso_date(self):
        semantic_date = parse_date(LayoutBlock.for_text('1991-01-01'))
        assert semantic_date.get_text() == '1991-01-01'
        assert semantic_date.year == 1991


class TestReferenceSegmenterSemanticExtractor:
    def test_should_extract_single_raw_reference(self):
        semantic_content_list = list(
            CitationSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<author>', LayoutBlock.for_text('Author 1')),
                ('<editor>', LayoutBlock.for_text('Editor 1')),
                ('<title>', LayoutBlock.for_text('Title 1')),
                ('<journal>', LayoutBlock.for_text('Journal 1')),
                ('<volume>', LayoutBlock.for_text('Volume 1')),
                ('<issue>', LayoutBlock.for_text('Issue 1')),
                ('<pages>', LayoutBlock.for_text('12-15')),
                ('<publisher>', LayoutBlock.for_text('Publisher 1')),
                ('<location>', LayoutBlock.for_text('Location 1')),
                ('<web>', LayoutBlock.for_text('http : // host / path')),
                ('<pubnum>', LayoutBlock.for_text('doi: 10.1234/test')),
                ('<date>', LayoutBlock.for_text('1991'))
            ])
        )
        assert len(semantic_content_list) == 1
        ref = semantic_content_list[0]
        assert isinstance(ref, SemanticReference)
        assert ref.view_by_type(SemanticRawAuthors).get_text() == 'Author 1'
        assert ref.view_by_type(SemanticRawEditors).get_text() == 'Editor 1'
        assert ref.view_by_type(SemanticTitle).get_text() == 'Title 1'
        assert ref.view_by_type(SemanticJournal).get_text() == 'Journal 1'
        assert ref.view_by_type(SemanticVolume).get_text() == 'Volume 1'
        assert ref.view_by_type(SemanticIssue).get_text() == 'Issue 1'
        assert ref.view_by_type(SemanticPageRange).get_text() == '12-15'
        page_range = list(ref.iter_by_type(SemanticPageRange))[0]
        assert page_range.from_page == '12'
        assert page_range.to_page == '15'
        assert ref.view_by_type(SemanticPublisher).get_text() == 'Publisher 1'
        assert ref.view_by_type(SemanticLocation).get_text() == 'Location 1'
        assert ref.view_by_type(SemanticExternalUrl).get_text() == 'http : // host / path'
        assert ref.view_by_type(SemanticExternalIdentifier).get_text() == 'doi: 10.1234/test'
        external_url = list(ref.iter_by_type(SemanticExternalUrl))[0]
        assert external_url.value == 'http://host/path'
        external_identifier = list(ref.iter_by_type(SemanticExternalIdentifier))[0]
        assert external_identifier.value == '10.1234/test'
        assert ref.view_by_type(SemanticDate).get_text() == '1991'
        semantic_date = list(ref.iter_by_type(SemanticDate))[0]
        assert semantic_date.year == 1991
        assert ref.content_id == 'b0'

    def test_should_add_raw_reference_semantic_content(self):
        semantic_marker = SemanticLabel(layout_block=LayoutBlock.for_text('1'))
        semantic_raw_ref_text = SemanticRawReferenceText(
            layout_block=LayoutBlock.for_text('Reference 1')
        )
        semantic_raw_ref = SemanticRawReference([
            semantic_marker, semantic_raw_ref_text
        ], content_id='raw1')
        semantic_content_list = list(
            CitationSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<author>', LayoutBlock.for_text('Author 1')),
                ('<title>', LayoutBlock.for_text('Title 1'))
            ], semantic_raw_reference=semantic_raw_ref)
        )
        assert len(semantic_content_list) == 1
        ref = semantic_content_list[0]
        assert isinstance(ref, SemanticReference)
        assert ref.view_by_type(SemanticLabel).get_text() == semantic_marker.get_text()
        assert (
            ref.view_by_type(SemanticRawReferenceText).get_text()
            == semantic_raw_ref_text.get_text()
        )
        assert ref.view_by_type(SemanticRawAuthors).get_text() == 'Author 1'
        assert ref.view_by_type(SemanticTitle).get_text() == 'Title 1'
        assert ref.content_id == semantic_raw_ref.content_id

    def test_should_reject_reference_without_any_detected_fields(self):
        semantic_raw_ref_text = SemanticRawReferenceText(
            layout_block=LayoutBlock.for_text('Reference 1')
        )
        semantic_raw_ref = SemanticRawReference([
            semantic_raw_ref_text
        ], content_id='raw1')
        semantic_content_list = list(
            CitationSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('O', LayoutBlock.for_text(semantic_raw_ref_text.get_text())),
            ], semantic_raw_reference=semantic_raw_ref)
        )
        assert len(semantic_content_list) == 1
        ref = semantic_content_list[0]
        assert isinstance(ref, SemanticInvalidReference)
        assert ref.get_text() == semantic_raw_ref_text.get_text()
