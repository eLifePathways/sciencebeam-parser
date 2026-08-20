from lxml import etree

from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL
from sciencebeam_parser.models.data import LabeledLayoutModelData, LayoutModelData
from sciencebeam_parser.training.jats.field_vocab import JatsSubFieldNames
from sciencebeam_parser.training.quality.counting import (
    count_citation_labels,
    count_entity_elements,
    get_labels_for_model_data_list
)


TEI_NS = 'http://www.tei-c.org/ns/1.0'


def _labeled(label) -> LabeledLayoutModelData:
    return LabeledLayoutModelData(data_line='token', label=label)


class TestCountEntityElements:
    def test_should_count_reference_segmenter_elements(self):
        tei_root = etree.fromstring(
            '<tei><text><listBibl>'
            '<bibl>Reference 1</bibl><bibl>Reference 2</bibl>'
            '</listBibl></text></tei>'
        )
        assert count_entity_elements('reference-segmenter', tei_root) == 2

    def test_should_count_namespaced_citation_elements(self):
        tei_root = etree.fromstring(
            f'<TEI xmlns="{TEI_NS}"><text><back><listBibl>'
            '<bibl>Reference 1</bibl><bibl>Reference 2</bibl><bibl>Reference 3</bibl>'
            '</listBibl></back></text></TEI>'
        )
        assert count_entity_elements('citation', tei_root) == 3

    def test_should_count_an_element_holding_only_a_label(self):
        # An element with no token of its own still exists in the TEI, and is the
        # difference between the element count and the entity count after parsing.
        tei_root = etree.fromstring(
            '<tei><text><listBibl>'
            '<bibl><label>1</label>Reference 1</bibl><bibl><label>2</label></bibl>'
            '</listBibl></text></tei>'
        )
        assert count_entity_elements('reference-segmenter', tei_root) == 2

    def test_should_return_none_for_a_model_declared_as_having_no_entity_count(self):
        tei_root = etree.fromstring('<tei><text>Some text</text></tei>')
        assert count_entity_elements('segmentation', tei_root) is None

    def test_should_return_none_for_an_undeclared_model(self):
        tei_root = etree.fromstring('<tei><text>Some text</text></tei>')
        assert count_entity_elements('not-a-model', tei_root) is None


class TestGetLabelsForModelDataList:
    def test_should_strip_the_begin_prefix(self):
        assert get_labels_for_model_data_list([
            _labeled('B-<title>'), _labeled('<title>')
        ]) == {'<title>'}

    def test_should_ignore_unlabeled_model_data(self):
        assert get_labels_for_model_data_list([
            LayoutModelData(data_line='token'), _labeled(None), _labeled('<date>')
        ]) == {'<date>'}


class TestCountCitationLabels:
    def test_should_count_a_label_the_jats_carries_and_the_data_marks(self):
        counts = count_citation_labels(
            [frozenset({JatsSubFieldNames.REFERENCE_ARTICLE_TITLE})],
            [[_labeled('<title>')]]
        )
        assert counts['<title>'] == {'jats': 1, 'marked': 1}

    def test_should_count_a_sub_field_the_page_does_not_print_as_unmarked(self):
        counts = count_citation_labels(
            [frozenset({JatsSubFieldNames.REFERENCE_DOI})],
            [[_labeled('<title>')]]
        )
        assert counts[IDENTIFIER_LABEL] == {'jats': 1, 'marked': 0}

    def test_should_count_identifiers_of_different_kinds_under_one_label(self):
        counts = count_citation_labels(
            [frozenset({
                JatsSubFieldNames.REFERENCE_DOI, JatsSubFieldNames.REFERENCE_PMID
            })],
            [[_labeled(IDENTIFIER_LABEL), _labeled('B-' + IDENTIFIER_LABEL)]]
        )
        assert counts[IDENTIFIER_LABEL] == {'jats': 1, 'marked': 1}

    def test_should_count_presence_per_reference_rather_than_occurrences(self):
        # <author> covers a whole author list, so several author tokens in one
        # reference are one reference marking the label.
        counts = count_citation_labels(
            [frozenset({JatsSubFieldNames.REFERENCE_AUTHOR})],
            [[_labeled('B-<author>'), _labeled('<author>'), _labeled('<author>')]]
        )
        assert counts['<author>'] == {'jats': 1, 'marked': 1}

    def test_should_count_a_sub_field_appearing_twice_in_one_reference_once(self):
        # fpage and lpage both map to <pages>, and <pages> is written once per
        # page number, so neither side may count an occurrence.
        counts = count_citation_labels(
            [frozenset({
                JatsSubFieldNames.REFERENCE_FPAGE, JatsSubFieldNames.REFERENCE_LPAGE
            })],
            [[_labeled('B-<pages>'), _labeled('B-<pages>')]]
        )
        assert counts['<pages>'] == {'jats': 1, 'marked': 1}

    def test_should_count_over_all_references(self):
        counts = count_citation_labels(
            [
                frozenset({JatsSubFieldNames.REFERENCE_ARTICLE_TITLE}),
                frozenset({JatsSubFieldNames.REFERENCE_ARTICLE_TITLE}),
                frozenset({JatsSubFieldNames.REFERENCE_YEAR}),
            ],
            [[_labeled('<title>')], [_labeled('<date>')], [_labeled('<date>')]]
        )
        assert counts['<title>'] == {'jats': 2, 'marked': 1}
        assert counts['<date>'] == {'jats': 1, 'marked': 2}
