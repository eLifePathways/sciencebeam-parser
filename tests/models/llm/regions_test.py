import json

import pytest

from sciencebeam_parser.models.llm.decode import (
    LlmMalformedResponseError,
    decode_regions_response,
    get_regions_response_schema,
    render_numbered_line_texts
)
from sciencebeam_parser.models.llm.tasks import get_segmentation_region_names


LABELS = get_segmentation_region_names()

LINES = ['Title', 'Abstract', 'Introduction', 'Method', 'References', '[1] Smith']


def get_content(*regions) -> str:
    return json.dumps({
        'regions': [
            {'start': start, 'end': end, 'label': label}
            for start, end, label in regions
        ]
    })


def decode(content: str, line_texts=None, max_regions: int = 64):
    return decode_regions_response(
        content, line_texts if line_texts is not None else LINES, LABELS, max_regions
    )


class TestSegmentationRegionNames:
    def test_should_offer_the_names_a_publisher_would_use(self):
        assert LABELS == [
            'front_matter', 'body', 'acknowledgements', 'appendix', 'references'
        ]


class TestRenderNumberedLineTexts:
    def test_should_number_from_zero_with_a_tab(self):
        assert render_numbered_line_texts(['a', 'b']) == '0\ta\n1\tb'


class TestRegionsResponseSchema:
    def test_should_close_the_label_set(self):
        schema = get_regions_response_schema(LABELS)
        assert schema['properties']['regions']['items']['properties']['label']['enum'] == LABELS

    def test_should_refuse_properties_it_does_not_name(self):
        schema = get_regions_response_schema(LABELS)
        assert schema['additionalProperties'] is False
        assert schema['properties']['regions']['items']['additionalProperties'] is False

    def test_should_ask_for_the_span_before_the_label(self):
        # generation follows property order, so the indices are committed to first
        schema = get_regions_response_schema(LABELS)
        assert list(schema['properties']['regions']['items']['properties']) == [
            'start', 'end', 'label'
        ]


class TestDecodeRegionsResponse:
    def test_should_map_the_region_name_to_the_label_the_pipeline_reads(self):
        labels = decode(get_content(
            (0, 1, 'front_matter'), (2, 3, 'body'), (4, 5, 'references')
        ))
        assert labels == [
            'B-<header>', 'I-<header>',
            'B-<body>', 'I-<body>',
            'B-<references>', 'I-<references>',
        ]

    def test_should_map_every_name(self):
        assert decode(get_content(
            (0, 0, 'front_matter'), (1, 1, 'body'), (2, 2, 'acknowledgements'),
            (3, 3, 'appendix'), (4, 5, 'references')
        )) == [
            'B-<header>', 'B-<body>', 'B-<acknowledgement>',
            'B-<annex>', 'B-<references>', 'I-<references>',
        ]

    def test_should_accept_one_region_covering_the_document(self):
        assert decode(get_content((0, 5, 'front_matter'))) == (
            ['B-<header>'] + ['I-<header>'] * 5
        )

    def test_should_allow_a_label_to_repeat(self):
        labels = decode(get_content(
            (0, 0, 'front_matter'), (1, 1, 'body'), (2, 5, 'front_matter')
        ))
        assert labels[:3] == ['B-<header>', 'B-<body>', 'B-<header>']

    def test_should_accept_a_region_per_line(self):
        labels = decode(get_content(
            *[(index, index, 'body') for index in range(len(LINES))]
        ))
        assert labels == ['B-<body>'] * len(LINES)


class TestDecodeRegionsResponseRejects:
    """Each guarantee is enforced on decode rather than trusted to the provider's
    schema, so a response that ignores the schema is refused rather than absorbed.
    """

    def test_a_response_that_is_not_json(self):
        with pytest.raises(LlmMalformedResponseError, match='not json'):
            decode('{"regions": [')

    def test_a_response_without_regions(self):
        with pytest.raises(LlmMalformedResponseError, match='no "regions"'):
            decode(json.dumps({'starts': [0]}))

    def test_regions_that_are_not_a_list(self):
        with pytest.raises(LlmMalformedResponseError, match='not a list'):
            decode(json.dumps({'regions': {'start': 0}}))

    def test_an_empty_region_list(self):
        with pytest.raises(LlmMalformedResponseError, match='empty'):
            decode(json.dumps({'regions': []}))

    def test_a_start_beyond_the_last_line(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(get_content((0, 1, 'front_matter'), (99, 99, 'body')))

    def test_an_end_beyond_the_last_line(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(get_content((0, 99, 'front_matter')))

    def test_a_negative_start(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(get_content((-1, 5, 'front_matter')))

    def test_a_start_that_is_not_an_integer(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [
                {'start': '0', 'end': 5, 'label': 'front_matter'}
            ]}))

    def test_a_boolean_start(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [
                {'start': True, 'end': 5, 'label': 'front_matter'}
            ]}))

    def test_a_region_that_ends_before_it_starts(self):
        with pytest.raises(LlmMalformedResponseError, match='before it starts'):
            decode(get_content((0, 3, 'front_matter'), (4, 2, 'body')))

    def test_a_gap_between_regions(self):
        # the dominant failure: a region left out, and the response says so
        with pytest.raises(LlmMalformedResponseError, match='a gap between'):
            decode(get_content((0, 1, 'front_matter'), (4, 5, 'references')))

    def test_an_overlap_between_regions(self):
        with pytest.raises(LlmMalformedResponseError, match='an overlap between'):
            decode(get_content((0, 3, 'front_matter'), (2, 5, 'body')))

    def test_a_first_region_that_leaves_lines_unlabelled(self):
        with pytest.raises(LlmMalformedResponseError, match='rather than 0'):
            decode(get_content((1, 5, 'front_matter')))

    def test_a_last_region_that_stops_short_of_the_document(self):
        with pytest.raises(LlmMalformedResponseError, match='rather than 5'):
            decode(get_content((0, 4, 'front_matter')))

    def test_a_label_outside_the_closed_set(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 5, 'headnote')))

    def test_a_grobid_label_rather_than_a_region_name(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 5, 'header')))

    def test_a_missing_label(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(json.dumps({'regions': [{'start': 0, 'end': 5}]}))

    def test_an_entry_carrying_document_text(self):
        with pytest.raises(LlmMalformedResponseError, match='unexpected key'):
            decode(json.dumps({'regions': [
                {'start': 0, 'end': 5, 'label': 'front_matter', 'text': 'Title'}
            ]}))

    def test_an_entry_that_is_not_an_object(self):
        with pytest.raises(LlmMalformedResponseError, match='malformed'):
            decode(json.dumps({'regions': [[0, 5, 'front_matter']]}))

    def test_more_regions_than_the_bound_allows(self):
        content = get_content(
            *[(index, index, 'body') for index in range(len(LINES))]
        )
        with pytest.raises(LlmMalformedResponseError, match='exceeds max_regions'):
            decode(content, max_regions=3)
