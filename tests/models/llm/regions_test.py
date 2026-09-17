import json

import pytest

from sciencebeam_parser.models.llm.decode import (
    LlmMalformedResponseError,
    decode_regions_response,
    get_regions_response_schema,
    render_numbered_line_texts
)
from sciencebeam_parser.models.llm.tasks import get_segmentation_labels


LABELS = get_segmentation_labels()

LINES = ['Title', 'Abstract', 'Introduction', 'Method', 'References', '[1] Smith']


def get_content(*regions) -> str:
    return json.dumps({
        'regions': [{'start': start, 'label': label} for start, label in regions]
    })


def decode(content: str, line_texts=None, max_regions: int = 64):
    return decode_regions_response(
        content, line_texts if line_texts is not None else LINES, LABELS, max_regions
    )


class TestSegmentationLabels:
    def test_should_be_the_five_the_pipeline_consumes(self):
        assert LABELS == ['header', 'body', 'acknowledgement', 'annex', 'references']


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

    def test_should_ask_for_the_start_before_the_label(self):
        # generation follows property order, so the index is committed to first
        schema = get_regions_response_schema(LABELS)
        assert list(schema['properties']['regions']['items']['properties']) == ['start', 'label']


class TestDecodeRegionsResponse:
    def test_should_label_every_line(self):
        labels = decode(get_content((0, 'header'), (2, 'body'), (4, 'references')))
        assert labels == [
            'B-<header>', 'I-<header>',
            'B-<body>', 'I-<body>',
            'B-<references>', 'I-<references>',
        ]

    def test_should_let_the_last_region_reach_the_final_line(self):
        labels = decode(get_content((0, 'header')))
        assert labels == ['B-<header>'] + ['I-<header>'] * 5

    def test_should_allow_a_label_to_repeat(self):
        labels = decode(get_content((0, 'header'), (1, 'body'), (2, 'header')))
        assert labels[:3] == ['B-<header>', 'B-<body>', 'B-<header>']

    def test_should_start_a_region_on_every_line_when_asked(self):
        labels = decode(get_content(*[(index, 'body') for index in range(len(LINES))]))
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
            decode(get_content((0, 'header'), (99, 'body')))

    def test_a_negative_start(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(get_content((-1, 'header')))

    def test_a_start_that_is_not_an_integer(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [{'start': '0', 'label': 'header'}]}))

    def test_a_boolean_start(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [{'start': True, 'label': 'header'}]}))

    def test_starts_that_do_not_ascend(self):
        with pytest.raises(LlmMalformedResponseError, match='ascending'):
            decode(get_content((0, 'header'), (4, 'body'), (2, 'references')))

    def test_a_repeated_start(self):
        with pytest.raises(LlmMalformedResponseError, match='ascending'):
            decode(get_content((0, 'header'), (2, 'body'), (2, 'references')))

    def test_a_first_region_that_leaves_lines_unlabelled(self):
        with pytest.raises(LlmMalformedResponseError, match='rather than 0'):
            decode(get_content((1, 'header')))

    def test_a_label_outside_the_closed_set(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 'headnote')))

    def test_a_missing_label(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(json.dumps({'regions': [{'start': 0}]}))

    def test_an_entry_carrying_document_text(self):
        with pytest.raises(LlmMalformedResponseError, match='unexpected key'):
            decode(json.dumps({'regions': [
                {'start': 0, 'label': 'header', 'text': 'Title'}
            ]}))

    def test_an_entry_that_is_not_an_object(self):
        with pytest.raises(LlmMalformedResponseError, match='malformed'):
            decode(json.dumps({'regions': [[0, 'header']]}))

    def test_more_regions_than_the_bound_allows(self):
        content = get_content(*[(index, 'body') for index in range(len(LINES))])
        with pytest.raises(LlmMalformedResponseError, match='exceeds max_regions'):
            decode(content, max_regions=3)
