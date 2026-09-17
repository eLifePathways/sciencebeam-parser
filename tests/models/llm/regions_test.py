import json

import pytest

from sciencebeam_parser.models.llm.decode import (
    render_layout_lines,
    LlmMalformedResponseError,
    decode_regions_response,
    get_regions_response_schema,
    render_numbered_line_texts
)
from sciencebeam_parser.models.llm.tasks import get_segmentation_region_names


LABELS = get_segmentation_region_names()

LINES = ['Title', 'Abstract', 'Introduction', 'Method', 'References', '[1] Smith']


def get_content(*regions) -> str:
    """Regions are given 1-based, as the prompt numbers them."""
    return json.dumps({
        'regions': [
            {'start': start + 1, 'end': end + 1, 'label': label}
            for start, end, label in regions
        ]
    })


def decode(content: str, line_texts=None, max_regions: int = 64):
    labels, _ = decode_regions_response(
        content, line_texts if line_texts is not None else LINES, LABELS, max_regions
    )
    return labels


def decode_with_unclaimed(content: str, max_regions: int = 64):
    return decode_regions_response(content, LINES, LABELS, max_regions)


class TestSegmentationRegionNames:
    def test_should_offer_the_names_a_publisher_would_use(self):
        assert LABELS == [
            'front_matter', 'body', 'acknowledgements', 'appendix', 'references',
            'other',
        ]


class TestRenderNumberedLineTexts:
    def test_should_number_from_one_with_a_tab(self):
        assert render_numbered_line_texts(['a', 'b']) == '1\ta\n2\tb'


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

    def test_should_collapse_a_region_per_line_into_one(self):
        labels = decode(get_content(
            *[(index, index, 'body') for index in range(len(LINES))]
        ))
        assert labels == ['B-<body>'] + ['I-<body>'] * (len(LINES) - 1)


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

    def test_a_start_below_one(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(json.dumps({'regions': [
                {'start': 0, 'end': 6, 'label': 'front_matter'}
            ]}))

    def test_a_start_that_is_not_an_integer(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [
                {'start': '1', 'end': 6, 'label': 'front_matter'}
            ]}))

    def test_a_boolean_start(self):
        with pytest.raises(LlmMalformedResponseError, match='not an integer'):
            decode(json.dumps({'regions': [
                {'start': True, 'end': 6, 'label': 'front_matter'}
            ]}))

    def test_a_region_that_ends_before_it_starts(self):
        with pytest.raises(LlmMalformedResponseError, match='before it starts'):
            decode(get_content((0, 3, 'front_matter'), (4, 2, 'body')))

    def test_an_overlap_between_regions(self):
        with pytest.raises(LlmMalformedResponseError, match='overlaps the next'):
            decode(get_content((0, 3, 'front_matter'), (2, 5, 'body')))

    def test_a_region_touching_the_next_one(self):
        with pytest.raises(LlmMalformedResponseError, match='overlaps the next'):
            decode(get_content((0, 3, 'front_matter'), (3, 5, 'body')))

    def test_a_label_outside_the_closed_set(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 5, 'headnote')))

    def test_a_grobid_label_rather_than_a_region_name(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 5, 'header')))

    def test_a_label_the_pipeline_predicts_but_the_prompt_does_not_offer(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(get_content((0, 5, 'footnote')))

    def test_a_missing_label(self):
        with pytest.raises(LlmMalformedResponseError, match='is not one of'):
            decode(json.dumps({'regions': [{'start': 1, 'end': 6}]}))

    def test_an_entry_carrying_document_text(self):
        with pytest.raises(LlmMalformedResponseError, match='unexpected key'):
            decode(json.dumps({'regions': [
                {'start': 1, 'end': 6, 'label': 'front_matter', 'text': 'Title'}
            ]}))

    def test_an_entry_that_is_not_an_object(self):
        with pytest.raises(LlmMalformedResponseError, match='malformed'):
            decode(json.dumps({'regions': [[1, 6, 'front_matter']]}))

    def test_more_regions_than_the_bound_allows(self):
        # counted as answered, before merging: the bound is about what a model
        # generated, since that is what runs it out of output
        content = get_content(*[
            (index, index, 'body' if index % 2 else 'references')
            for index in range(len(LINES))
        ])
        with pytest.raises(LlmMalformedResponseError, match='exceeds max_regions'):
            decode(content, max_regions=3)


class TestUnclaimedLines:
    """A line no region claims is `<other>`, which the pipeline reads no field
    from. That is how a running head or a page number leaves the output instead
    of joining whichever region surrounds it.
    """

    def test_should_label_a_gap_as_other(self):
        labels, unclaimed = decode_with_unclaimed(get_content(
            (0, 1, 'front_matter'), (3, 5, 'body')
        ))
        assert labels == [
            'B-<header>', 'I-<header>',
            'B-<other>',
            'B-<body>', 'I-<body>', 'I-<body>',
        ]
        assert unclaimed == 1

    def test_should_label_lines_before_the_first_region_as_other(self):
        labels, unclaimed = decode_with_unclaimed(get_content((2, 5, 'body')))
        assert labels[:3] == ['B-<other>', 'I-<other>', 'B-<body>']
        assert unclaimed == 2

    def test_should_label_lines_after_the_last_region_as_other(self):
        labels, unclaimed = decode_with_unclaimed(get_content((0, 3, 'body')))
        assert labels[4:] == ['B-<other>', 'I-<other>']
        assert unclaimed == 2

    def test_should_report_no_unclaimed_lines_when_regions_tile_the_document(self):
        _, unclaimed = decode_with_unclaimed(get_content(
            (0, 1, 'front_matter'), (2, 5, 'body')
        ))
        assert unclaimed == 0

    def test_should_still_label_every_line(self):
        labels, _ = decode_with_unclaimed(get_content((1, 1, 'body'), (4, 4, 'references')))
        assert len(labels) == len(LINES)


class TestRegionsSchemaBound:
    """`maxItems` binds on the shipped provider, so the bound is expressed where
    it can prevent an answer rather than only reject one.
    """

    def test_should_not_cap_the_array_in_the_schema(self):
        # maxItems binds here, and closing the array mid-document loses the tail
        # silently; the count is checked on decode instead
        assert 'maxItems' not in get_regions_response_schema(LABELS)['properties']['regions']


class TestMergeAdjacentRegions:
    """A model asked for regions subdivides continuous text: one measured answer
    split a body into 62 consecutive `body` regions and ran out of room before
    reaching the bibliography.
    """

    def test_should_merge_touching_regions_of_the_same_kind(self):
        labels = decode(get_content(
            (0, 1, 'body'), (2, 3, 'body'), (4, 5, 'body')
        ))
        assert labels == ['B-<body>'] + ['I-<body>'] * 5

    def test_should_keep_regions_of_different_kinds_apart(self):
        labels = decode(get_content((0, 2, 'body'), (3, 5, 'references')))
        assert labels[3] == 'B-<references>'

    def test_should_keep_same_kind_regions_that_a_gap_separates(self):
        labels = decode(get_content((0, 1, 'body'), (3, 5, 'body')))
        assert labels == [
            'B-<body>', 'I-<body>', 'B-<other>',
            'B-<body>', 'I-<body>', 'I-<body>',
        ]


class TestRenderLayoutLines:
    def test_should_mark_pages_blocks_and_emphasis_without_numbering_them(self):
        rendered = render_layout_lines(
            ['Title', 'Author', 'Intro', 'text'],
            ['BLOCKSTART', 'BLOCKSTART', 'BLOCKIN', 'BLOCKIN'],
            ['PAGESTART', 'PAGEIN', 'PAGESTART', 'PAGEIN'],
            ['1', '0', '0', '0'],
            ['0', '1', '0', '0'],
        )
        assert rendered == '\n'.join([
            '1\t**Title**',
            '',
            '2\t*Author*',
            '--- page 2 ---',
            '3\tIntro',
            '4\ttext',
        ])


class TestOtherRegion:
    """`other` is offered so furniture can be named rather than skipped: asking a
    model to leave lines out is a negation, and it read the surrounding region as
    the answer instead.
    """

    def test_should_map_other_to_the_label_nothing_reads(self):
        labels = decode(get_content(
            (0, 1, 'front_matter'), (2, 2, 'other'), (3, 5, 'body')
        ))
        assert labels == [
            'B-<header>', 'I-<header>', 'B-<other>',
            'B-<body>', 'I-<body>', 'I-<body>',
        ]

    def test_should_report_a_named_other_region_as_claimed(self):
        _, unclaimed = decode_with_unclaimed(get_content(
            (0, 1, 'front_matter'), (2, 2, 'other'), (3, 5, 'body')
        ))
        assert unclaimed == 0
