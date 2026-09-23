import json

import pytest

from sciencebeam_parser.models.llm.decode import (
    clip_regions_to_core,
    get_line_windows,
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


def decode(content: str, line_texts=None):
    labels, _, _ = decode_regions_response(
        content, line_texts if line_texts is not None else LINES, LABELS
    )
    return labels


def decode_with_unclaimed(content: str):
    labels, unclaimed, _ = decode_regions_response(content, LINES, LABELS)
    return labels, unclaimed


def decode_with_touching(content: str):
    labels, _, touching = decode_regions_response(content, LINES, LABELS)
    return labels, touching


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

    def test_an_overlap_of_more_than_one_line(self):
        with pytest.raises(LlmMalformedResponseError, match='overlaps the next'):
            decode(get_content((0, 3, 'front_matter'), (2, 5, 'body')))

    def test_an_overlap_that_would_empty_the_earlier_region(self):
        with pytest.raises(LlmMalformedResponseError, match='overlaps the next'):
            decode(get_content((0, 0, 'front_matter'), (0, 5, 'body')))

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


class TestTouchingRegions:
    """Models chain regions -- 79..107 then 107..126 -- reading the end as where
    the next begins. Both statements are true only if the earlier region ends a
    line sooner, so that is how it is read.
    """

    def test_should_read_a_touching_pair_as_ending_a_line_sooner(self):
        labels, touching = decode_with_touching(get_content(
            (0, 2, 'front_matter'), (2, 5, 'body')
        ))
        assert labels == [
            'B-<header>', 'I-<header>',
            'B-<body>', 'I-<body>', 'I-<body>', 'I-<body>',
        ]
        assert touching == 1

    def test_should_resolve_a_whole_chain(self):
        labels, touching = decode_with_touching(get_content(
            (0, 2, 'front_matter'), (2, 4, 'body'), (4, 5, 'references')
        ))
        assert labels == [
            'B-<header>', 'I-<header>',
            'B-<body>', 'I-<body>',
            'B-<references>', 'I-<references>',
        ]
        assert touching == 2

    def test_should_report_none_when_regions_already_meet_cleanly(self):
        _, touching = decode_with_touching(get_content(
            (0, 1, 'front_matter'), (2, 5, 'body')
        ))
        assert touching == 0


class TestMaxLineChars:
    def test_should_keep_the_whole_line_by_default(self):
        assert render_numbered_line_texts(['a much longer line']) == '1\ta much longer line'

    def test_should_cut_each_line_to_its_first_characters(self):
        assert render_numbered_line_texts(
            ['abcdefghij', 'short'], max_line_chars=4
        ) == '1\tabcd\n2\tshor'


class TestLineWindows:
    """Cores tile the document, so no line is answered for twice and there is
    nothing to reconcile where two windows meet. The overlap is context.
    """

    def test_should_be_one_window_when_the_document_fits(self):
        assert get_line_windows(100, 400, 40) == [(0, 100, 0, 100)]

    def test_should_be_one_window_when_windowing_is_off(self):
        assert get_line_windows(5000, 0, 40) == [(0, 5000, 0, 5000)]

    def test_should_tile_the_document_with_cores(self):
        windows = get_line_windows(1000, 400, 40)
        assert [(w.core_start, w.core_end) for w in windows] == [
            (0, 400), (400, 800), (800, 1000)
        ]

    def test_should_give_each_core_context_either_side(self):
        windows = get_line_windows(1000, 400, 40)
        assert [(w.context_start, w.context_end) for w in windows] == [
            (0, 440), (360, 840), (760, 1000)
        ]

    def test_should_leave_no_line_uncovered(self):
        covered = [
            line for w in get_line_windows(937, 400, 40)
            for line in range(w.core_start, w.core_end)
        ]
        assert covered == list(range(937))


class TestClipRegionsToCore:
    def test_should_move_window_coordinates_into_the_document(self):
        window = get_line_windows(1000, 400, 40)[1]      # context 360..840
        assert clip_regions_to_core([(50, 100, 'body')], window) == [(410, 460, 'body')]

    def test_should_drop_a_region_that_lies_wholly_in_the_context(self):
        window = get_line_windows(1000, 400, 40)[1]
        assert not clip_regions_to_core([(0, 20, 'body')], window)

    def test_should_cut_a_region_back_to_the_core(self):
        window = get_line_windows(1000, 400, 40)[1]
        assert clip_regions_to_core([(0, 100, 'body')], window) == [(400, 460, 'body')]


class TestEndOneOffTheEnd:
    """A region running to the end of the input is the common case, and windowing
    makes it the case for every window but the last.
    """

    def test_should_read_an_end_one_past_the_last_line_as_the_last_line(self):
        assert decode(json.dumps({'regions': [
            {'start': 1, 'end': len(LINES) + 1, 'label': 'body'}
        ]})) == ['B-<body>'] + ['I-<body>'] * (len(LINES) - 1)

    def test_should_still_reject_an_end_further_out(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(json.dumps({'regions': [
                {'start': 1, 'end': len(LINES) + 2, 'label': 'body'}
            ]}))

    def test_should_drop_a_trailing_region_that_begins_past_the_last_line(self):
        """One entry too many, chained off a region that already ran to the end."""
        assert decode(json.dumps({'regions': [
            {'start': 1, 'end': len(LINES), 'label': 'body'},
            {'start': len(LINES) + 1, 'end': len(LINES) + 1, 'label': 'references'},
        ]})) == ['B-<body>'] + ['I-<body>'] * (len(LINES) - 1)

    def test_should_reject_a_response_whose_only_region_begins_past_the_end(self):
        with pytest.raises(LlmMalformedResponseError, match='no line has a label'):
            decode(json.dumps({'regions': [
                {'start': len(LINES) + 1, 'end': len(LINES) + 1, 'label': 'body'}
            ]}))

    def test_should_still_reject_a_start_further_out(self):
        with pytest.raises(LlmMalformedResponseError, match='out of range'):
            decode(json.dumps({'regions': [
                {'start': len(LINES) + 2, 'end': len(LINES) + 2, 'label': 'body'}
            ]}))
