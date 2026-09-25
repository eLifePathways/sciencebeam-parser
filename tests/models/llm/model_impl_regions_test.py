"""The regions shape through `predict_labels`, where segmentation differs from
every other task: its rows are already lines, so the text is a feature column.
"""
import json

import pytest

from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.decode import (
    LlmInputTooLargeError, LlmMalformedResponseError
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl

from tests.models.llm.model_impl_test import FakeClient, FakeContent


SEGMENTATION_CONFIG = {
    'task': 'segmentation',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'regions-v1',
    'response_shape': 'regions',
    'provider': 'venice',
}

SEGMENTATION_LINE_TEXTS = ['Title of it', 'Introduction', 'References', '[1] Smith']

WHOLE_LINE_TEXT_INDEX = get_feature_column_index('segmentation', 'whole_line_text')


def segmentation_feature_rows():
    """Segmentation's rows are lines, so one row per line and the text is a column."""
    return [
        ['x'] * WHOLE_LINE_TEXT_INDEX + [line_text]
        for line_text in SEGMENTATION_LINE_TEXTS
    ]


SEGMENTATION_TOKENS = ['Title', 'Introduction', 'References', '[1]']


def get_segmentation_model_impl(content: FakeContent = None, **overrides):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config({**SEGMENTATION_CONFIG, **overrides}),
        client=FakeClient(content=content)
    )


def get_regions_content(*regions) -> str:
    """Regions are given 1-based, as the prompt numbers them."""
    return json.dumps({
        'regions': [
            {'start': start + 1, 'end': end + 1, 'label': label}
            for start, end, label in regions
        ]
    })


class TestLlmModelImplRegionsShape:
    def test_should_resolve_the_whole_line_text_column_by_name(self):
        model_impl = get_segmentation_model_impl(get_regions_content((0, 3, 'front_matter')))
        assert model_impl.whole_line_text_index == WHOLE_LINE_TEXT_INDEX

    def test_should_label_every_row(self):
        model_impl = get_segmentation_model_impl(
            get_regions_content(
                (0, 0, 'front_matter'), (1, 1, 'body'), (2, 3, 'references')
            )
        )
        result = model_impl.predict_labels(
            [SEGMENTATION_TOKENS], [segmentation_feature_rows()]
        )
        assert result[0] == [
            ('Title', 'B-<header>'),
            ('Introduction', 'B-<body>'),
            ('References', 'B-<references>'),
            ('[1]', 'I-<references>'),
        ]

    def test_should_prompt_with_the_line_text_rather_than_the_row_token(self):
        model_impl = get_segmentation_model_impl(get_regions_content((0, 3, 'front_matter')))
        model_impl.predict_labels([SEGMENTATION_TOKENS], [segmentation_feature_rows()])
        prompt = model_impl.client.prompts[0]
        assert '1\tTitle of it' in prompt
        assert '4\t[1] Smith' in prompt

    def test_should_close_the_label_set_in_the_schema_it_sends(self):
        model_impl = get_segmentation_model_impl(get_regions_content((0, 3, 'front_matter')))
        assert model_impl.labels == [
            'front_matter', 'body', 'acknowledgements', 'appendix', 'references',
            'other',
        ]

    def test_should_return_nothing_for_an_empty_document(self):
        model_impl = get_segmentation_model_impl(get_regions_content((0, 3, 'front_matter')))
        assert model_impl.predict_labels([[]], [[]]) == [[]]

    def test_should_raise_when_the_document_exceeds_max_input_lines(self):
        model_impl = get_segmentation_model_impl(
            get_regions_content((0, 3, 'front_matter')), max_input_lines=2
        )
        with pytest.raises(LlmInputTooLargeError, match='may not label reliably'):
            model_impl.predict_labels(
                [SEGMENTATION_TOKENS], [segmentation_feature_rows()]
            )


WINDOWED = {'window_lines': 2, 'window_overlap': 0, 'max_concurrent_requests': 1}

FIRST_WINDOW_ANSWER = json.dumps(
    {'regions': [{'start': 1, 'end': 2, 'label': 'front_matter'}]}
)


class TestWindowThatFails:
    """A window whose call fails is carried by the region before it. The first
    window has no region before it."""

    def test_should_fail_the_document_when_the_first_window_fails(self):
        model_impl = get_segmentation_model_impl(content='not json', **WINDOWED)
        with pytest.raises(LlmMalformedResponseError) as exc_info:
            model_impl.predict_labels(
                [SEGMENTATION_TOKENS], [segmentation_feature_rows()]
            )
        assert 'no region before it to carry' in str(exc_info.value)

    def test_should_carry_the_previous_region_when_a_later_window_fails(self):
        model_impl = get_segmentation_model_impl(
            content=[FIRST_WINDOW_ANSWER, 'not json'], **WINDOWED
        )
        result = model_impl.predict_labels(
            [SEGMENTATION_TOKENS], [segmentation_feature_rows()]
        )
        assert [label for _, label in result[0]] == [
            'B-<header>', 'I-<header>', 'I-<header>', 'I-<header>'
        ]
