"""The regions shape through `predict_labels`, where segmentation differs from
every other task: its rows are already lines, so the text is a feature column.
"""
import json
from typing import Optional

import pytest

from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.decode import LlmInputTooLargeError
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl

from tests.models.llm.model_impl_test import FakeClient


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


def get_segmentation_model_impl(content: Optional[str] = None, **overrides):
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


class TestRegionsPromptVersions:
    def test_should_state_the_documents_own_line_range_in_v2(self):
        model_impl = get_segmentation_model_impl(
            get_regions_content((0, 3, 'front_matter')), prompt_version='regions-v2'
        )
        model_impl.predict_labels([SEGMENTATION_TOKENS], [segmentation_feature_rows()])
        prompt = model_impl.client.prompts[0]
        assert '<1..4>' in prompt
        assert '{{last_line}}' not in prompt

    def test_should_leave_the_shape_block_out_of_v1(self):
        model_impl = get_segmentation_model_impl(
            get_regions_content((0, 3, 'front_matter'))
        )
        model_impl.predict_labels([SEGMENTATION_TOKENS], [segmentation_feature_rows()])
        assert '"regions"' not in model_impl.client.prompts[0]
