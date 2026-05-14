import logging
from typing import Iterable

import pytest

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta
)

from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    feature_linear_scaling_int
)
from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.models.segmentation.data import (
    NBBINS_POSITION,
    SegmentationLineFeatures,
    SegmentationLineFeaturesProvider,
    calculate_page_main_areas,
    get_text_pattern
)


LOGGER = logging.getLogger(__name__)


@pytest.fixture(name='features_provider')
def _features_provider():
    return SegmentationLineFeaturesProvider(
        document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
        use_first_token_of_block=False
    )


class TestGetTextPattern:
    def test_should_keep_lowercase_characters(self):
        assert get_text_pattern('abc') == 'abc'

    def test_should_keep_uppercase_characters_and_convert_to_lowercase(self):
        assert get_text_pattern('ABC') == 'abc'

    def test_should_keep_spaces(self):
        assert get_text_pattern('abc abc') == 'abc abc'

    def test_should_remove_punctuation(self):
        assert get_text_pattern('abc.,:;') == 'abc'

    def test_should_remove_digits(self):
        assert get_text_pattern('abc123') == 'abc'


def _iter_line_features(
    features_provider: SegmentationLineFeaturesProvider,
    layout_document: LayoutDocument
) -> Iterable[SegmentationLineFeatures]:
    yield from features_provider.iter_line_features(
        layout_document
    )


class TestCalculatePageMainAreas:
    def test_should_use_full_page_when_only_odd_pages(self):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=600, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        assert result[1] == BoundingBox(0, 0, 500, 700)

    def test_should_use_full_page_when_no_page_coordinates(self):
        block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=600, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('text', coordinates=block_coords)
            ])])
        ])
        result = calculate_page_main_areas(layout_document)
        assert result[0] is None

    def test_should_filter_small_blocks(self):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        # height=15 < 20: filtered out
        small_block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=15, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=small_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        # filtered → fallback to full page
        assert result[1] == BoundingBox(0, 0, 500, 700)

    def test_should_compute_main_area_from_odd_and_even_pages(self):
        odd_page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        even_page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=2)
        odd_block_coords = LayoutPageCoordinates(x=50, y=60, width=400, height=580, page_number=1)
        even_block_coords = LayoutPageCoordinates(x=60, y=60, width=380, height=580, page_number=2)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=odd_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(odd_page_coords)
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=even_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(even_page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        # odd page: min_left=50, max_right=450, max_bottom=640
        # width = 450 - 50 + 1 = 401, height = 640 - 60 + 1 = 581
        assert result[1] == BoundingBox(50, 60, 401, 581)
        # even page: min_left=60, max_right=440
        # width = 440 - 60 + 1 = 381, height = 581
        assert result[2] == BoundingBox(60, 60, 381, 581)


class TestSegmentationLineFeaturesProvider:
    def test_should_provide_page_and_block_status_for_multi_line_blocks(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('line1'),
                LayoutLine.for_text('line2'),
                LayoutLine.for_text('line3')
            ])])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'page_status': features.get_page_status(),
                'block_status': features.get_block_status()
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {'page_status': 'PAGESTART', 'block_status': 'BLOCKSTART'},
            {'page_status': 'PAGEIN', 'block_status': 'BLOCKIN'},
            {'page_status': 'PAGEEND', 'block_status': 'BLOCKEND'}
        ]

    def test_should_provide_page_and_block_status_for_single_token_blocks(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[
                LayoutBlock.for_text('line1'),
                LayoutBlock.for_text('line2'),
                LayoutBlock.for_text('line3')
            ])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'page_status': features.get_page_status(),
                'block_status': features.get_block_status()
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {'page_status': 'PAGESTART', 'block_status': 'BLOCKSTART'},
            {'page_status': 'PAGEIN', 'block_status': 'BLOCKSTART'},
            {'page_status': 'PAGEEND', 'block_status': 'BLOCKSTART'}
        ]

    def test_should_provide_line_text(self, features_provider: SegmentationLineFeaturesProvider):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('first1 second1 this is a line'),
                LayoutLine.for_text('first2 second2 this is a line')
            ])])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'line_text': features.line_text,
                'token_text': features.token_text,
                'second_token_text': features.second_token_text
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {
                'line_text': 'first1 second1 this is a line',
                'token_text': 'first1',
                'second_token_text': 'second1'
            },
            {
                'line_text': 'first2 second2 this is a line',
                'token_text': 'first2',
                'second_token_text': 'second2'
            },
        ]

    def test_should_provide_punctuation_profile(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('a .: b'),
            ])])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'line_punctuation_profile': features.get_line_punctuation_profile(),
                'line_punctuation_profile_length_feature': (
                    features.get_line_punctuation_profile_length_feature()
                ),
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {
                'line_punctuation_profile': '.:',
                'line_punctuation_profile_length_feature': '2'
            },
        ]

    def test_should_provide_block_relative_line_length(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('1'),
                LayoutLine.for_text('12'),
                LayoutLine.for_text('1234567890'),
            ])])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'str_block_relative_line_length_feature': (
                    features.get_str_block_relative_line_length_feature()
                )
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {
                'str_block_relative_line_length_feature': '1',  # 1 * 10 / 10
            },
            {
                'str_block_relative_line_length_feature': '2',  # 2 * 10 / 10
            },
            {
                'str_block_relative_line_length_feature': '10',  # 10 * 10 / 10
            },
        ]

    def test_should_provide_block_relative_document_token_position(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text(f'line{i}')
                for i in range(10)
            ])])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'str_relative_document_position': (
                    features.get_str_relative_document_position()
                )
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {
                'str_relative_document_position': str(feature_linear_scaling_int(
                    i, 10, NBBINS_POSITION
                )),
            }
            for i in range(10)
        ]

    def test_should_provide_is_main_area_true_when_block_inside_page(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        block_coords = LayoutPageCoordinates(x=100, y=100, width=50, height=20, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        feature_values = [
            features.get_str_is_main_area()
            for features in _iter_line_features(features_provider, layout_document)
        ]
        assert feature_values == ['1']

    def test_should_provide_is_main_area_false_when_block_outside_page(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        block_coords = LayoutPageCoordinates(x=600, y=100, width=50, height=20, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        feature_values = [
            features.get_str_is_main_area()
            for features in _iter_line_features(features_provider, layout_document)
        ]
        assert feature_values == ['0']

    def test_should_provide_is_main_area_false_when_no_page_coordinates(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        block_coords = LayoutPageCoordinates(x=100, y=100, width=50, height=20, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('text', coordinates=block_coords)
            ])])
        ])
        feature_values = [
            features.get_str_is_main_area()
            for features in _iter_line_features(features_provider, layout_document)
        ]
        assert feature_values == ['0']

    def test_should_provide_is_main_area_true_when_no_block_coordinates(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[LayoutLine.for_text('text')])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        feature_values = [
            features.get_str_is_main_area()
            for features in _iter_line_features(features_provider, layout_document)
        ]
        assert feature_values == ['1']

    def test_should_provide_repetitive_pattern_feature(
        self,
        features_provider: SegmentationLineFeaturesProvider
    ):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[
                LayoutBlock.for_text('this is repetitive'),
                LayoutBlock.for_text('this is not')
            ]),
            LayoutPage(blocks=[
                LayoutBlock.for_text('this is repetitive'),
                LayoutBlock.for_text('it is different')
            ])
        ])
        feature_values = []
        for features in _iter_line_features(features_provider, layout_document):
            feature_values.append({
                'get_str_is_repetitive_pattern': (
                    features.get_str_is_repetitive_pattern()
                ),
                'get_str_is_first_repetitive_pattern': (
                    features.get_str_is_first_repetitive_pattern()
                )
            })
        LOGGER.debug('feature_values: %r', feature_values)
        assert feature_values == [
            {
                'get_str_is_repetitive_pattern': '1',
                'get_str_is_first_repetitive_pattern': '1'
            },
            {
                'get_str_is_repetitive_pattern': '0',
                'get_str_is_first_repetitive_pattern': '0'
            },
            {
                'get_str_is_repetitive_pattern': '1',
                'get_str_is_first_repetitive_pattern': '0'
            },
            {
                'get_str_is_repetitive_pattern': '0',
                'get_str_is_first_repetitive_pattern': '0'
            },
        ]
