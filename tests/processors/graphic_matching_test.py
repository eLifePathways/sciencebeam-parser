import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pytest

import PIL.Image

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutGraphic,
    LayoutLine,
    LayoutLineMeta,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticFigure,
    SemanticGraphic,
    SemanticLabel,
    SemanticMixedContentWrapper
)
from sciencebeam_parser.processors.graphic_matching import (
    BoundingBoxDistance,
    BoundingBoxDistanceBetween,
    BoundingBoxDistanceGraphicMatcher,
    GraphicRelatedBlockTextGraphicMatcher,
    OpticalCharacterRecognitionGraphicMatcher,
    get_normalized_bounding_box_for_page_coordinates_and_page_meta,
    get_bounding_box_list_distance,
    get_normalized_bounding_box_list_for_layout_block,
    get_normalized_bounding_box_list_for_layout_graphic
)


LOGGER = logging.getLogger(__name__)


PAGE_META_1 = LayoutPageMeta(
    page_number=1,
    coordinates=LayoutPageCoordinates(
        x=0,
        y=0,
        width=400,
        height=600,
        page_number=1
    )
)

LINE_META_1 = LayoutLineMeta(
    line_id=1,
    page_meta=PAGE_META_1
)

COORDINATES_1 = LayoutPageCoordinates(
    x=10,
    y=100,
    width=200,
    height=100,
    page_number=1
)


GRAPHIC_ABOVE_FIGURE_COORDINATES_1 = LayoutPageCoordinates(
    x=10,
    y=100,
    width=200,
    height=100,
    page_number=1
)

FIGURE_BELOW_GRAPHIC_COORDINATES_1 = LayoutPageCoordinates(
    x=10,
    y=GRAPHIC_ABOVE_FIGURE_COORDINATES_1.y + GRAPHIC_ABOVE_FIGURE_COORDINATES_1.height + 10,
    width=200,
    height=20,
    page_number=1
)

FAR_AWAY_COORDINATES_1 = LayoutPageCoordinates(
    x=10,
    y=10,
    width=200,
    height=20,
    page_number=10
)

FAR_AWAY_COORDINATES_2 = LayoutPageCoordinates(
    x=10,
    y=10,
    width=200,
    height=20,
    page_number=11
)


TOKEN_1 = 'token1'


@pytest.fixture(name='ocr_model_mock')
def _ocr_model_mock() -> MagicMock:
    return MagicMock(name='ocr_model')


def _get_semantic_content_for_page_coordinates(
    coordinates: LayoutPageCoordinates,
    text: str = 'dummy',
    line_meta: Optional[LayoutLineMeta] = None
) -> SemanticContentWrapper:
    if line_meta is None:
        line_meta = LINE_META_1._replace(
            page_meta=PAGE_META_1._replace(
                page_number=coordinates.page_number
            )
        )

    return SemanticFigure(
        layout_block=LayoutBlock.for_tokens([
            LayoutToken(
                text=text,
                coordinates=coordinates,
                line_meta=line_meta
            )
        ])
    )


def _get_bounding_box_list_distance_sort_key(
    bounding_box_list_1: Sequence[LayoutPageCoordinates],
    bounding_box_list_2: Sequence[LayoutPageCoordinates]
) -> Tuple[float, ...]:
    return get_bounding_box_list_distance(
        bounding_box_list_1, bounding_box_list_2
    )


class TestGetBoundingBoxListDistance:
    def test_should_return_distance_between_vertical_adjacent_bounding_boxes(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1],
            [COORDINATES_1.move_by(dy=COORDINATES_1.height)]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 0
        assert bounding_box_distance.delta_y == 0
        assert bounding_box_distance.euclidean_distance == 0

    def test_should_return_distance_between_horizontal_adjacent_bounding_boxes(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1],
            [COORDINATES_1.move_by(dx=COORDINATES_1.width)]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 0
        assert bounding_box_distance.delta_y == 0
        assert bounding_box_distance.euclidean_distance == 0

    def test_should_return_delta_x_for_bounding_box_left_right(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1],
            [COORDINATES_1.move_by(dx=COORDINATES_1.width + 10)]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 10
        assert bounding_box_distance.delta_y == 0
        assert bounding_box_distance.euclidean_distance == 10

    def test_should_return_delta_x_for_bounding_box_right_left(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1.move_by(dx=COORDINATES_1.width + 10)],
            [COORDINATES_1]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 10
        assert bounding_box_distance.delta_y == 0
        assert bounding_box_distance.euclidean_distance == 10

    def test_should_return_delta_y_for_bounding_box_above_below(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1],
            [COORDINATES_1.move_by(dy=COORDINATES_1.height + 10)]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 0
        assert bounding_box_distance.delta_y == 10
        assert bounding_box_distance.euclidean_distance == 10

    def test_should_return_delta_y_for_bounding_box_below_above(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [COORDINATES_1.move_by(dy=COORDINATES_1.height + 10)],
            [COORDINATES_1]
        )
        assert bounding_box_distance.page_number_diff == 0
        assert bounding_box_distance.delta_x == 0
        assert bounding_box_distance.delta_y == 10
        assert bounding_box_distance.euclidean_distance == 10

    def test_should_calculate_distance_based_on_relative_bounding_boxes(self):
        bounding_box_distance = get_bounding_box_list_distance(
            [LayoutPageCoordinates(
                x=0.1,
                y=0.8,
                width=1.0,
                height=0.1,
                page_number=1
            )],
            [LayoutPageCoordinates(
                x=0.1,
                y=0.1 + 1.0,
                width=1.0,
                height=0.2,
                page_number=2
            )]
        )
        assert bounding_box_distance.page_number_diff == 1
        assert bounding_box_distance.delta_x == 0
        assert round(bounding_box_distance.delta_y, 3) == 0.2
        assert round(bounding_box_distance.euclidean_distance, 3) == 0.2


class TestGetNormalizedBoundingBoxForPageCoordinatesAndPageMeta:
    def test_should_scale_coordinates(self):
        result = get_normalized_bounding_box_for_page_coordinates_and_page_meta(
            coordinates=LayoutPageCoordinates(x=10, y=10, width=20, height=20, page_number=0),
            page_meta=LayoutPageMeta.for_coordinates(
                LayoutPageCoordinates(x=0, y=0, width=100, height=1000, page_number=0)
            )
        )
        LOGGER.debug('result: %r', result)
        assert result == LayoutPageCoordinates(x=0.1, y=0.01, width=0.2, height=0.02, page_number=0)

    def test_should_scale_coordinates_and_adjust_page_number_to_y(self):
        result = get_normalized_bounding_box_for_page_coordinates_and_page_meta(
            coordinates=LayoutPageCoordinates(x=10, y=10, width=20, height=20, page_number=5),
            page_meta=LayoutPageMeta.for_coordinates(
                LayoutPageCoordinates(x=0, y=0, width=100, height=1000, page_number=5)
            )
        )
        LOGGER.debug('result: %r', result)
        assert result == LayoutPageCoordinates(x=0.1, y=5.01, width=0.2, height=0.02, page_number=5)


class TestGetNormalizedBoundingBoxListForLayoutGraphic:
    def test_should_scale_coordinates(self):
        result = get_normalized_bounding_box_list_for_layout_graphic(
            LayoutGraphic(
                coordinates=LayoutPageCoordinates(x=10, y=10, width=20, height=20, page_number=0),
                page_meta=LayoutPageMeta.for_coordinates(
                    LayoutPageCoordinates(x=0, y=0, width=100, height=1000, page_number=0)
                )
            )
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        assert result[0] == LayoutPageCoordinates(
            x=0.1, y=0.01, width=0.2, height=0.02, page_number=0
        )

    def test_should_scale_coordinates_and_adjust_page_number_to_y(self):
        result = get_normalized_bounding_box_list_for_layout_graphic(
            LayoutGraphic(
                coordinates=LayoutPageCoordinates(x=10, y=10, width=20, height=20, page_number=5),
                page_meta=LayoutPageMeta.for_coordinates(
                    LayoutPageCoordinates(x=0, y=0, width=100, height=1000, page_number=5)
                )
            )
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        assert result[0] == LayoutPageCoordinates(
            x=0.1, y=5.01, width=0.2, height=0.02, page_number=5
        )


class TestGetNormalizedBoundingBoxListForLayoutBlock:
    def test_should_scale_coordinates_of_single_line_block(self):
        result = get_normalized_bounding_box_list_for_layout_block(
            LayoutBlock(
                lines=[
                    LayoutLine(
                        tokens=[LayoutToken(
                            TOKEN_1,
                            coordinates=LayoutPageCoordinates(
                                x=10, y=10, width=20, height=20, page_number=5
                            ),
                            line_meta=LayoutLineMeta(
                                line_id=1,
                                page_meta=LayoutPageMeta.for_coordinates(
                                    LayoutPageCoordinates(
                                        x=0, y=0, width=100, height=1000, page_number=5
                                    )
                                )
                            )
                        )],
                    )
                ]
            )
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        assert result[0] == LayoutPageCoordinates(
            x=0.1, y=5.01, width=0.2, height=0.02, page_number=5
        )


class TestBoundingBoxDistance:
    def test_is_better_than_should_return_true_for_none(self):
        assert BoundingBoxDistance(euclidean_distance=1.0).is_better_than(
            None
        ) is True

    def test_is_better_than_should_return_true_for_smaller_distance(self):
        assert BoundingBoxDistance(euclidean_distance=1.0).is_better_than(
            BoundingBoxDistance(euclidean_distance=2.0)
        ) is True

    def test_is_better_than_should_return_false_for_larger_distance(self):
        assert BoundingBoxDistance(euclidean_distance=2.0).is_better_than(
            BoundingBoxDistance(euclidean_distance=1.0)
        ) is False


class TestBoundingBoxDistanceBetween:
    def test_is_better_than_should_return_true_for_none(self):
        assert BoundingBoxDistanceBetween(
            bounding_box_distance=BoundingBoxDistance(euclidean_distance=1.0),
            bounding_box_ref_1=MagicMock(name='bounding_box_ref_1'),
            bounding_box_ref_2=MagicMock(name='bounding_box_ref_2')
        ).is_better_than(None) is True

    def test_is_better_than_should_return_true_for_smaller_distance(self):
        assert BoundingBoxDistanceBetween(
            bounding_box_distance=BoundingBoxDistance(euclidean_distance=1.0),
            bounding_box_ref_1=MagicMock(name='bounding_box_ref_1'),
            bounding_box_ref_2=MagicMock(name='bounding_box_ref_2')
        ).is_better_than(BoundingBoxDistanceBetween(
            bounding_box_distance=BoundingBoxDistance(euclidean_distance=2.0),
            bounding_box_ref_1=MagicMock(name='bounding_box_ref_1'),
            bounding_box_ref_2=MagicMock(name='bounding_box_ref_2')
        )) is True

    def test_is_better_than_should_return_false_for_larger_distance(self):
        assert BoundingBoxDistanceBetween(
            bounding_box_distance=BoundingBoxDistance(euclidean_distance=2.0),
            bounding_box_ref_1=MagicMock(name='bounding_box_ref_1'),
            bounding_box_ref_2=MagicMock(name='bounding_box_ref_2')
        ).is_better_than(BoundingBoxDistanceBetween(
            bounding_box_distance=BoundingBoxDistance(euclidean_distance=1.0),
            bounding_box_ref_1=MagicMock(name='bounding_box_ref_1'),
            bounding_box_ref_2=MagicMock(name='bounding_box_ref_2')
        )) is False


class TestBoundingBoxDistanceGraphicMatcher:
    def test_should_return_empty_list_with_empty_list_of_graphics(self):
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[],
            candidate_semantic_content_list=[SemanticMixedContentWrapper()]
        )
        assert not result

    def test_should_match_graphic_above_semantic_content(self):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=GRAPHIC_ABOVE_FIGURE_COORDINATES_1,
            page_meta=PAGE_META_1
        ))
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=FIGURE_BELOW_GRAPHIC_COORDINATES_1
        )
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                _get_semantic_content_for_page_coordinates(
                    coordinates=FAR_AWAY_COORDINATES_1
                ),
                candidate_semantic_content_1,
                _get_semantic_content_for_page_coordinates(
                    coordinates=FAR_AWAY_COORDINATES_2
                )
            ]
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        first_match = result.graphic_matches[0]
        assert first_match.semantic_graphic == semantic_graphic_1
        assert first_match.candidate_semantic_content == candidate_semantic_content_1

    def test_should_not_match_further_away_graphic_to_same_semantic_content(self):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=GRAPHIC_ABOVE_FIGURE_COORDINATES_1,
            page_meta=PAGE_META_1
        ))
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=FIGURE_BELOW_GRAPHIC_COORDINATES_1
        )
        further_away_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FIGURE_BELOW_GRAPHIC_COORDINATES_1.move_by(dy=500),
            page_meta=PAGE_META_1
        ))
        further_away_graphic_2 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FIGURE_BELOW_GRAPHIC_COORDINATES_1.move_by(dy=1000),
            page_meta=PAGE_META_1
        ))
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[
                further_away_graphic_1,
                semantic_graphic_1,
                further_away_graphic_2
            ],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        first_match = result.graphic_matches[0]
        assert first_match.semantic_graphic == semantic_graphic_1
        assert first_match.candidate_semantic_content == candidate_semantic_content_1

    def test_should_not_match_empty_graphic(self):
        empty_semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=COORDINATES_1._replace(
                width=0, height=0
            )
        ))
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=COORDINATES_1
        )
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[empty_semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert not result

    def test_should_not_match_graphic_on_another_page(self):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=COORDINATES_1._replace(
                page_number=COORDINATES_1.page_number + 1
            ),
            page_meta=PAGE_META_1._replace(
                page_number=PAGE_META_1.page_number + 1
            )
        ))
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=COORDINATES_1
        )
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert not result.graphic_matches
        assert result.unmatched_graphics == [semantic_graphic_1]

    def test_should_match_graphic_at_the_top_of_the_next_page(self):
        page_meta_1 = LayoutPageMeta.for_coordinates(LayoutPageCoordinates(
            x=0, y=0, width=100, height=200, page_number=1
        ))
        assert page_meta_1.coordinates is not None
        page_meta_2 = LayoutPageMeta.for_coordinates(
            page_meta_1.coordinates._replace(page_number=2)
        )
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=LayoutPageCoordinates(
                x=20, y=180, width=60, height=20, page_number=1
            ),
            line_meta=LayoutLineMeta(page_meta=page_meta_1)
        )
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=10, width=60, height=50, page_number=2
            ),
            page_meta=page_meta_2
        ))
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert len(result) == 1
        first_match = result.graphic_matches[0]
        assert first_match.semantic_graphic == semantic_graphic_1
        assert first_match.candidate_semantic_content == candidate_semantic_content_1

    def test_should_match_continuation_graphic_at_the_top_of_the_next_page(self):
        page_meta_1 = LayoutPageMeta.for_coordinates(LayoutPageCoordinates(
            x=0, y=0, width=100, height=200, page_number=1
        ))
        assert page_meta_1.coordinates is not None
        page_meta_2 = LayoutPageMeta.for_coordinates(
            page_meta_1.coordinates._replace(page_number=2)
        )
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=LayoutPageCoordinates(
                x=20, y=110, width=60, height=20, page_number=1
            ),
            line_meta=LayoutLineMeta(page_meta=page_meta_1)
        )
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=140, width=60, height=50, page_number=1
            ),
            page_meta=page_meta_1,
            local_file_path='test-graphic1.png'
        ))
        semantic_graphic_2 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=10, width=60, height=50, page_number=2
            ),
            page_meta=page_meta_2,
            local_file_path='test-graphic2.png'
        ))
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1, semantic_graphic_2],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        LOGGER.debug('result.graphic_matches[].local_file_path: %r', [
            layout_graphic.local_file_path
            for graphic_match in result.graphic_matches
            if (layout_graphic := graphic_match.semantic_graphic.layout_graphic) is not None
        ])
        assert len(result) == 2
        first_match = result.graphic_matches[0]
        assert first_match.semantic_graphic == semantic_graphic_1
        assert first_match.candidate_semantic_content == candidate_semantic_content_1
        second_match = result.graphic_matches[1]
        assert second_match.semantic_graphic == semantic_graphic_2
        assert second_match.candidate_semantic_content == candidate_semantic_content_1
        assert not result.unmatched_graphics

    def test_should_match_continuation_graphic_to_closer_graphic(self):
        page_meta_1 = LayoutPageMeta.for_coordinates(LayoutPageCoordinates(
            x=0, y=0, width=100, height=200, page_number=1
        ))
        assert page_meta_1.coordinates is not None
        page_meta_2 = LayoutPageMeta.for_coordinates(
            page_meta_1.coordinates._replace(page_number=2)
        )
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=LayoutPageCoordinates(
                x=20, y=110, width=60, height=20, page_number=1
            ),
            text='candidate_semantic_content_1',
            line_meta=LayoutLineMeta(page_meta=page_meta_1)
        )
        candidate_semantic_content_2 = _get_semantic_content_for_page_coordinates(
            coordinates=LayoutPageCoordinates(
                x=20, y=100, width=60, height=20, page_number=2
            ),
            text='candidate_semantic_content_2',
            line_meta=LayoutLineMeta(page_meta=page_meta_2)
        )
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=140, width=60, height=50, page_number=1
            ),
            page_meta=page_meta_1,
            local_file_path='test-graphic1.png'
        ))
        semantic_graphic_2 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=10, width=60, height=20, page_number=2
            ),
            page_meta=page_meta_2,
            local_file_path='test-graphic2.png'
        ))
        semantic_graphic_3 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=LayoutPageCoordinates(
                x=20, y=130, width=60, height=20, page_number=2
            ),
            page_meta=page_meta_2,
            local_file_path='test-graphic3.png'
        ))
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1, semantic_graphic_2, semantic_graphic_3],
            candidate_semantic_content_list=[
                candidate_semantic_content_1,
                candidate_semantic_content_2
            ]
        )
        LOGGER.debug('result: %r', result)
        LOGGER.debug('result.graphic_matches[].local_file_path: %r', [
            layout_graphic.local_file_path
            for graphic_match in result.graphic_matches
            if (layout_graphic := graphic_match.semantic_graphic.layout_graphic) is not None
        ])
        assert [
            (graphic_match.candidate_semantic_content.get_text(), graphic_match.semantic_graphic)
            for graphic_match in result.graphic_matches
        ] == [
            ('candidate_semantic_content_1', semantic_graphic_1),
            ('candidate_semantic_content_1', semantic_graphic_2),
            ('candidate_semantic_content_2', semantic_graphic_3)
        ]
        assert not result.unmatched_graphics

    @pytest.mark.parametrize(
        "graphic_type,should_match",
        [("svg", False), ("bitmap", True)]
    )
    def test_should_match_graphic_of_specific_type(
        self,
        graphic_type: str,
        should_match: bool
    ):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=GRAPHIC_ABOVE_FIGURE_COORDINATES_1,
            graphic_type=graphic_type,
            page_meta=PAGE_META_1
        ))
        candidate_semantic_content_1 = _get_semantic_content_for_page_coordinates(
            coordinates=FIGURE_BELOW_GRAPHIC_COORDINATES_1
        )
        result = BoundingBoxDistanceGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        if should_match:
            assert len(result) == 1
            first_match = result.graphic_matches[0]
            assert first_match.semantic_graphic == semantic_graphic_1
        else:
            assert not result.graphic_matches
            assert result.unmatched_graphics == [semantic_graphic_1]


class TestGraphicRelatedBlockTextGraphicMatcher:
    @pytest.mark.parametrize(
        "related_text,figure_label,should_match",
        [
            ("Figure 1", "Figure 1", True),
            ("Figure 1", "Figure 2", False),
            ("Fig 1", "Figure 1", True),
            ("F 1", "Figure 1", False),
            ("Fug 1", "Figure 1", False),
            ("Other\nFigure 1\nMore", "Figure 1", True)
        ]
    )
    def test_should_match_based_on_figure_label(
        self,
        related_text: str,
        figure_label: str,
        should_match: bool
    ):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FAR_AWAY_COORDINATES_1,
            related_block=LayoutBlock.for_text(related_text)
        ))
        candidate_semantic_content_1 = SemanticFigure([
            SemanticLabel(layout_block=LayoutBlock.for_text(figure_label))
        ])
        result = GraphicRelatedBlockTextGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        if should_match:
            assert len(result) == 1
            first_match = result.graphic_matches[0]
            assert first_match.semantic_graphic == semantic_graphic_1
        else:
            assert not result.graphic_matches
            assert result.unmatched_graphics == [semantic_graphic_1]

    def test_should_ignore_layout_graphic_without_related_block(
        self
    ):
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FAR_AWAY_COORDINATES_1,
            related_block=None
        ))
        candidate_semantic_content_1 = SemanticFigure([
            SemanticLabel(layout_block=LayoutBlock.for_text('Figure 1'))
        ])
        result = GraphicRelatedBlockTextGraphicMatcher().get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert not result.graphic_matches
        assert result.unmatched_graphics == [semantic_graphic_1]


class TestOpticalCharacterRecognitionGraphicMatcher:
    @pytest.mark.parametrize(
        "ocr_text,figure_label,should_match",
        [
            ("Figure 1", "Figure 1", True),
            ("Figure 1", "Figure 2", False),
            ("Fig 1", "Figure 1", True),
            ("F 1", "Figure 1", False),
            ("Fug 1", "Figure 1", False),
            ("Other\nFigure 1\nMore", "Figure 1", True)
        ]
    )
    def test_should_match_based_on_figure_label(
        self,
        ocr_model_mock: MagicMock,
        ocr_text: str,
        figure_label: str,
        should_match: bool,
        tmp_path:  Path
    ):
        local_graphic_path = tmp_path / 'image.png'
        PIL.Image.new('RGB', (10, 10), (0, 1, 2)).save(local_graphic_path)
        ocr_model_mock.predict_single.return_value.get_text.return_value = ocr_text
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FAR_AWAY_COORDINATES_1,
            local_file_path=str(local_graphic_path)
        ))
        candidate_semantic_content_1 = SemanticFigure([
            SemanticLabel(layout_block=LayoutBlock.for_text(figure_label))
        ])
        result = OpticalCharacterRecognitionGraphicMatcher(
            ocr_model=ocr_model_mock
        ).get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        if should_match:
            assert len(result) == 1
            first_match = result.graphic_matches[0]
            assert first_match.semantic_graphic == semantic_graphic_1
        else:
            assert not result.graphic_matches
            assert result.unmatched_graphics == [semantic_graphic_1]

    def test_should_ignore_layout_graphic_without_local_path(
        self,
        ocr_model_mock: MagicMock
    ):
        ocr_model_mock.predict_single.return_value.get_text.side_effect = RuntimeError
        semantic_graphic_1 = SemanticGraphic(layout_graphic=LayoutGraphic(
            coordinates=FAR_AWAY_COORDINATES_1,
            local_file_path=None
        ))
        candidate_semantic_content_1 = SemanticFigure([
            SemanticLabel(layout_block=LayoutBlock.for_text('Figure 1'))
        ])
        result = OpticalCharacterRecognitionGraphicMatcher(
            ocr_model=ocr_model_mock
        ).get_graphic_matches(
            semantic_graphic_list=[semantic_graphic_1],
            candidate_semantic_content_list=[
                candidate_semantic_content_1
            ]
        )
        LOGGER.debug('result: %r', result)
        assert not result.graphic_matches
        assert result.unmatched_graphics == [semantic_graphic_1]
