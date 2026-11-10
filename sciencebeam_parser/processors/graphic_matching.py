import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, cast

import PIL.Image

from sciencebeam_parser.utils.image import get_image_with_max_resolution
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutGraphic,
    LayoutPageCoordinates,
    LayoutPageMeta
)
from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticGraphic,
    SemanticLabel,
    SemanticMixedContentWrapper,
    SemanticTable
)
from sciencebeam_parser.ocr_models.ocr_model import OpticalCharacterRecognitionModel
from sciencebeam_parser.processors.ref_matching import SimpleContentIdMatcher


LOGGER = logging.getLogger(__name__)


DEFAULT_MAX_GRAPHIC_DISTANCE = 0.3

DEFAULT_MAX_ENCLOSED_TEXT_RATIO = 0.5

FIGURE_REFERENCE_PATTERN = re.compile(
    r'^\s*(?:figure|fig\.?|table)\s*([0-9]+|[ivxlc]+)\s*[.:]?\s*$',
    re.IGNORECASE
)


def get_referenced_label(text_line: str) -> Optional[str]:
    match = FIGURE_REFERENCE_PATTERN.match(text_line)
    return match.group(1) if match else None


class GraphicMatch(NamedTuple):
    semantic_graphic: SemanticGraphic
    candidate_semantic_content: SemanticContentWrapper

    def __repr__(self) -> str:
        return '%s(semantic_graphic=%r, candidate_semantic_content=%s)' % (
            type(self).__name__,
            self.semantic_graphic.get_short_semantic_content_repr(),
            self.candidate_semantic_content.get_short_semantic_content_repr()
        )


class GraphicMatchResult(NamedTuple):
    graphic_matches: Sequence[GraphicMatch]
    unmatched_graphics: Sequence[SemanticGraphic]

    def __len__(self):
        return len(self.graphic_matches)

    def __iter__(self):
        return iter(self.graphic_matches)


class GraphicMatcher(ABC):
    @abstractmethod
    def get_graphic_matches(
        self,
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper]
    ) -> GraphicMatchResult:
        pass


class ChainedGraphicMatcher(GraphicMatcher):
    def __init__(self, graphic_matchers: Sequence[GraphicMatcher]):
        super().__init__()
        self.graphic_matchers = graphic_matchers

    def get_graphic_matches(
        self,
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper]
    ) -> GraphicMatchResult:
        current_graphic_match_result = GraphicMatchResult(
            graphic_matches=[],
            unmatched_graphics=semantic_graphic_list
        )
        for graphic_matcher in self.graphic_matchers:
            if not current_graphic_match_result.unmatched_graphics:
                break
            temp_graphic_matches = graphic_matcher.get_graphic_matches(
                current_graphic_match_result.unmatched_graphics,
                candidate_semantic_content_list
            )
            if not temp_graphic_matches.graphic_matches:
                continue
            current_graphic_match_result = GraphicMatchResult(
                graphic_matches=(
                    list(current_graphic_match_result.graphic_matches)
                    + list(temp_graphic_matches.graphic_matches)
                ),
                unmatched_graphics=temp_graphic_matches.unmatched_graphics
            )
        return current_graphic_match_result


class BoundingBoxDistance(NamedTuple):
    page_number_diff: int = 0
    delta_x: float = 0
    delta_y: float = 0
    euclidean_distance: float = 0

    def get_sort_key(self):
        return self.euclidean_distance

    def is_better_than(self, other: Optional['BoundingBoxDistance']) -> bool:
        if not other:
            return True
        return self.get_sort_key() < other.get_sort_key()


def get_bounding_box_distance(
    bounding_box_1: LayoutPageCoordinates,
    bounding_box_2: LayoutPageCoordinates
) -> BoundingBoxDistance:
    page_number_diff = abs(
        bounding_box_1.page_number - bounding_box_2.page_number
    )
    delta_x = max(
        0,
        bounding_box_2.x - (bounding_box_1.x + bounding_box_1.width),
        bounding_box_1.x - (bounding_box_2.x + bounding_box_2.width)
    )
    delta_y = max(
        0,
        bounding_box_2.y - (bounding_box_1.y + bounding_box_1.height),
        bounding_box_1.y - (bounding_box_2.y + bounding_box_2.height)
    )
    euclidean_distance = math.sqrt(delta_x**2 + delta_y**2)
    return BoundingBoxDistance(
        page_number_diff=page_number_diff,
        delta_x=delta_x,
        delta_y=delta_y,
        euclidean_distance=euclidean_distance
    )


def get_sorted_bounding_box_distances(
    bounding_box_distance_list: Iterable[BoundingBoxDistance]
) -> List[BoundingBoxDistance]:
    return sorted(
        bounding_box_distance_list,
        key=BoundingBoxDistance.get_sort_key
    )


def get_bounding_box_list_distance(
    bounding_box_list_1: Sequence[LayoutPageCoordinates],
    bounding_box_list_2: Sequence[LayoutPageCoordinates]
) -> BoundingBoxDistance:
    sorted_distances = get_sorted_bounding_box_distances([
        get_bounding_box_distance(bounding_box_1, bounding_box_2)
        for bounding_box_1 in bounding_box_list_1
        for bounding_box_2 in bounding_box_list_2
    ])
    return sorted_distances[0]


class BoundingBoxRef(NamedTuple):
    key: int
    bounding_box_list: Sequence[LayoutPageCoordinates]
    semantic_content: SemanticContentWrapper
    text_bounding_box_list: Sequence[LayoutPageCoordinates] = ()

    def __repr__(self) -> str:
        return '%s(key=%r, bounding_box_list=%r, semantic_content=%s)' % (
            type(self).__name__,
            self.key,
            self.bounding_box_list,
            self.semantic_content.get_short_semantic_content_repr()
        )

    def get_distance_to(self, other: 'BoundingBoxRef') -> BoundingBoxDistance:
        return get_bounding_box_list_distance(
            self.bounding_box_list,
            other.bounding_box_list
        )

    def with_extended_bounding_box_list(
        self,
        bounding_box_list: Sequence[LayoutPageCoordinates]
    ) -> 'BoundingBoxRef':
        return self._replace(  # pylint: disable=no-member
            bounding_box_list=list(self.bounding_box_list) + list(bounding_box_list)
        )


class BoundingBoxDistanceBetween(NamedTuple):
    bounding_box_distance: BoundingBoxDistance
    bounding_box_ref_1: BoundingBoxRef
    bounding_box_ref_2: BoundingBoxRef

    def get_sort_key(self):
        return self.bounding_box_distance.get_sort_key()

    def is_better_than(self, other: Optional['BoundingBoxDistanceBetween']) -> bool:
        if not other:
            return True
        return self.bounding_box_distance.is_better_than(other.bounding_box_distance)


def get_graphic_match_for_distance_between(
    distance_between: BoundingBoxDistanceBetween
) -> GraphicMatch:
    return GraphicMatch(
        semantic_graphic=cast(
            SemanticGraphic,
            distance_between.bounding_box_ref_1.semantic_content
        ),
        candidate_semantic_content=(
            distance_between.bounding_box_ref_2.semantic_content
        )
    )


def get_intersection_area(
    coordinates_1: LayoutPageCoordinates,
    coordinates_2: LayoutPageCoordinates
) -> float:
    if coordinates_1.page_number != coordinates_2.page_number:
        return 0.0
    width = (
        min(coordinates_1.x + coordinates_1.width, coordinates_2.x + coordinates_2.width)
        - max(coordinates_1.x, coordinates_2.x)
    )
    height = (
        min(coordinates_1.y + coordinates_1.height, coordinates_2.y + coordinates_2.height)
        - max(coordinates_1.y, coordinates_2.y)
    )
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def get_enclosed_text_ratio(
    graphic_bounding_box_list: Sequence[LayoutPageCoordinates],
    text_bounding_box_list: Sequence[LayoutPageCoordinates]
) -> float:
    text_area = sum(
        coordinates.width * coordinates.height
        for coordinates in text_bounding_box_list
    )
    if not text_area:
        return 0.0
    enclosed_area = sum(
        get_intersection_area(text_coordinates, graphic_coordinates)
        for text_coordinates in text_bounding_box_list
        for graphic_coordinates in graphic_bounding_box_list
    )
    return min(1.0, enclosed_area / text_area)


def get_normalized_bounding_box_for_page_coordinates_and_page_meta(
    coordinates: LayoutPageCoordinates,
    page_meta: LayoutPageMeta
) -> LayoutPageCoordinates:
    page_coordinates = page_meta.coordinates
    assert page_coordinates is not None
    return LayoutPageCoordinates(
        x=coordinates.x / page_coordinates.width,
        y=(coordinates.y / page_coordinates.height) + page_meta.page_number,
        width=coordinates.width / page_coordinates.width,
        height=coordinates.height / page_coordinates.height,
        page_number=coordinates.page_number
    )


def get_normalized_bounding_box_list_for_layout_graphic(
    layout_graphic: LayoutGraphic
) -> Sequence[LayoutPageCoordinates]:
    if not layout_graphic.coordinates:
        return []
    return [get_normalized_bounding_box_for_page_coordinates_and_page_meta(
        layout_graphic.coordinates,
        page_meta=layout_graphic.page_meta
    )]


def get_normalized_bounding_box_list_for_layout_block(
    layout_block: LayoutBlock
) -> Sequence[LayoutPageCoordinates]:
    page_meta_by_page_number = {
        token.line_meta.page_meta.page_number: token.line_meta.page_meta
        for line in layout_block.lines
        for token in line.tokens
    }
    LOGGER.debug('page_meta_by_page_number: %r', page_meta_by_page_number)
    merged_coordinates_list = layout_block.get_merged_coordinates_list()
    return [
        get_normalized_bounding_box_for_page_coordinates_and_page_meta(
            coordinates=coordinates,
            page_meta=page_meta_by_page_number[coordinates.page_number]
        )
        for coordinates in merged_coordinates_list
    ]


def get_coordinates_sort_key(coordinates: LayoutPageCoordinates):
    return (coordinates.page_number, coordinates.y, coordinates.x)


def get_semantic_content_sort_key(semantic_content: SemanticContentWrapper):
    for token in semantic_content.iter_tokens():
        if token.coordinates:
            return get_coordinates_sort_key(token.coordinates)
    raise RuntimeError('no token with coordinates found')


def get_semantic_graphic_sort_key(semantic_graphic: SemanticGraphic):
    assert semantic_graphic.layout_graphic
    assert semantic_graphic.layout_graphic.coordinates
    return get_coordinates_sort_key(semantic_graphic.layout_graphic.coordinates)


def get_graphic_match_sort_key(graphic_match: GraphicMatch):
    return (
        get_semantic_content_sort_key(graphic_match.candidate_semantic_content),
        get_semantic_graphic_sort_key(graphic_match.semantic_graphic)
    )


def get_sorted_graphic_matches(graphic_matches: Sequence[GraphicMatch]) -> Sequence[GraphicMatch]:
    return sorted(graphic_matches, key=get_graphic_match_sort_key)


def _get_candidate_bounding_box_ref(
    candidate_semantic_content: SemanticContentWrapper
) -> BoundingBoxRef:
    bounding_box_list = get_normalized_bounding_box_list_for_layout_block(
        candidate_semantic_content.merged_block
    )
    return BoundingBoxRef(
        id(candidate_semantic_content),
        bounding_box_list=bounding_box_list,
        semantic_content=candidate_semantic_content,
        text_bounding_box_list=bounding_box_list
    )


class _BoundingBoxDistanceGraphicMatcherInstance(NamedTuple):
    graphic_bounding_box_ref_list: Sequence[BoundingBoxRef]
    candidate_bounding_box_ref_list: Sequence[BoundingBoxRef]
    max_distance: float
    max_enclosed_text_ratio: float = DEFAULT_MAX_ENCLOSED_TEXT_RATIO

    @staticmethod
    def create(
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper],
        ignored_graphic_types: Set[str],
        max_distance: float,
        max_enclosed_text_ratio: float = DEFAULT_MAX_ENCLOSED_TEXT_RATIO
    ) -> '_BoundingBoxDistanceGraphicMatcherInstance':
        graphic_bounding_box_ref_list = [
            BoundingBoxRef(
                id(semantic_graphic),
                bounding_box_list=get_normalized_bounding_box_list_for_layout_graphic(
                    semantic_graphic.layout_graphic
                ),
                semantic_content=semantic_graphic
            )
            for semantic_graphic in semantic_graphic_list
            if (
                semantic_graphic.layout_graphic
                and semantic_graphic.layout_graphic.coordinates
                and semantic_graphic.layout_graphic.graphic_type not in ignored_graphic_types
            )
        ]
        LOGGER.debug('graphic_bounding_box_ref_list: %r', graphic_bounding_box_ref_list)
        candidate_bounding_box_ref_list = [
            _get_candidate_bounding_box_ref(candidate_semantic_content)
            for candidate_semantic_content in candidate_semantic_content_list
        ]
        LOGGER.debug('candidate_bounding_box_ref_list: %r', candidate_bounding_box_ref_list)
        return _BoundingBoxDistanceGraphicMatcherInstance(
            graphic_bounding_box_ref_list=graphic_bounding_box_ref_list,
            candidate_bounding_box_ref_list=candidate_bounding_box_ref_list,
            max_distance=max_distance,
            max_enclosed_text_ratio=max_enclosed_text_ratio
        )

    def is_accept_distance(self, distance_between: BoundingBoxDistanceBetween) -> bool:
        return distance_between.bounding_box_distance.euclidean_distance < self.max_distance

    def is_enclosing_candidate_text(
        self,
        distance_between: BoundingBoxDistanceBetween
    ) -> bool:
        if not isinstance(
            distance_between.bounding_box_ref_2.semantic_content, SemanticTable
        ):
            return False
        enclosed_text_ratio = get_enclosed_text_ratio(
            distance_between.bounding_box_ref_1.bounding_box_list,
            distance_between.bounding_box_ref_2.text_bounding_box_list
        )
        if enclosed_text_ratio < self.max_enclosed_text_ratio:
            return False
        LOGGER.debug(
            'graphic encloses %.2f of the candidate text: %r',
            enclosed_text_ratio, distance_between
        )
        return True

    def with_graphic_bounding_box_ref_list(
        self,
        graphic_bounding_box_ref_list: Sequence[BoundingBoxRef]
    ) -> '_BoundingBoxDistanceGraphicMatcherInstance':
        return self._replace(  # pylint: disable=no-member
            graphic_bounding_box_ref_list=graphic_bounding_box_ref_list
        )

    def with_candidate_bounding_box_ref_list(
        self,
        candidate_bounding_box_ref_list: Sequence[BoundingBoxRef]
    ) -> '_BoundingBoxDistanceGraphicMatcherInstance':
        return self._replace(  # pylint: disable=no-member
            candidate_bounding_box_ref_list=candidate_bounding_box_ref_list
        )

    def get_sorted_distances_between_for_graphic_bounding_box_ref(
        self,
        graphic_bounding_box_ref: BoundingBoxRef
    ) -> Sequence[BoundingBoxDistanceBetween]:
        return sorted(
            [
                BoundingBoxDistanceBetween(
                    bounding_box_distance=graphic_bounding_box_ref.get_distance_to(
                        candidate_bounding_box_ref
                    ),
                    bounding_box_ref_1=graphic_bounding_box_ref,
                    bounding_box_ref_2=candidate_bounding_box_ref
                )
                for candidate_bounding_box_ref in self.candidate_bounding_box_ref_list
            ],
            key=BoundingBoxDistanceBetween.get_sort_key
        )

    def iter_accepted_distances_between_for_graphic_bounding_box_ref(
        self,
        graphic_bounding_box_ref: BoundingBoxRef
    ) -> Sequence[BoundingBoxDistanceBetween]:
        return [
            distance_between
            for distance_between in (
                self.get_sorted_distances_between_for_graphic_bounding_box_ref(
                    graphic_bounding_box_ref
                )
            )
            if not self.is_enclosing_candidate_text(distance_between)
        ]

    def get_best_distance_between_batch(self) -> Sequence[BoundingBoxDistanceBetween]:
        best_distance_between_by_candidate_key: Dict[int, BoundingBoxDistanceBetween] = {}
        for graphic_bounding_box_ref in self.graphic_bounding_box_ref_list:
            sorted_distances_between = (
                self.iter_accepted_distances_between_for_graphic_bounding_box_ref(
                    graphic_bounding_box_ref
                )
            )
            if not sorted_distances_between:
                continue
            best_distance_between = sorted_distances_between[0]
            if not self.is_accept_distance(best_distance_between):
                LOGGER.debug('not accepting distance: %r', best_distance_between)
                continue
            LOGGER.debug('sorted_distances_between: %r', sorted_distances_between)
            candidate_key = best_distance_between.bounding_box_ref_2.key
            previous_best_distance_between = (
                best_distance_between_by_candidate_key.get(candidate_key)
            )
            if not best_distance_between.is_better_than(previous_best_distance_between):
                LOGGER.debug(
                    'found better previous best distance between: %r > %r',
                    previous_best_distance_between,
                    best_distance_between
                )
                continue
            if previous_best_distance_between:
                LOGGER.debug(
                    'found better best distance between: %r < %r',
                    previous_best_distance_between,
                    best_distance_between
                )
            LOGGER.debug('accept candidate: %r -> %r', candidate_key, best_distance_between)
            best_distance_between_by_candidate_key[
                candidate_key
            ] = best_distance_between
        return list(best_distance_between_by_candidate_key.values())

    def iter_remaining_candidate_bounding_box_refs(
        self,
        best_distance_between_batch: Iterable[BoundingBoxDistanceBetween]
    ) -> Iterable[BoundingBoxRef]:
        for best_distance_between in best_distance_between_batch:
            yield best_distance_between.bounding_box_ref_2.with_extended_bounding_box_list(
                best_distance_between.bounding_box_ref_1.bounding_box_list
            )

    def iter_graphic_matches(self) -> Iterable[GraphicMatch]:
        graphic_matches: List[GraphicMatch] = []
        matcher = self
        while matcher.graphic_bounding_box_ref_list:
            best_distance_between_batch = matcher.get_best_distance_between_batch()
            graphic_matches.extend([
                get_graphic_match_for_distance_between(distance_between)
                for distance_between in best_distance_between_batch
            ])
            matched_graphic_keys = {
                distance_between.bounding_box_ref_1.key
                for distance_between in best_distance_between_batch
            }
            LOGGER.debug('matched_graphic_keys: %r', matched_graphic_keys)
            remaining_graphic_bounding_box_ref_list = [
                graphic_bounding_box_ref
                for graphic_bounding_box_ref in matcher.graphic_bounding_box_ref_list
                if graphic_bounding_box_ref.key not in matched_graphic_keys
            ]
            LOGGER.debug(
                'remaining_graphic_bounding_box_ref_list: %r',
                remaining_graphic_bounding_box_ref_list
            )
            if (
                len(remaining_graphic_bounding_box_ref_list)
                == len(matcher.graphic_bounding_box_ref_list)
            ):
                # not matched anything in this round
                break
            remaining_candidate_bounding_box_ref_list = list(
                matcher.iter_remaining_candidate_bounding_box_refs(
                    best_distance_between_batch
                )
            )
            LOGGER.debug(
                'remaining_candidate_bounding_box_ref_list: %r',
                remaining_candidate_bounding_box_ref_list
            )
            matcher = matcher.with_candidate_bounding_box_ref_list(
                remaining_candidate_bounding_box_ref_list
            ).with_graphic_bounding_box_ref_list(remaining_graphic_bounding_box_ref_list)
        return get_sorted_graphic_matches(graphic_matches)


class BoundingBoxDistanceGraphicMatcher(GraphicMatcher):
    def __init__(
        self,
        max_distance: float = DEFAULT_MAX_GRAPHIC_DISTANCE,
        max_enclosed_text_ratio: float = DEFAULT_MAX_ENCLOSED_TEXT_RATIO
    ):
        super().__init__()
        # we ignore svgs for now because they currently represent the whole page
        # rather than individual images
        self.ignored_graphic_types = {'svg'}
        self.max_distance = max_distance
        self.max_enclosed_text_ratio = max_enclosed_text_ratio

    def get_graphic_matches(
        self,
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper]
    ) -> GraphicMatchResult:
        matcher = _BoundingBoxDistanceGraphicMatcherInstance.create(
            semantic_graphic_list=semantic_graphic_list,
            candidate_semantic_content_list=candidate_semantic_content_list,
            ignored_graphic_types=self.ignored_graphic_types,
            max_distance=self.max_distance,
            max_enclosed_text_ratio=self.max_enclosed_text_ratio
        )
        graphic_matches = list(matcher.iter_graphic_matches())
        all_matched_graphic_keys = {
            id(graphic_match.semantic_graphic)
            for graphic_match in graphic_matches
        }
        unmatched_graphics = [
            semantic_graphic
            for semantic_graphic in semantic_graphic_list
            if id(semantic_graphic) not in all_matched_graphic_keys
        ]
        return GraphicMatchResult(
            graphic_matches=graphic_matches,
            unmatched_graphics=unmatched_graphics
        )


class AbstractGraphicTextGraphicMatcher(GraphicMatcher):
    @abstractmethod
    def get_text_for_semantic_graphic(self, semantic_graphic: SemanticGraphic) -> str:
        pass

    def get_text_for_candidate_semantic_content(
        self,
        semantic_content: SemanticContentWrapper
    ) -> str:
        assert isinstance(semantic_content, SemanticMixedContentWrapper)
        return semantic_content.get_text_by_type(SemanticLabel)

    def get_graphic_matches(
        self,
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper]
    ) -> GraphicMatchResult:
        graphic_text_list = [
            self.get_text_for_semantic_graphic(semantic_graphic)
            for semantic_graphic in semantic_graphic_list
        ]
        LOGGER.debug('graphic_text_list: %r', graphic_text_list)
        candidate_label_text_list = [
            self.get_text_for_candidate_semantic_content(candidate_semantic_content)
            for candidate_semantic_content in candidate_semantic_content_list
        ]
        LOGGER.debug('candidate_label_text_list: %r', candidate_label_text_list)
        content_id_matcher = SimpleContentIdMatcher(
            {str(index): value for index, value in enumerate(candidate_label_text_list)},
            prefix_length=3
        )
        graphic_matches: List[GraphicMatch] = []
        unmatched_graphics: List[SemanticGraphic] = []
        for graphic_index, graphic_text in enumerate(graphic_text_list):
            semantic_graphic = semantic_graphic_list[graphic_index]
            candidate_id: Optional[str] = None
            for graphic_text_line in graphic_text.splitlines():
                if not graphic_text_line:
                    continue
                candidate_id = content_id_matcher.get_id_by_text(graphic_text_line)
                if candidate_id is not None:
                    break
            LOGGER.debug('get_id_by_text: candidate_id=%r (text=%r)', candidate_id, graphic_text)
            if candidate_id is None:
                unmatched_graphics.append(semantic_graphic)
                continue
            candidate_semantic_content = candidate_semantic_content_list[int(candidate_id)]
            graphic_matches.append(GraphicMatch(
                semantic_graphic=semantic_graphic,
                candidate_semantic_content=candidate_semantic_content
            ))
        return GraphicMatchResult(
            graphic_matches=graphic_matches,
            unmatched_graphics=unmatched_graphics
        )


class GraphicRelatedBlockTextGraphicMatcher(AbstractGraphicTextGraphicMatcher):
    """Matches on a label the graphic itself names, such as a figure page's own heading.

    Only a line naming a label counts: a page's remaining text is axis ticks and
    legend entries, which match a label of their own by accident.
    """

    def get_text_for_semantic_graphic(self, semantic_graphic: SemanticGraphic) -> str:
        assert semantic_graphic.layout_graphic
        related_block = semantic_graphic.layout_graphic.related_block
        if not related_block:
            return ''
        text_lines: List[str] = []
        for line in related_block.lines:
            for text_line in line.text.splitlines():
                text_line = text_line.strip()
                referenced_label = get_referenced_label(text_line)
                if not referenced_label:
                    continue
                text_lines.append(text_line)
                text_lines.append(referenced_label)
        return '\n'.join(text_lines)


DEFAULT_OCR_MAX_RESOLUTION = 1024


class OpticalCharacterRecognitionGraphicMatcher(AbstractGraphicTextGraphicMatcher):
    def __init__(
        self,
        ocr_model: OpticalCharacterRecognitionModel,
        max_resolution: int = DEFAULT_OCR_MAX_RESOLUTION
    ):
        super().__init__()
        self.ocr_model = ocr_model
        self.max_resolution = max_resolution

    def get_text_for_semantic_graphic(self, semantic_graphic: SemanticGraphic) -> str:
        assert semantic_graphic.layout_graphic
        if semantic_graphic.layout_graphic.graphic_type == 'svg':
            return ''
        if not semantic_graphic.layout_graphic.local_file_path:
            # images generated by CV object detection may not have a local file path
            # if we don't output assets
            LOGGER.info(
                'no local image found for layout graphic: %r',
                semantic_graphic.layout_graphic
            )
            return ''
        with PIL.Image.open(semantic_graphic.layout_graphic.local_file_path) as image:
            return self.ocr_model.predict_single(
                get_image_with_max_resolution(image, self.max_resolution)
            ).get_text()
