import logging
import re
from typing import Counter, Dict, Iterable, List, Optional, Set

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutGraphic,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.utils.tokenizer import get_tokenized_tokens
from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    DocumentFeaturesContext,
    FeatureDef,
    ModelDataGenerator,
    LayoutModelData,
    feature_linear_scaling_int,
    _LINESCALE,
    get_str_bool_feature_value
)


LOGGER = logging.getLogger(__name__)


NBSP = '\u00A0'


def format_feature_text(text: str) -> str:
    return re.sub(" |\t", NBSP, text.strip())


NBBINS_POSITION = 12

EMPTY_LAYOUT_TOKEN = LayoutToken('')
EMPTY_LAYOUT_LINE = LayoutLine([])

# Matches GROBID's Document.MIN_DISTANCE (coordinate units from pdfalto ALTO output)
_GRAPHIC_MIN_DISTANCE = 100.0

VECTOR_GRAPHIC_TYPE = 'svg'


def _is_graphic_connected_to_block(
    graphic: LayoutGraphic,
    block_y: float,
    block_height: float
) -> bool:
    if not graphic.coordinates:
        return False
    gc = graphic.coordinates
    return (
        abs((gc.y + gc.height) - block_y) < _GRAPHIC_MIN_DISTANCE
        or abs(gc.y - (block_y + block_height)) < _GRAPHIC_MIN_DISTANCE
    )


def get_block_status(line_index: int, line_count: int) -> str:
    return (
        'BLOCKSTART' if line_index == 0
        else (
            'BLOCKEND'
            if line_index == line_count - 1
            else 'BLOCKIN'
        )
    )


def get_page_status(
    block_index: int, block_count: int,
    is_first_block_token: bool,
    is_last_block_token: bool
) -> str:
    return (
        'PAGESTART' if block_index == 0 and is_first_block_token
        else (
            'PAGEEND'
            if block_index == block_count - 1 and is_last_block_token
            else 'PAGEIN'
        )
    )


# based on:
# https://github.com/kermitt2/grobid/blob/0.9.0/grobid-core/src/main/java/org/grobid/core/document/Document.java#L463-L537
def calculate_page_main_areas(  # pylint: disable=too-many-locals,too-many-branches
    layout_document: LayoutDocument
) -> Dict[int, Optional[BoundingBox]]:
    left_even: Counter[int] = Counter()
    right_even: Counter[int] = Counter()
    left_odd: Counter[int] = Counter()
    right_odd: Counter[int] = Counter()
    top_counter: Counter[int] = Counter()
    bottom_counter: Counter[int] = Counter()

    for page in layout_document.pages:
        page_number = page.meta.page_number
        for block in page.blocks:
            coords_list = block.get_merged_coordinates_list()
            if not coords_list:
                continue
            coords = coords_list[0]
            # Filter small blocks (page numbers, journal headers, etc.)
            if (coords.x == 0 or coords.height < 20
                    or coords.width < 20 or coords.height * coords.width < 3000):
                continue
            x, y, width, height = coords.x, coords.y, coords.width, coords.height
            if page_number % 2 == 0:
                left_even[int(x)] += 1
                right_even[int(x + width)] += 1
            else:
                left_odd[int(x)] += 1
                right_odd[int(x + width)] += 1
            top_counter[int(y)] += 1
            bottom_counter[int(y + height)] += 1

    result: Dict[int, Optional[BoundingBox]] = {}

    if left_even and left_odd:
        page_y = min(top_counter)
        page_height = max(bottom_counter) - page_y + 1
        page_odd_x = min(left_odd)
        page_odd_width = max(right_odd) - page_odd_x + 1
        if len(layout_document.pages) > 1:
            page_even_x = min(left_even)
            page_even_width = max(right_even) - page_even_x + 1
        else:
            page_even_x = 0
            page_even_width = 0
        for page in layout_document.pages:
            page_number = page.meta.page_number
            if page_number % 2 == 0 and len(layout_document.pages) > 1:
                result[page_number] = BoundingBox(
                    page_even_x, page_y, page_even_width, page_height
                )
            else:
                result[page_number] = BoundingBox(
                    page_odd_x, page_y, page_odd_width, page_height
                )
    else:
        # Fallback: use full page dimensions when not enough data for both even/odd
        for page in layout_document.pages:
            page_number = page.meta.page_number
            if page.meta.coordinates:
                coords = page.meta.coordinates
                result[page_number] = BoundingBox(0, 0, coords.width, coords.height)
            else:
                result[page_number] = None

    return result


# based on:
# https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/features/FeatureFactory.java#L359-L367
def get_text_pattern(text: str) -> str:
    # Matches GROBID's FeatureFactory.getPattern: remove everything except letters, then lowercase
    return re.sub(r'[^a-zA-Z]', '', text).lower()


def count_block_tokens_like_grobid(block_lines: List[LayoutLine]) -> int:
    # Approximate Grobid's block.getTokens().size():
    # - each ALTO String is sub-tokenized via GrobidAnalyzer (same delimiters as our tokenizer)
    # - endTextLine appends a '\n' LayoutToken per line
    # - endTextBlock appends a '\n' LayoutToken per block
    content_count = sum(
        len(get_tokenized_tokens(token.text))
        for line in block_lines
        for token in line.tokens
    )
    newline_count = len(block_lines) + 1  # one per TextLine + one per TextBlock
    return content_count + newline_count


class SegmentationLineFeatures(ContextAwareLayoutTokenFeatures):
    def __init__(
        self,
        document_features_context: DocumentFeaturesContext,
        layout_token: LayoutToken = EMPTY_LAYOUT_TOKEN
    ):
        super().__init__(
            layout_token,
            document_features_context=document_features_context,
            layout_line=EMPTY_LAYOUT_LINE
        )
        self.line_text = ''
        self.second_token_text = ''
        self.page_blocks: List[LayoutBlock] = []
        self.page_block_index: int = 0
        self.block_lines: List[LayoutLine] = []
        self.block_line_index: int = 0
        self.previous_layout_token: Optional[LayoutToken] = None
        self.max_block_line_text_length = 0
        self.document_token_count = 0
        self.document_token_index = 0
        self.page_token_count: int = 0
        self.page_token_index: int = 0
        self.is_repetitive_pattern: bool = False
        self.is_first_repetitive_pattern: bool = False
        self.page_main_area_bounding_box: Optional[BoundingBox] = None
        self.block_bounding_box: Optional[BoundingBox] = None
        self.is_vector_around: bool = False
        self.is_bitmap_around: bool = False

    def get_block_status(self) -> str:
        return get_block_status(self.block_line_index, len(self.block_lines))

    def get_page_status(self) -> str:
        return get_page_status(
            self.page_block_index, len(self.page_blocks),
            is_first_block_token=self.block_line_index == 0,
            is_last_block_token=self.block_line_index == len(self.block_lines) - 1
        )

    def get_formatted_whole_line_feature(self) -> str:
        return format_feature_text(self.line_text)

    def get_str_is_repetitive_pattern(self) -> str:
        return get_str_bool_feature_value(self.is_repetitive_pattern)

    def get_str_is_first_repetitive_pattern(self) -> str:
        return get_str_bool_feature_value(self.is_first_repetitive_pattern)

    def get_dummy_str_is_repetitive_pattern(self) -> str:
        return '0'

    def get_dummy_str_is_first_repetitive_pattern(self) -> str:
        return '0'

    def get_str_is_main_area(self) -> str:
        if self.page_main_area_bounding_box is None:
            return get_str_bool_feature_value(False)
        if self.block_bounding_box is None:
            return get_str_bool_feature_value(True)
        return get_str_bool_feature_value(
            bool(self.page_main_area_bounding_box.intersection(self.block_bounding_box))
        )

    def get_str_is_vector_around(self) -> str:
        return get_str_bool_feature_value(self.is_vector_around)

    def get_str_is_bitmap_around(self) -> str:
        return get_str_bool_feature_value(self.is_bitmap_around)

    def get_str_block_relative_line_length_feature(self) -> str:
        return str(feature_linear_scaling_int(
            len(self.line_text),
            self.max_block_line_text_length,
            _LINESCALE
        ))

    def get_str_relative_document_position(self) -> str:
        return str(feature_linear_scaling_int(
            self.document_token_index,
            self.document_token_count,
            NBBINS_POSITION
        ))

    def get_str_relative_page_position(self) -> str:
        # Matches Grobid's relativePagePositionChar: tokens before current block / total page tokens
        return str(feature_linear_scaling_int(
            self.page_token_index,
            self.page_token_count,
            NBBINS_POSITION
        ))


class SegmentationLineFeaturesProvider:
    def __init__(
        self,
        document_features_context: DocumentFeaturesContext,
        use_first_token_of_block: bool
    ):
        self.document_features_context = document_features_context
        self.use_first_token_of_block = use_first_token_of_block

    def iter_line_features(  # pylint: disable=too-many-locals,too-many-statements
        self,
        layout_document: LayoutDocument
    ) -> Iterable[SegmentationLineFeatures]:
        segmentation_line_features = SegmentationLineFeatures(
            document_features_context=self.document_features_context
        )
        previous_token: Optional[LayoutToken] = None
        # Matches GROBID: documentLenghtChar initialised to -1, then += block.getTokens().size()
        # for each block. block.getTokens().size() ≈ count_block_tokens_like_grobid.
        segmentation_line_features.document_token_count = -1 + sum(
            count_block_tokens_like_grobid(block.lines)
            for block in layout_document.iter_all_blocks()
        )
        pattern_candididate_block_iterable = (
            block
            for page in layout_document.pages
            for block_index, block in enumerate(page.blocks)
            if block_index < 2 or block_index > len(page.blocks) - 2
        )
        pattern_candididate_line_iterable = (
            block.lines[0]
            for block in pattern_candididate_block_iterable
            if block.lines and block.lines[0].tokens
        )
        all_pattern_by_line_id = {
            id(line): get_text_pattern(line.text)
            for line in pattern_candididate_line_iterable
        }
        LOGGER.debug('all_pattern_by_line_id: %s', all_pattern_by_line_id)
        pattern_by_line_id = {
            key: value
            for key, value in all_pattern_by_line_id.items()
            if len(value) > 8  # matches GROBID: pattern.length() > 8
        }
        pattern_counter = Counter(pattern_by_line_id.values())
        LOGGER.debug('pattern_counter: %s', pattern_counter)
        seen_repetitive_patterns: Set[str] = set()
        document_token_index = 0
        page_main_areas = calculate_page_main_areas(layout_document)
        for page in layout_document.pages:
            blocks = page.blocks
            segmentation_line_features.page_blocks = blocks
            segmentation_line_features.page_main_area_bounding_box = (
                page_main_areas.get(page.meta.page_number)
            )
            segmentation_line_features.page_token_count = sum(
                count_block_tokens_like_grobid(block.lines)
                for block in blocks
            )
            page_token_index = 0
            for block_index, block in enumerate(blocks):
                block_coords_list = block.get_merged_coordinates_list()
                block_coords = block_coords_list[0] if block_coords_list else None
                segmentation_line_features.block_bounding_box = (
                    block_coords.bounding_box if block_coords else None
                )
                block_y = block_coords.y if block_coords else 0.0
                block_height = block_coords.height if block_coords else 0.0
                is_vector_around = False
                is_bitmap_around = False
                for graphic in page.graphics:
                    if _is_graphic_connected_to_block(graphic, block_y, block_height):
                        if graphic.graphic_type == VECTOR_GRAPHIC_TYPE:
                            is_vector_around = True
                        else:
                            is_bitmap_around = True
                segmentation_line_features.is_vector_around = is_vector_around
                segmentation_line_features.is_bitmap_around = is_bitmap_around
                segmentation_line_features.page_block_index = block_index
                segmentation_line_features.page_token_index = page_token_index
                # Matches GROBID: nn is set once per block (before the line loop) so all lines
                # in a block share the same relativeDocumentPosition.
                segmentation_line_features.document_token_index = document_token_index
                block_lines = block.lines
                segmentation_line_features.block_lines = block_lines
                block_line_texts = [line.text for line in block_lines]
                max_block_line_text_length = max(len(text) for text in block_line_texts)
                first_block_token = next(iter(block.iter_all_tokens()), None)
                assert first_block_token
                for line_index, line in enumerate(block_lines):
                    segmentation_line_features.layout_line = line
                    segmentation_line_features.block_line_index = line_index
                    segmentation_line_features.max_block_line_text_length = (
                        max_block_line_text_length
                    )
                    line_text = block_line_texts[line_index]
                    retokenized_token_texts = re.split(r" |\t|\f|\u00A0", line_text)
                    if not retokenized_token_texts:
                        continue
                    if self.use_first_token_of_block:
                        # Java GROBID uses the first token in the block
                        token = first_block_token
                    else:
                        token = line.tokens[0]
                    segmentation_line_features.layout_token = token
                    segmentation_line_features.line_text = line_text
                    segmentation_line_features.concatenated_line_tokens_text = line_text
                    segmentation_line_features.token_text = retokenized_token_texts[0].strip()
                    segmentation_line_features.second_token_text = (
                        retokenized_token_texts[1] if len(retokenized_token_texts) >= 2 else ''
                    )
                    segmentation_line_features.previous_layout_token = previous_token
                    line_pattern = pattern_by_line_id.get(id(line), '')
                    LOGGER.debug('line_pattern: %r', line_pattern)
                    segmentation_line_features.is_repetitive_pattern = (
                        pattern_counter[line_pattern] > 1
                    )
                    segmentation_line_features.is_first_repetitive_pattern = (
                        segmentation_line_features.is_repetitive_pattern
                        and line_pattern not in seen_repetitive_patterns
                    )
                    if segmentation_line_features.is_first_repetitive_pattern:
                        seen_repetitive_patterns.add(line_pattern)
                    yield segmentation_line_features
                    previous_token = token
                document_token_index += count_block_tokens_like_grobid(block_lines)
                page_token_index += count_block_tokens_like_grobid(block_lines)


class SegmentationDataGenerator(ModelDataGenerator):
    def __init__(
        self,
        document_features_context: DocumentFeaturesContext,
        use_first_token_of_block: bool
    ):
        self.document_features_context = document_features_context
        self.use_first_token_of_block = use_first_token_of_block
        self._feature_defs: List[FeatureDef[SegmentationLineFeatures]] = [
            FeatureDef('token_text', lambda f: f.token_text),
            FeatureDef('second_token_text', lambda f: f.second_token_text or f.token_text),
            FeatureDef('lower_token_text', lambda f: f.get_lower_token_text()),
            FeatureDef('prefix_1', lambda f: f.get_prefix(1)),
            FeatureDef('prefix_2', lambda f: f.get_prefix(2)),
            FeatureDef('prefix_3', lambda f: f.get_prefix(3)),
            FeatureDef('prefix_4', lambda f: f.get_prefix(4)),
            FeatureDef('block_status', lambda f: f.get_block_status()),
            FeatureDef('page_status', lambda f: f.get_page_status()),
            FeatureDef('token_font_status', lambda f: f.get_token_font_status()),
            FeatureDef('token_font_size', lambda f: f.get_token_font_size_feature()),
            FeatureDef('is_bold', lambda f: f.get_str_is_bold()),
            FeatureDef('is_italic', lambda f: f.get_str_is_italic()),
            FeatureDef('capitalisation', lambda f: f.get_capitalisation_status_using_allcap()),
            FeatureDef('digit_status', lambda f: f.get_digit_status_using_containsdigits()),
            FeatureDef('is_single_char', lambda f: f.get_str_is_single_char()),
            FeatureDef('is_proper_name', lambda f: f.get_str_is_proper_name()),
            FeatureDef('is_common_name', lambda f: f.get_str_is_common_name()),
            FeatureDef('is_first_name', lambda f: f.get_dummy_str_is_first_name()),
            FeatureDef('is_year', lambda f: f.get_str_is_year()),
            FeatureDef('is_month', lambda f: f.get_str_is_month()),
            FeatureDef('is_email', lambda f: f.get_dummy_str_is_email()),
            FeatureDef('is_http', lambda f: f.get_str_is_http()),
            FeatureDef('relative_document_position',
                       lambda f: f.get_str_relative_document_position()),
            FeatureDef('relative_page_position',
                       lambda f: f.get_str_relative_page_position()),
            FeatureDef('punctuation_profile', lambda f: f.get_line_punctuation_profile()),
            FeatureDef('punctuation_profile_length',
                       lambda f: f.get_line_punctuation_profile_length_feature()),
            FeatureDef('block_relative_line_length',
                       lambda f: f.get_str_block_relative_line_length_feature()),
            FeatureDef('is_bitmap_around', lambda f: f.get_str_is_bitmap_around()),
            FeatureDef('is_vector_around', lambda f: f.get_str_is_vector_around()),
            FeatureDef('is_repetitive_pattern', lambda f: f.get_str_is_repetitive_pattern()),
            FeatureDef('is_first_repetitive_pattern',
                       lambda f: f.get_str_is_first_repetitive_pattern()),
            FeatureDef('is_main_area', lambda f: f.get_str_is_main_area()),
            FeatureDef('whole_line_text', lambda f: f.get_formatted_whole_line_feature()),
        ]

    @property
    def feature_names(self) -> List[str]:
        return [fd.name for fd in self._feature_defs]

    def iter_model_data_for_layout_document(
        self,
        layout_document: LayoutDocument
    ) -> Iterable[LayoutModelData]:
        features_provider = SegmentationLineFeaturesProvider(
            document_features_context=self.document_features_context,
            use_first_token_of_block=self.use_first_token_of_block
        )
        for features in features_provider.iter_line_features(layout_document):
            yield LayoutModelData(
                layout_line=features.layout_line,
                data_line=' '.join(fd.extract(features) for fd in self._feature_defs)
            )
