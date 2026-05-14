import logging
import re
from typing import Counter, Iterable, List, Optional, Set

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.utils.bounding_box import BoundingBox
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
# https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/features/FeatureFactory.java#L359-L367
def get_text_pattern(text: str) -> str:
    # Note: original code is meant to shadow numbers but are actually removed
    return re.sub(r'[^a-zA-Z ]', '', text).lower()


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
        self.is_repetitive_pattern: bool = False
        self.is_first_repetitive_pattern: bool = False
        self.page_main_area_bounding_box: Optional[BoundingBox] = None
        self.block_bounding_box: Optional[BoundingBox] = None

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
        segmentation_line_features.document_token_count = sum(
            len(line.tokens)
            for block in layout_document.iter_all_blocks()
            for line in block.lines
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
            if len(value) >= 8  # Java GROBID sometimes counts an additional trailing space
        }
        pattern_counter = Counter(pattern_by_line_id.values())
        LOGGER.debug('pattern_counter: %s', pattern_counter)
        seen_repetitive_patterns: Set[str] = set()
        document_token_index = 0
        for page in layout_document.pages:
            blocks = page.blocks
            segmentation_line_features.page_blocks = blocks
            segmentation_line_features.page_main_area_bounding_box = (
                page.meta.coordinates.bounding_box
                if page.meta.coordinates
                else None
            )
            for block_index, block in enumerate(blocks):
                block_coords_list = block.get_merged_coordinates_list()
                segmentation_line_features.block_bounding_box = (
                    block_coords_list[0].bounding_box if block_coords_list else None
                )
                segmentation_line_features.page_block_index = block_index
                block_lines = block.lines
                segmentation_line_features.block_lines = block_lines
                block_line_texts = [line.text for line in block_lines]
                max_block_line_text_length = max(len(text) for text in block_line_texts)
                first_block_token = next(iter(block.iter_all_tokens()), None)
                assert first_block_token
                for line_index, line in enumerate(block_lines):
                    segmentation_line_features.document_token_index = document_token_index
                    document_token_index += len(line.tokens)
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
            FeatureDef('is_proper_name', lambda f: f.get_dummy_str_is_proper_name()),
            FeatureDef('is_common_name', lambda f: f.get_dummy_str_is_common_name()),
            FeatureDef('is_first_name', lambda f: f.get_dummy_str_is_first_name()),
            FeatureDef('is_year', lambda f: f.get_dummy_str_is_year()),
            FeatureDef('is_month', lambda f: f.get_dummy_str_is_month()),
            FeatureDef('is_email', lambda f: f.get_dummy_str_is_email()),
            FeatureDef('is_http', lambda f: f.get_dummy_str_is_http()),
            FeatureDef('relative_document_position',
                       lambda f: f.get_str_relative_document_position()),
            FeatureDef('relative_page_position',
                       lambda f: f.get_dummy_str_relative_page_position()),
            FeatureDef('punctuation_profile', lambda f: f.get_line_punctuation_profile()),
            FeatureDef('punctuation_profile_length',
                       lambda f: f.get_line_punctuation_profile_length_feature()),
            FeatureDef('block_relative_line_length',
                       lambda f: f.get_str_block_relative_line_length_feature()),
            FeatureDef('is_bitmap_around', lambda f: f.get_dummy_str_is_bitmap_around()),
            FeatureDef('is_vector_around', lambda f: f.get_dummy_str_is_vector_around()),
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
