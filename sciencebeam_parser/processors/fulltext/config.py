from typing import NamedTuple, Set

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_noise_filter import (
    NOISE_ACTION_DROP,
    LayoutNoiseFilterConfig,
)

from sciencebeam_parser.processors.document_page_image import (
    DEFAULT_PDF_RENDER_DPI
)
from sciencebeam_parser.processors.graphic_matching import DEFAULT_MAX_GRAPHIC_DISTANCE


class RequestFieldNames:
    """
    "Abstract" field names that should be independent from the model architecture.
    """
    TITLE = 'title'
    ABSTRACT = 'abstract'
    AUTHORS = 'authors'
    AFFILIATIONS = 'affiliations'
    REFERENCES = 'references'


FRONT_FIELDS = {
    RequestFieldNames.TITLE,
    RequestFieldNames.ABSTRACT,
    RequestFieldNames.AUTHORS,
    RequestFieldNames.AFFILIATIONS
}


class FullTextProcessorConfig(NamedTuple):
    extract_front: bool = True
    extract_authors: bool = True
    extract_affiliations: bool = True
    extract_body_sections: bool = True
    extract_acknowledgements: bool = True
    extract_back_sections: bool = True
    extract_references: bool = True
    extract_citation_fields: bool = True
    extract_citation_authors: bool = True
    extract_citation_editors: bool = False
    extract_figure_fields: bool = True
    extract_table_fields: bool = True
    merge_raw_authors: bool = False
    deduplicate_raw_authors: bool = True
    deduplicate_raw_authors_use_initial_fallback: bool = True
    extract_graphic_bounding_boxes: bool = True
    extract_graphic_assets: bool = False
    use_cv_model: bool = False
    cv_render_dpi: float = DEFAULT_PDF_RENDER_DPI
    use_ocr_model: bool = False
    replace_text_by_cv_graphic: bool = False
    max_graphic_distance: float = DEFAULT_MAX_GRAPHIC_DISTANCE
    noise_filter_enabled: bool = False
    noise_filter_repetition_fraction: float = 0.5
    noise_filter_position_consistency_fraction: float = 0.8
    noise_filter_max_position_stddev: float = 0.05
    noise_filter_max_height_ratio: float = 2.0
    noise_filter_preserve_first_page_head: bool = False
    noise_filter_preserve_first_page_foot: bool = False
    noise_filter_outside_main_area: bool = False
    noise_filter_min_repeating_pattern_length: int = 3
    noise_filter_max_letterless_length: int = 12
    noise_filter_action: str = NOISE_ACTION_DROP

    @staticmethod
    def from_app_config(app_config: AppConfig) -> 'FullTextProcessorConfig':
        return FullTextProcessorConfig()._replace(
            **app_config.get('processors', {}).get('fulltext', {})
        )

    def get_layout_noise_filter_config(self) -> LayoutNoiseFilterConfig:
        return LayoutNoiseFilterConfig(
            enabled=self.noise_filter_enabled,
            repetition_fraction=self.noise_filter_repetition_fraction,
            position_consistency_fraction=self.noise_filter_position_consistency_fraction,
            max_position_stddev=self.noise_filter_max_position_stddev,
            max_height_ratio=self.noise_filter_max_height_ratio,
            preserve_first_page_head=self.noise_filter_preserve_first_page_head,
            preserve_first_page_foot=self.noise_filter_preserve_first_page_foot,
            filter_outside_main_area=self.noise_filter_outside_main_area,
            min_repeating_pattern_length=self.noise_filter_min_repeating_pattern_length,
            max_letterless_length=self.noise_filter_max_letterless_length,
            action=self.noise_filter_action,
        )

    def get_for_requested_field_names(
        self,
        request_field_names: Set[str]
    ) -> 'FullTextProcessorConfig':
        if not request_field_names:
            return self
        remaining_field_names = request_field_names - FRONT_FIELDS - {RequestFieldNames.REFERENCES}
        if remaining_field_names:
            return self
        extract_front = bool(FRONT_FIELDS & request_field_names)
        extract_authors = RequestFieldNames.AUTHORS in request_field_names
        extract_affiliations = RequestFieldNames.AFFILIATIONS in request_field_names
        extract_references = RequestFieldNames.REFERENCES in request_field_names
        return self._replace(  # pylint: disable=no-member
            extract_front=extract_front,
            extract_authors=extract_authors,
            extract_affiliations=extract_affiliations,
            extract_body_sections=False,
            extract_acknowledgements=False,
            extract_back_sections=False,
            extract_references=extract_references,
            extract_graphic_bounding_boxes=False
        )

    def get_for_header_document(self) -> 'FullTextProcessorConfig':
        return self.get_for_requested_field_names(FRONT_FIELDS)
