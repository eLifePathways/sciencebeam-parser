from typing import NamedTuple, Set

from sciencebeam_parser.config.config import AppConfig

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
    extract_graphic_bounding_boxes: bool = True
    extract_graphic_assets: bool = False
    use_cv_model: bool = False
    cv_render_dpi: float = DEFAULT_PDF_RENDER_DPI
    use_ocr_model: bool = False
    replace_text_by_cv_graphic: bool = False
    max_graphic_distance: float = DEFAULT_MAX_GRAPHIC_DISTANCE
    noise_filter_enabled: bool = False
    noise_filter_repetition_fraction: float = 0.5
    noise_filter_preserve_first_page_head: bool = False
    noise_filter_preserve_first_page_foot: bool = False

    @staticmethod
    def from_app_config(app_config: AppConfig) -> 'FullTextProcessorConfig':
        return FullTextProcessorConfig()._replace(
            **app_config.get('processors', {}).get('fulltext', {})
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
