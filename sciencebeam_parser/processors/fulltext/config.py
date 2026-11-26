import dataclasses
from typing import Mapping, NamedTuple, Sequence, Set, Tuple

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.dehyphenation import (
    DEFAULT_DEHYPHENATE_MODE,
    DEHYPHENATE_MODES,
    DehyphenateMode
)
from sciencebeam_parser.document.layout_noise_filter import LayoutNoiseFilterConfig
from sciencebeam_parser.processors.graphic_regions import GraphicRegionConfig

from sciencebeam_parser.processors.document_page_image import (
    DEFAULT_PDF_RENDER_DPI
)
from sciencebeam_parser.models.header.extract import (
    ABSTRACTS_MODES,
    DEFAULT_ABSTRACTS_MODE,
    DEFAULT_TITLES_MODE,
    TITLES_MODES
)
from sciencebeam_parser.models.citation.extract import (
    DEFAULT_FRAGMENT_LINK_DOI,
    DEFAULT_REFERENCE_DOIS_MODE,
    DEFAULT_REFERENCE_KEPT_BY_ANY_OF,
    DEFAULT_REFERENCE_TITLES_MODE,
    REFERENCE_DOIS_MODES,
    REFERENCE_TITLES_MODES,
    validate_fragment_link_doi,
    validate_reference_field_names
)
from sciencebeam_parser.processors.graphic_matching import (
    DEFAULT_MAX_ENCLOSED_TEXT_RATIO,
    DEFAULT_MAX_GRAPHIC_DISTANCE
)


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


@dataclasses.dataclass(frozen=True)
class ExtractConfig:
    front: bool = True
    front_authors: bool = True
    front_affiliations: bool = True
    body_sections: bool = True
    acknowledgements: bool = True
    back_sections: bool = True
    references: bool = True
    reference_fields: bool = True
    reference_authors: bool = True
    reference_editors: bool = False
    figure_fields: bool = True
    table_fields: bool = True
    graphics: bool = True
    graphic_assets: bool = False


@dataclasses.dataclass(frozen=True)
class FrontConfig:
    select_abstracts: str = DEFAULT_ABSTRACTS_MODE
    join_abstract_continuations: bool = True
    select_titles: str = DEFAULT_TITLES_MODE
    split_titles_by_language: bool = True
    dehyphenate: str = DEFAULT_DEHYPHENATE_MODE
    merge_author_blocks: bool = False
    deduplicate_authors: bool = True
    deduplicate_authors_by_initials: bool = True


@dataclasses.dataclass(frozen=True)
class ReferencesConfig:
    kept_by_any_of: Sequence[str] = DEFAULT_REFERENCE_KEPT_BY_ANY_OF
    keep_other_text: bool = True
    fragment_link_doi: str = DEFAULT_FRAGMENT_LINK_DOI
    select_titles: str = DEFAULT_REFERENCE_TITLES_MODE
    select_dois: str = DEFAULT_REFERENCE_DOIS_MODE


VALUES_BY_SETTING: Mapping[Tuple[str, str], Sequence[str]] = {
    ('front', 'select_abstracts'): ABSTRACTS_MODES,
    ('front', 'select_titles'): TITLES_MODES,
    ('front', 'dehyphenate'): DEHYPHENATE_MODES,
    ('references', 'select_titles'): REFERENCE_TITLES_MODES,
    ('references', 'select_dois'): REFERENCE_DOIS_MODES
}


class FullTextProcessorConfig(NamedTuple):
    extract: ExtractConfig = ExtractConfig()
    front: FrontConfig = FrontConfig()
    references: ReferencesConfig = ReferencesConfig()
    use_cv_model: bool = False
    cv_render_dpi: float = DEFAULT_PDF_RENDER_DPI
    use_ocr_model: bool = False
    replace_text_by_cv_graphic: bool = False
    max_graphic_distance: float = DEFAULT_MAX_GRAPHIC_DISTANCE
    max_enclosed_text_ratio: float = DEFAULT_MAX_ENCLOSED_TEXT_RATIO
    noise_filter: LayoutNoiseFilterConfig = LayoutNoiseFilterConfig()
    graphic_regions: GraphicRegionConfig = GraphicRegionConfig()

    @staticmethod
    def from_app_config(app_config: AppConfig) -> 'FullTextProcessorConfig':
        props = dict(app_config.get('processors', {}).get('fulltext', {}))
        nested_props = {
            name: props.pop(name, None)
            for name in ('extract', 'front', 'references', 'noise_filter', 'graphic_regions')
        }
        references_props = nested_props['references']
        if references_props and 'kept_by_any_of' in references_props:
            nested_props['references'] = {
                **references_props,
                'kept_by_any_of': tuple(validate_reference_field_names(
                    references_props['kept_by_any_of']
                ))
            }
        config = FullTextProcessorConfig()._replace(**props)
        for name, name_props in nested_props.items():
            if not name_props:
                continue
            config = config._replace(**{  # pylint: disable=no-member
                name: dataclasses.replace(getattr(config, name), **name_props)
            })
        if config.front.dehyphenate is False:  # YAML reads a bare `off` as false
            config = config._replace(front=dataclasses.replace(
                config.front, dehyphenate=DehyphenateMode.OFF
            ))
        validate_fragment_link_doi(
            config.references.fragment_link_doi, config.references.kept_by_any_of
        )
        for (scope, name), values in VALUES_BY_SETTING.items():
            value = getattr(getattr(config, scope), name)
            if value not in values:
                quoting_hint = (
                    ' (YAML reads a bare word such as on or yes as a boolean; quote it)'
                    if isinstance(value, bool) else ''
                )
                raise ValueError(
                    f'unknown {scope}.{name}: {value!r}, expected one of {list(values)}'
                    + quoting_hint
                )
        return config

    def get_for_requested_field_names(
        self,
        request_field_names: Set[str]
    ) -> 'FullTextProcessorConfig':
        if not request_field_names:
            return self
        remaining_field_names = request_field_names - FRONT_FIELDS - {RequestFieldNames.REFERENCES}
        if remaining_field_names:
            return self
        return self._replace(extract=dataclasses.replace(  # pylint: disable=no-member
            self.extract,
            front=bool(FRONT_FIELDS & request_field_names),
            front_authors=RequestFieldNames.AUTHORS in request_field_names,
            front_affiliations=RequestFieldNames.AFFILIATIONS in request_field_names,
            body_sections=False,
            acknowledgements=False,
            back_sections=False,
            references=RequestFieldNames.REFERENCES in request_field_names,
            graphics=False
        ))

    def get_for_header_document(self) -> 'FullTextProcessorConfig':
        return self.get_for_requested_field_names(FRONT_FIELDS)
