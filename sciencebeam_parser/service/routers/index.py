import json
import os
from typing import List, Sequence, Tuple

from fastapi import APIRouter, Response

from sciencebeam_parser.app.parser import (
    DOC_TO_PDF_SUPPORTED_MEDIA_TYPES,
    ZIP_ENTRY_NAME_BY_MEDIA_TYPE,
    ScienceBeamParser
)
from sciencebeam_parser.processors.fulltext.config import RequestFieldNames
from sciencebeam_parser.service.api.app import API_PATH
from sciencebeam_parser.service.api.routers.convert import CONVERT_RESPONSE_MEDIA_TYPES
from sciencebeam_parser.service.assets import (
    PAGE_CACHE_CONTROL,
    STATIC_DIRECTORY,
    get_versioned_html
)
from sciencebeam_parser.utils.media_types import (
    MediaTypes,
    guess_extension_for_media_type
)


INDEX_HTML_FILE = os.path.join(STATIC_DIRECTORY, 'index.html')

DEPLOYMENT_PLACEHOLDER = '__DEPLOYMENT_JSON__'


SOURCE_MEDIA_TYPE_ORDER: Sequence[str] = (
    MediaTypes.PDF,
    MediaTypes.DOCX,
    MediaTypes.DOTX,
    MediaTypes.DOC,
    MediaTypes.RTF
)


OUTPUT_MEDIA_TYPE_LABELS = {
    MediaTypes.JATS_XML: 'JATS XML',
    MediaTypes.TEI_XML: 'TEI XML',
    MediaTypes.JATS_ZIP: 'JATS XML with figures (zip)',
    MediaTypes.TEI_ZIP: 'TEI XML with figures (zip)',
    MediaTypes.PDF: 'The PDF as parsed'
}


TEXT_OUTPUT_MEDIA_TYPES = {MediaTypes.JATS_XML, MediaTypes.TEI_XML}


READABLE_OUTPUT_MEDIA_TYPES = {MediaTypes.JATS_XML, MediaTypes.JATS_ZIP}


FRONT_MATTER_FIELD_NAMES: Sequence[str] = (
    RequestFieldNames.TITLE,
    RequestFieldNames.ABSTRACT,
    RequestFieldNames.AUTHORS,
    RequestFieldNames.AFFILIATIONS
)

INCLUDES_GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ('Front matter', FRONT_MATTER_FIELD_NAMES),
    ('References', (RequestFieldNames.REFERENCES,))
)


def get_accepted_source_media_types(doc_to_pdf_enabled: bool) -> List[str]:
    accepted_media_types = {MediaTypes.PDF}
    if doc_to_pdf_enabled:
        accepted_media_types |= DOC_TO_PDF_SUPPORTED_MEDIA_TYPES
    return [
        media_type
        for media_type in SOURCE_MEDIA_TYPE_ORDER
        if media_type in accepted_media_types
    ]


def get_deployment_props(sciencebeam_parser: ScienceBeamParser) -> dict:
    return {
        'api_path': API_PATH,
        'accepted_source_media_types': [
            {
                'media_type': media_type,
                'extension': guess_extension_for_media_type(media_type)
            }
            for media_type in get_accepted_source_media_types(
                sciencebeam_parser.doc_to_pdf_enabled
            )
        ],
        'output_media_types': [
            {
                'media_type': media_type,
                'label': OUTPUT_MEDIA_TYPE_LABELS[media_type],
                'extension': guess_extension_for_media_type(media_type),
                'is_text': media_type in TEXT_OUTPUT_MEDIA_TYPES,
                'is_readable': media_type in READABLE_OUTPUT_MEDIA_TYPES,
                'archive_xml_name': ZIP_ENTRY_NAME_BY_MEDIA_TYPE.get(media_type)
            }
            for media_type in CONVERT_RESPONSE_MEDIA_TYPES
        ],
        'includes_groups': [
            {'label': label, 'field_names': list(field_names)}
            for label, field_names in INCLUDES_GROUPS
        ]
    }


def get_embedded_json(props: dict) -> str:
    return json.dumps(props).replace('<', '\\u003c')


def get_index_html(props: dict) -> str:
    with open(INDEX_HTML_FILE, 'r', encoding='utf-8') as html_fp:
        html = html_fp.read()
    assert DEPLOYMENT_PLACEHOLDER in html, \
        f'{INDEX_HTML_FILE} has no {DEPLOYMENT_PLACEHOLDER}'
    return get_versioned_html(
        html.replace(DEPLOYMENT_PLACEHOLDER, get_embedded_json(props))
    )


def create_index_router(sciencebeam_parser: ScienceBeamParser) -> APIRouter:
    """The page, carrying what this deployment accepts rather than asking for it.

    Whether the Word family is converted is a property of the configuration this
    process was started with, and no route states it. Rendering it into the page
    keeps the file a request is refused for the same set the request would be
    refused for, without a route that exists only for this page.
    """
    router = APIRouter()

    deployment_props = get_deployment_props(sciencebeam_parser)

    @router.get('/')
    def index() -> Response:
        return Response(
            get_index_html(deployment_props),
            media_type='text/html',
            headers={'Cache-Control': PAGE_CACHE_CONTROL}
        )

    return router
