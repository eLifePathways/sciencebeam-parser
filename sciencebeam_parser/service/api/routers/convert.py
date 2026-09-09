import logging
from typing import Annotated, Iterator, Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    ScienceBeamParserSession,
    ScienceBeamParserSessionSource
)
from sciencebeam_parser.service.api.dependencies import (
    ScienceBeamParserSessionDependencyFactory,
    ScienceBeamParserSessionSourceDependencyFactory,
    assert_and_get_first_accept_matching_media_type_factory,
    get_media_data_wrapper,
    get_sciencebeam_parser,
    get_session_source_for_data_wrapper
)
from sciencebeam_parser.service.api.routers.docs import (
    PDF_CONTENT_DOC,
    TEI_AND_JATS_XML_CONTENT_DOC,
    TEI_AND_JATS_ZIP_CONTENT_DOC
)
from sciencebeam_parser.service.api.routers.utils import get_processed_source_to_response_media_type
from sciencebeam_parser.utils.data_wrapper import MediaDataWrapper
from sciencebeam_parser.utils.media_types import MediaTypes
from sciencebeam_parser.utils.text import parse_comma_separated_value


LOGGER = logging.getLogger(__name__)


def get_convert_sciencebeam_parser_session_dependency_factory(
) -> ScienceBeamParserSessionDependencyFactory:
    def get_session(
        *,
        sciencebeam_parser: Annotated[ScienceBeamParser, Depends(get_sciencebeam_parser)],
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        includes: Annotated[
            Optional[str],
            Query(description="Comma-separated list of requested fields, e.g. title,abstract")
        ] = None
    ) -> Iterator[ScienceBeamParserSession]:
        includes_list = parse_comma_separated_value(includes or '')
        LOGGER.info('includes_list: %r', includes_list)
        fulltext_processor_config = (
            sciencebeam_parser
            .fulltext_processor_config
            .get_for_requested_field_names(set(includes_list))
        )
        with sciencebeam_parser.get_new_session(
            fulltext_processor_config=fulltext_processor_config
        ) as session:
            session.document_request_parameters.first_page = first_page
            session.document_request_parameters.last_page = last_page
            yield session

    return get_session


def get_convert_sciencebeam_parser_session_source_dependency_factory(
) -> ScienceBeamParserSessionSourceDependencyFactory:
    def get_source(
        *,
        session: Annotated[
            ScienceBeamParserSession,
            Depends(
                get_convert_sciencebeam_parser_session_dependency_factory()
            )
        ],
        data_wrapper: Annotated[MediaDataWrapper, Depends(get_media_data_wrapper)],
    ) -> Iterator[ScienceBeamParserSessionSource]:
        yield get_session_source_for_data_wrapper(session, data_wrapper)

    return get_source


def create_convert_router() -> APIRouter:
    router = APIRouter(tags=['convert'])

    @router.post(
        '/convert',
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    **TEI_AND_JATS_XML_CONTENT_DOC,
                    **TEI_AND_JATS_ZIP_CONTENT_DOC,
                    MediaTypes.PDF: PDF_CONTENT_DOC
                }
            },
            406: {"description": "No acceptable media type"},
        },
    )
    def process_convert_api(
        source: Annotated[
            ScienceBeamParserSessionSource,
            Depends(
                get_convert_sciencebeam_parser_session_source_dependency_factory()
            )
        ],
        response_media_type: Annotated[
            str,
            Depends(
                assert_and_get_first_accept_matching_media_type_factory(
                    [
                        MediaTypes.JATS_XML, MediaTypes.TEI_XML,
                        MediaTypes.JATS_ZIP, MediaTypes.TEI_ZIP,
                        MediaTypes.PDF
                    ]
                )
            )
        ],
    ) -> FileResponse:
        return get_processed_source_to_response_media_type(
            source=source,
            response_media_type=response_media_type
        )

    return router
