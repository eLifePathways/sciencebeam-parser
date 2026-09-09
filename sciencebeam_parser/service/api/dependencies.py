import logging
from typing import Annotated, Iterator, Optional, Protocol, Sequence

from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi import (
    Depends,
    File,
    HTTPException,
    Header,
    Request,
    UploadFile,
    status
)


from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    ScienceBeamParserSession,
    ScienceBeamParserSessionSource
)
from sciencebeam_parser.utils.data_wrapper import (
    MediaDataWrapper,
    get_data_wrapper_with_improved_media_type_or_filename
)
from sciencebeam_parser.utils.media_types import MediaTypes


LOGGER = logging.getLogger(__name__)


class ScienceBeamParserSessionDependencyFactory(Protocol):
    def __call__(self, *args, **kwargs) -> Iterator[ScienceBeamParserSession]:
        pass


class ScienceBeamParserSessionSourceDependencyFactory(Protocol):
    def __call__(self, *args, **kwargs) -> Iterator[ScienceBeamParserSessionSource]:
        pass


def get_media_data_wrapper_for_upload_file(
    upload_file: StarletteUploadFile,
    filename: Optional[str] = None
) -> MediaDataWrapper:
    data = upload_file.file.read()
    return MediaDataWrapper(
        data=data,
        media_type=upload_file.content_type or MediaTypes.OCTET_STREAM,
        filename=upload_file.filename or filename,
    )


async def get_media_data_wrapper(
    request: Request,
    input: Annotated[Optional[UploadFile], File()] = None,  # pylint: disable=redefined-builtin
    filename: Optional[str] = None,
) -> MediaDataWrapper:
    """
    Prefer the documented `input` param, but also accept:
    - legacy 'file' field in multipart form
    - raw request body
    """
    if input is not None:
        LOGGER.info('Using file upload `input`')
        return get_media_data_wrapper_for_upload_file(input)

    content_type = request.headers.get("content-type", "")
    LOGGER.info("Content-Type: %r", content_type)
    if (
        content_type.startswith("multipart/form-data")
        or content_type.startswith('application/x-www-form')
    ):
        form = await request.form()

        file = form.get('file')
        LOGGER.info('file: %r (%r)', file, type(file))
        if isinstance(file, StarletteUploadFile):
            LOGGER.info('Using file upload `file`')
            return get_media_data_wrapper_for_upload_file(file)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="multipart request without 'input' or 'file' field",
        )
    body = await request.body()
    if body:
        LOGGER.info('Using body as source')
        return MediaDataWrapper(
            data=body,
            media_type=MediaTypes.OCTET_STREAM,
            filename=filename,
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="one of 'file', 'input' or raw body is required",
    )


def get_sciencebeam_parser(request: Request) -> ScienceBeamParser:
    return request.app.state.sciencebeam_parser


def get_sciencebeam_parser_session_dependency_factory(
    **session_kwargs,
) -> ScienceBeamParserSessionDependencyFactory:
    def get_session(
        *,
        sciencebeam_parser: Annotated[ScienceBeamParser, Depends(get_sciencebeam_parser)],
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> Iterator[ScienceBeamParserSession]:
        with sciencebeam_parser.get_new_session(**session_kwargs) as session:
            session.document_request_parameters.first_page = first_page
            session.document_request_parameters.last_page = last_page
            yield session

    return get_session


def get_session_source_for_data_wrapper(
    session: ScienceBeamParserSession,
    data_wrapper: MediaDataWrapper
) -> ScienceBeamParserSessionSource:
    """Turn an upload into a session source, for every route that takes one.

    Shared rather than repeated, because a route with its own copy is a route
    that silently stops logging or naming the document when this one changes.
    """
    data_wrapper = get_data_wrapper_with_improved_media_type_or_filename(
        data_wrapper
    )
    # The uploaded name is the only thing tying the rest of this request's log
    # lines to a document; without it a run is anonymous documents.
    LOGGER.info(
        'processing document: filename=%r media_type=%r session=%s',
        data_wrapper.filename, data_wrapper.media_type, session.temp_path
    )
    source_path = session.temp_path / "source.file"
    source_path.write_bytes(data_wrapper.data)
    return session.get_source(
        source_path=str(source_path),
        source_media_type=data_wrapper.media_type,
        source_name=data_wrapper.filename,
    )


def get_sciencebeam_parser_session_source_dependency_factory(
    **session_kwargs,
) -> ScienceBeamParserSessionSourceDependencyFactory:
    def get_source(
        *,
        session: Annotated[
            ScienceBeamParserSession,
            Depends(
                get_sciencebeam_parser_session_dependency_factory(**session_kwargs)
            )
        ],
        data_wrapper: Annotated[MediaDataWrapper, Depends(get_media_data_wrapper)],
    ) -> Iterator[ScienceBeamParserSessionSource]:
        yield get_session_source_for_data_wrapper(session, data_wrapper)

    return get_source


class GetResponseMediaTypeDependencyFactory(Protocol):
    def __call__(self, *args, **kwargs) -> str:
        pass


def assert_and_get_first_accept_matching_media_type_factory(
    available_media_types: Sequence[str]
) -> GetResponseMediaTypeDependencyFactory:
    def get_media_type(
        *,
        accept: Annotated[Optional[str], Header(alias="Accept")] = None,
    ) -> str:
        LOGGER.debug('accept: %r', accept)
        # no Accept → default to first supported type
        if not accept:
            return available_media_types[0]

        # very simple matching; extend if you need full RFC handling
        if accept == "*/*":
            return available_media_types[0]

        for media_type in available_media_types:
            if media_type == accept:
                return media_type

        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"Supported media types: {', '.join(available_media_types)}",
        )

    return get_media_type
