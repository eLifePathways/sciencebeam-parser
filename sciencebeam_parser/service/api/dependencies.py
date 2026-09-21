import logging
from typing import Annotated, Callable, Iterator, Optional, Protocol, Sequence

from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi import (
    Depends,
    File,
    HTTPException,
    Header,
    Query,
    Request,
    UploadFile,
    status
)


from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    ScienceBeamParserSession,
    ScienceBeamParserSessionSource
)
from sciencebeam_parser.app.profiles import (
    ProfileBundle,
    ProfileNotSelectableError,
    ProfileRegistry,
    TooManyLoadedModelsError,
    record_request_profile
)
from sciencebeam_parser.config.config import UnknownProfileError
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
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


def get_profile_registry(request: Request) -> ProfileRegistry:
    return request.app.state.sciencebeam_parser.profile_registry


PROFILE_QUERY_DESCRIPTION = (
    'Name of the profile to serve this request with, from those the deployment '
    'declares selectable. Defaults to the deployment\'s own profile.'
)


def get_profile_bundle(
    *,
    profile_registry: Annotated[ProfileRegistry, Depends(get_profile_registry)],
    profile: Annotated[
        Optional[str], Query(description=PROFILE_QUERY_DESCRIPTION)
    ] = None
) -> ProfileBundle:
    """Which models and processor config serve this request.

    One dependency for every route that serves a document, rather than a
    parameter each route remembers to declare: a route that quietly ignored it
    would make a comparison read as a null result rather than as a mistake.
    """
    try:
        bundle = profile_registry.get_bundle(profile)
    except (UnknownProfileError, ProfileNotSelectableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except TooManyLoadedModelsError as exc:
        # Not the request's fault, and it may succeed later: what this process
        # already holds is what refused it.
        LOGGER.warning('refusing profile %r: %s', profile, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    record_request_profile(bundle)
    return bundle


GetFullTextProcessorConfig = Callable[[FullTextProcessorConfig], FullTextProcessorConfig]


def get_sciencebeam_parser_session_dependency_factory(
    get_fulltext_processor_config: Optional[GetFullTextProcessorConfig] = None,
    **session_kwargs,
) -> ScienceBeamParserSessionDependencyFactory:
    """A session over the requested profile, narrowed by what the route asks for.

    The route contributes a narrowing rather than a processor config, because the
    config it narrows arrives with the profile and is no longer known when the
    router is built.
    """
    def get_session(
        *,
        sciencebeam_parser: Annotated[ScienceBeamParser, Depends(get_sciencebeam_parser)],
        profile_bundle: Annotated[ProfileBundle, Depends(get_profile_bundle)],
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> Iterator[ScienceBeamParserSession]:
        fulltext_processor_config = profile_bundle.fulltext_processor_config
        if get_fulltext_processor_config is not None:
            fulltext_processor_config = get_fulltext_processor_config(
                fulltext_processor_config
            )
        with sciencebeam_parser.get_new_session(
            fulltext_processor_config=fulltext_processor_config,
            fulltext_models=profile_bundle.fulltext_models,
            **session_kwargs
        ) as session:
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
    get_fulltext_processor_config: Optional[GetFullTextProcessorConfig] = None,
    **session_kwargs,
) -> ScienceBeamParserSessionSourceDependencyFactory:
    def get_source(
        *,
        session: Annotated[
            ScienceBeamParserSession,
            Depends(
                get_sciencebeam_parser_session_dependency_factory(
                    get_fulltext_processor_config, **session_kwargs
                )
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
