import logging

from fastapi import (
    FastAPI,
    Request,
    Response
)
from fastapi.responses import JSONResponse


from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    UnsupportedRequestMediaTypeScienceBeamParserError
)
from sciencebeam_parser.models.llm.usage import (
    USAGE_HEADER_NAME,
    get_request_llm_usage_header_value,
    start_request_llm_usage
)
from sciencebeam_parser.service.api.routers.convert import create_convert_router
from sciencebeam_parser.service.api.routers.grobid import create_grobid_router
from sciencebeam_parser.service.api.routers.low_level import create_low_level_router
from sciencebeam_parser.service.api.routers.models import create_models_router
from sciencebeam_parser.service.api.routers.status import create_status_router


LOGGER = logging.getLogger(__name__)


def create_api_app(
    sciencebeam_parser: ScienceBeamParser
) -> FastAPI:
    app = FastAPI()
    app.state.sciencebeam_parser = sciencebeam_parser

    app.include_router(create_status_router())
    app.include_router(create_convert_router())
    app.include_router(create_grobid_router(
        fulltext_processor_config=sciencebeam_parser.fulltext_processor_config
    ))
    app.include_router(create_low_level_router())
    app.include_router(create_models_router(
        sciencebeam_parser=sciencebeam_parser
    ))

    def with_llm_usage_header(response: Response) -> Response:
        header_value = get_request_llm_usage_header_value()
        if header_value is not None:
            response.headers[USAGE_HEADER_NAME] = header_value
        return response

    @app.middleware('http')
    async def add_llm_usage_header(request: Request, call_next):
        """What the request spent, beside the body rather than in it.

        The accumulator is created here, in the request's own task, so that the
        exception handler below can still read it: a document that failed has
        already paid for the responses it got.
        """
        start_request_llm_usage()
        return with_llm_usage_header(await call_next(request))

    @app.exception_handler(Exception)
    async def log_unhandled_exceptions(
        request: Request,
        exc: Exception  # pylint: disable=unused-argument
    ):
        LOGGER.exception("Unhandled exception on %s %s", request.method, request.url)
        return with_llm_usage_header(JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        ))

    @app.exception_handler(UnsupportedRequestMediaTypeScienceBeamParserError)
    async def handle_unsupported_request_media_type(
        request: Request,  # pylint: disable=unused-argument
        exc: UnsupportedRequestMediaTypeScienceBeamParserError
    ):
        LOGGER.info("Unsupported request media type: %s", exc)
        return JSONResponse(
            status_code=406,
            content={"detail": str(exc)},
        )

    @app.get('/')
    def api_root() -> dict:
        return {
            'links': {}
        }

    return app
