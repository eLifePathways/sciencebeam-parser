import logging

from fastapi import (
    FastAPI,
    Request,
    Response
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse


from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    UnsupportedRequestMediaTypeScienceBeamParserError
)
from sciencebeam_parser.app.profiles import (
    get_request_profile_headers,
    start_request_profile
)
from sciencebeam_parser.models.llm.usage import (
    USAGE_HEADER_NAME,
    get_request_llm_usage_header_value,
    start_request_llm_usage
)
from sciencebeam_parser.service.api.dependencies import (
    add_profile_names_to_openapi_schema
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
    app.include_router(create_grobid_router())
    app.include_router(create_low_level_router())
    app.include_router(create_models_router(
        sciencebeam_parser=sciencebeam_parser
    ))

    def openapi_naming_selectable_profiles() -> dict:
        if not app.openapi_schema:
            app.openapi_schema = add_profile_names_to_openapi_schema(
                get_openapi(
                    title=app.title,
                    version=app.version,
                    openapi_version=app.openapi_version,
                    description=app.description,
                    routes=app.routes
                ),
                sciencebeam_parser.profile_registry.get_available_profile_names()
            )
        return app.openapi_schema

    app.openapi = openapi_naming_selectable_profiles  # type: ignore[method-assign]

    def with_request_headers(response: Response) -> Response:
        header_value = get_request_llm_usage_header_value()
        if header_value is not None:
            response.headers[USAGE_HEADER_NAME] = header_value
        for name, value in get_request_profile_headers().items():
            response.headers[name] = value
        return response

    @app.middleware('http')
    async def add_request_headers(request: Request, call_next):
        """What served the request and what it spent, beside the body.

        Both are created here, in the request's own task, so that the exception
        handler below can still read them: a document that failed has already
        paid for the responses it got, and was still served by a profile.
        """
        start_request_llm_usage()
        start_request_profile()
        return with_request_headers(await call_next(request))

    @app.exception_handler(Exception)
    async def log_unhandled_exceptions(
        request: Request,
        exc: Exception  # pylint: disable=unused-argument
    ):
        LOGGER.exception("Unhandled exception on %s %s", request.method, request.url)
        return with_request_headers(JSONResponse(
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
