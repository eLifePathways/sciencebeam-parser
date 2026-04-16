import logging

from fastapi import (
    FastAPI,
    Request
)
from fastapi.responses import JSONResponse


from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    UnsupportedRequestMediaTypeScienceBeamParserError
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

    @app.exception_handler(Exception)
    async def log_unhandled_exceptions(
        request: Request,
        exc: Exception  # pylint: disable=unused-argument
    ):
        LOGGER.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

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
