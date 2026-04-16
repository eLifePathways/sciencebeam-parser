import logging

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse


LOGGER = logging.getLogger(__name__)


def create_status_router() -> APIRouter:
    router = APIRouter(tags=['status'])

    @router.get('/isalive', response_class=PlainTextResponse)
    def process_isalive_api() -> str:
        return 'true'

    return router
