import logging
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from sciencebeam_parser.config.config import AppConfig


LOGGER = logging.getLogger(__name__)


CONFIG_KEY = 'max_upload_bytes'


def get_max_upload_bytes(config: AppConfig) -> Optional[int]:
    """The configured limit in bytes, or `None` for no limit."""
    value = config.get(CONFIG_KEY)
    return int(value) if value else None


def add_upload_size_limit(app: FastAPI, config: AppConfig) -> None:
    """Make `app` answer 413 over the configured limit, if there is one.

    Added to the application rather than wrapped around it, so that the
    middleware's own control flow stays inside its error handling.
    """
    max_bytes = get_max_upload_bytes(config)
    if not max_bytes:
        LOGGER.info('upload size limit: none')
        return
    LOGGER.info('upload size limit: %d bytes', max_bytes)
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=max_bytes)
