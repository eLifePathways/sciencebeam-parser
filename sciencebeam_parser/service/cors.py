import logging
from typing import List

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from sciencebeam_parser.app.profiles import (
    PROFILE_DIGEST_HEADER_NAME,
    PROFILE_HEADER_NAME,
    PROFILE_REQUESTED_HEADER_NAME
)
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.llm.usage import USAGE_HEADER_NAME


LOGGER = logging.getLogger(__name__)


ANY_ORIGIN = '*'

ANY_HEADER = '*'


# Read from the constants the responses are built from, so what a page may read
# cannot drift from what the API emits.
EXPOSED_HEADER_NAMES = [
    PROFILE_HEADER_NAME,
    PROFILE_DIGEST_HEADER_NAME,
    PROFILE_REQUESTED_HEADER_NAME,
    USAGE_HEADER_NAME
]

ALLOWED_METHOD_NAMES = ['GET', 'POST', 'OPTIONS']

PREFLIGHT_MAX_AGE_SEC = 24 * 60 * 60


def get_allowed_origins(config: AppConfig) -> List[str]:
    """The origins whose pages may call the API, empty for none."""
    configured = (config.get('cors') or {}).get('allowed_origins') or []
    if isinstance(configured, str):
        configured = [configured]
    return list(configured)


def with_cross_origin_access(app: ASGIApp, config: AppConfig) -> ASGIApp:
    """`app`, answering the origins the config permits, or unchanged for none.

    Wraps rather than being added to the app, so that the `500` its own error
    middleware produces is sent through this and carries the access headers.
    """
    allowed_origins = get_allowed_origins(config)
    if not allowed_origins:
        LOGGER.info('cross-origin access: no origin permitted')
        return app
    LOGGER.info('cross-origin access: %r', allowed_origins)
    return CORSMiddleware(
        app,
        allow_origins=allowed_origins,
        allow_methods=ALLOWED_METHOD_NAMES,
        allow_headers=[ANY_HEADER],
        expose_headers=EXPOSED_HEADER_NAMES,
        allow_credentials=False,
        max_age=PREFLIGHT_MAX_AGE_SEC
    )
