import argparse
import logging
import os
from logging.config import dictConfig
from typing import Optional, Sequence

from fastapi import FastAPI
import uvicorn

from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.service.api.app import create_api_app
from sciencebeam_parser.service.routers.index import create_index_router
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE


LOGGER = logging.getLogger(__name__)


def create_app_for_parser(
    sciencebeam_parser: ScienceBeamParser
) -> FastAPI:
    app = FastAPI(title='ScienceBeam Parser')

    index_router = create_index_router()
    app.include_router(index_router, include_in_schema=False)

    api_app = create_api_app(sciencebeam_parser)
    app.mount('/api', api_app)

    return app


def create_app_for_config(
    config: AppConfig,
    base_config: Optional[AppConfig] = None
) -> FastAPI:
    return create_app_for_parser(
        ScienceBeamParser.from_config(config, base_config=base_config)
    )


def get_base_app_config() -> AppConfig:
    """The config as the file declares it, before any profile is chosen.

    What a request selects is resolved from here rather than from the
    deployment's own resolved config: resolving a second profile onto an already
    resolved one deep-merges the two, leaving keys of the first behind.
    """
    return AppConfig.load_yaml(DEFAULT_CONFIG_FILE)


def get_default_profile_name(base_config: AppConfig) -> Optional[str]:
    return base_config.get_active_profile_name(
        os.environ.get('SCIENCEBEAM_PARSER__PROFILE')
    )


def get_app_config(base_config: Optional[AppConfig] = None) -> AppConfig:
    if base_config is None:
        base_config = get_base_app_config()
    return (
        base_config
        .resolve_profile(get_default_profile_name(base_config))
        .apply_environment_variables()
    )


def apply_logging_config(logging_config: Optional[dict] = None):
    if logging_config:
        for handler_config in logging_config.get('handlers', {}).values():
            filename = handler_config.get('filename')
            if not filename:
                continue
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        try:
            dictConfig(logging_config)
        except ValueError:
            LOGGER.info('logging_config: %r', logging_config)
            raise


def create_app() -> FastAPI:
    base_config = get_base_app_config()
    config = get_app_config(base_config)
    apply_logging_config(config.get('logging'))
    return create_app_for_config(config, base_config=base_config)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host', required=False,
        help='Host to bind server to.'
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help='The port to listen to.'
    )
    parsed_args = parser.parse_args(argv)
    return parsed_args


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    base_config = get_base_app_config()
    config = get_app_config(base_config)
    apply_logging_config(config.get('logging'))
    LOGGER.info('app config: %s', config)
    app = create_app_for_config(config, base_config=base_config)
    uvicorn.run(
        app,
        port=args.port,
        host=args.host
    )


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    main()
