import os
import re

from fastapi import Response
from fastapi.staticfiles import StaticFiles


STATIC_PATH = '/static'

STATIC_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'static'
)

STATIC_REFERENCE_PATTERN = re.compile(
    r'(?P<attribute>href|src)="' + STATIC_PATH + r'/(?P<path>[^"?]+)"'
)

VENDORED_PATH = '/swagger-ui/'

PAGE_CACHE_CONTROL = 'no-store'
ASSET_CACHE_CONTROL = 'no-cache'
VENDORED_CACHE_CONTROL = 'public, max-age=31536000, immutable'


def get_asset_version(relative_path: str) -> str:
    try:
        return str(int(os.path.getmtime(
            os.path.join(STATIC_DIRECTORY, relative_path)
        )))
    except OSError:
        return '0'


def get_versioned_static_url(relative_path: str) -> str:
    return f'{STATIC_PATH}/{relative_path}?v={get_asset_version(relative_path)}'


def get_versioned_html(html: str) -> str:
    """The page, naming the version of each asset it asks for.

    A page and the assets it names have to be of one version, and what a
    browser already holds is the half that goes stale without saying so.
    """
    return STATIC_REFERENCE_PATTERN.sub(
        lambda match: (
            f'{match.group("attribute")}="'
            f'{get_versioned_static_url(match.group("path"))}"'
        ),
        html
    )


def get_cache_control_for_path(path: str) -> str:
    return (
        VENDORED_CACHE_CONTROL if VENDORED_PATH in path else ASSET_CACHE_CONTROL
    )


class StaticAssets(StaticFiles):
    """What the page names, said to be worth re-asking about.

    An asset a page names can carry its version in the url; one that another
    asset imports cannot, because nothing rewrites an import. Revalidating is
    what covers the second, and costs a not-modified where nothing changed.
    The vendored copies are the exception: their url carries their version.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        scope = args[2] if len(args) > 2 else kwargs['scope']
        response.headers['Cache-Control'] = get_cache_control_for_path(
            scope.get('path', '')
        )
        return response
