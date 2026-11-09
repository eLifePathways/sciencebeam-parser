import os
import re
import time
from pathlib import Path
from sciencebeam_parser.service.assets import (
    ASSET_CACHE_CONTROL,
    VENDORED_CACHE_CONTROL,
    get_asset_version,
    get_cache_control_for_path,
    get_versioned_html,
    get_versioned_static_url
)


class TestGetAssetVersion:
    def test_should_follow_the_file_it_names(self, tmp_path: Path, monkeypatch):
        asset = tmp_path / 'index.js'
        asset.write_text('first')
        monkeypatch.setattr(
            'sciencebeam_parser.service.assets.STATIC_DIRECTORY', str(tmp_path)
        )
        before = get_asset_version('index.js')
        os.utime(asset, (time.time() + 10, time.time() + 10))
        assert get_asset_version('index.js') != before

    def test_should_not_fail_for_a_file_that_is_not_there(self):
        assert get_asset_version('nothing-here.js') == '0'


class TestGetVersionedStaticUrl:
    def test_should_name_the_version_of_what_it_points_at(self):
        url = get_versioned_static_url('index.js')
        assert url.startswith('/static/index.js?v=')
        assert url.split('?v=')[1] == get_asset_version('index.js')


class TestGetVersionedHtml:
    def test_should_version_what_the_page_names(self):
        html = get_versioned_html(
            '<link rel="stylesheet" href="/static/index.css">'
            '<script type="module" src="/static/index.js"></script>'
        )
        assert re.search(r'href="/static/index\.css\?v=\d+"', html)
        assert re.search(r'src="/static/index\.js\?v=\d+"', html)

    def test_should_leave_what_it_does_not_serve(self):
        html = '<a href="/api/docs">docs</a><img src="data:image/png;base64,x">'
        assert get_versioned_html(html) == html

    def test_should_leave_a_reference_that_already_names_a_version(self):
        html = '<script src="/static/index.js?v=1"></script>'
        assert get_versioned_html(html) == html


class TestGetCacheControlForPath:
    def test_should_have_the_page_assets_re_asked_about(self):
        assert get_cache_control_for_path('/index.js') == ASSET_CACHE_CONTROL
        assert get_cache_control_for_path('/index.css') == ASSET_CACHE_CONTROL

    def test_should_let_the_vendored_copies_be_held(self):
        assert get_cache_control_for_path(
            '/swagger-ui/swagger-ui-bundle.js'
        ) == VENDORED_CACHE_CONTROL
