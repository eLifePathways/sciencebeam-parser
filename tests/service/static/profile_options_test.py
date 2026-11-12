"""Which profiles the page's menu offers.

A profile that tracks another serves exactly what that one serves, so offering
both would offer one conversion twice under two names.
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest


HARNESS_FILE = Path(__file__).parent / 'get_profile_options.mjs'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the profile menu that the page runs in a browser'
)


def get_menu(profiles: List[dict]) -> dict:
    result = subprocess.run(
        ['node', str(HARNESS_FILE)],
        input=json.dumps(profiles),
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)


def profile(name: str, tracks: Optional[str] = None) -> dict:
    return {'name': name, 'label': name, 'tracks': tracks, 'is_default': False}


class TestGetProfileOptions:
    def test_should_offer_one_option_for_a_profile_and_the_one_it_tracks(self):
        menu = get_menu([profile('pinned_a'), profile('tracking_a', tracks='pinned_a')])
        assert menu['options'] == ['tracking_a']

    def test_should_offer_a_profile_nothing_tracks(self):
        menu = get_menu([profile('pinned_a'), profile('pinned_b')])
        assert menu['options'] == ['pinned_a', 'pinned_b']

    def test_should_keep_a_profile_whose_target_this_deployment_withholds(self):
        """Dropping it would leave a deployment that narrowed the list with nothing."""
        menu = get_menu([profile('tracking_a', tracks='pinned_a')])
        assert menu['options'] == ['tracking_a']

    def test_should_offer_nothing_where_the_service_lists_nothing(self):
        assert get_menu([])['options'] == []

    def test_should_keep_the_order_the_service_listed(self):
        menu = get_menu([
            profile('pinned_b'), profile('tracking_a', tracks='pinned_a'), profile('pinned_a')
        ])
        assert menu['options'] == ['pinned_b', 'tracking_a']


class TestGetTrackedProfile:
    def test_should_find_what_a_profile_tracks(self):
        menu = get_menu([profile('pinned_a'), profile('tracking_a', tracks='pinned_a')])
        assert menu['tracked']['tracking_a'] == 'pinned_a'

    def test_should_find_nothing_for_a_profile_that_tracks_nothing(self):
        menu = get_menu([profile('pinned_a')])
        assert menu['tracked']['pinned_a'] is None

    def test_should_find_nothing_where_the_deployment_withholds_the_target(self):
        menu = get_menu([profile('tracking_a', tracks='pinned_a')])
        assert menu['tracked']['tracking_a'] is None
