"""How the page divides the profile menu.

A profile that tracks another serves that one's models, so the menu has to say
which of the two a reader is choosing rather than offering the same combination
twice.
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest


HARNESS_FILE = Path(__file__).parent / 'get_profile_option_groups.mjs'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the profile menu that the page runs in a browser'
)


def get_profile_option_groups(profiles: List[dict]) -> List[dict]:
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


class TestGetProfileOptionGroups:
    def test_should_separate_a_tracking_profile_from_a_pinned_one(self):
        groups = get_profile_option_groups([
            profile('pinned_a'), profile('tracking_a', tracks='pinned_a')
        ])
        assert [
            (group['label'], [item['name'] for item in group['profiles']])
            for group in groups
        ] == [
            ('Recommended', ['tracking_a']),
            ('A specific combination', ['pinned_a'])
        ]

    def test_should_offer_what_the_service_recommends_first(self):
        groups = get_profile_option_groups([
            profile('pinned_a'), profile('tracking_a', tracks='pinned_a')
        ])
        assert groups[0]['profiles'][0]['name'] == 'tracking_a'

    def test_should_not_head_a_list_that_holds_one_kind(self):
        groups = get_profile_option_groups([profile('pinned_a'), profile('pinned_b')])
        assert [group['label'] for group in groups] == [None]
        assert [item['name'] for item in groups[0]['profiles']] == ['pinned_a', 'pinned_b']

    def test_should_not_head_a_list_that_tracks_throughout(self):
        groups = get_profile_option_groups([profile('tracking_a', tracks='pinned_a')])
        assert [group['label'] for group in groups] == [None]

    def test_should_offer_nothing_where_the_service_lists_nothing(self):
        assert get_profile_option_groups([]) == []
