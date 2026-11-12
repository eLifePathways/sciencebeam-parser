"""What the profiles endpoint and the profile parameter tell a caller.

Its own module rather than part of `app_test`, which has as much of the API as
one file should carry.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx2
import pytest

from sciencebeam_parser.app.profiles import ProfileDescription
from sciencebeam_parser.service.api.app import create_api_app


PROFILE_NAME_1 = 'profile1'
PROFILE_NAME_2 = 'profile2'

PROFILE_DESCRIPTIONS = [
    ProfileDescription(
        name=PROFILE_NAME_1,
        label='Profile 1',
        description='What profile 1 serves.',
        alias_names=['profile_one'],
        is_default=True
    ),
    ProfileDescription(
        name=PROFILE_NAME_2,
        label=None,
        description=None,
        alias_names=[],
        is_default=False
    ),
]


@pytest.fixture(name='sciencebeam_parser_mock')
def _sciencebeam_parser_mock() -> MagicMock:
    mock = MagicMock(name='sciencebeam_parser_mock')
    mock.profile_registry.get_available_profile_names.return_value = [
        PROFILE_NAME_1, PROFILE_NAME_2
    ]
    mock.profile_registry.get_profile_descriptions.return_value = PROFILE_DESCRIPTIONS
    return mock


@pytest.fixture(name='test_client')
def _test_client(sciencebeam_parser_mock: MagicMock) -> TestClient:
    return TestClient(create_api_app(sciencebeam_parser=sciencebeam_parser_mock))


def _get_ok_json(response: httpx2.Response) -> dict:
    assert response.status_code == 200
    return response.json()


def _without_none_values(value):
    """What the schema document keeps of a response: FastAPI encodes it without nulls."""
    if isinstance(value, dict):
        return {
            key: _without_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none_values(item) for item in value]
    return value


class TestProfileList:
    def test_should_say_what_every_selectable_profile_is_for(
        self, test_client: TestClient
    ):
        assert _get_ok_json(test_client.get('/profiles')) == {
            'profiles': [
                {
                    'name': PROFILE_NAME_1,
                    'label': 'Profile 1',
                    'description': 'What profile 1 serves.',
                    'alias_names': ['profile_one'],
                    'is_default': True,
                    'tracks': None
                },
                {
                    'name': PROFILE_NAME_2,
                    'label': None,
                    'description': None,
                    'alias_names': [],
                    'is_default': False,
                    'tracks': None
                },
            ]
        }

    def test_should_document_what_this_deployment_serves_as_the_response_example(
        self, sciencebeam_parser_mock: MagicMock, test_client: TestClient
    ):
        """What a reader of the generated docs sees without executing anything."""
        schema = create_api_app(sciencebeam_parser=sciencebeam_parser_mock).openapi()
        example = (
            schema['paths']['/profiles']['get']
            ['responses']['200']['content']['application/json']['example']
        )
        assert example == _without_none_values(_get_ok_json(test_client.get('/profiles')))


class TestSelectableProfilesInTheSchema:
    def test_should_name_every_selectable_profile_on_every_route(
        self, sciencebeam_parser_mock: MagicMock
    ):
        """So the docs offer the choice rather than a free text box."""
        schema = create_api_app(sciencebeam_parser=sciencebeam_parser_mock).openapi()
        enums = [
            sub_schema['enum']
            for operations in schema['paths'].values()
            for operation in operations.values()
            for parameter in operation.get('parameters', [])
            if parameter['name'] == 'profile'
            for sub_schema in parameter['schema'].get(
                'anyOf', [parameter['schema']]
            )
            if sub_schema.get('type') == 'string'
        ]
        assert enums
        assert all(enum == [PROFILE_NAME_1, PROFILE_NAME_2] for enum in enums)

    def test_should_keep_the_named_branch_first(
        self, sciencebeam_parser_mock: MagicMock
    ):
        """What makes the docs render a list rather than a free text box.

        Swagger UI merges only the first branch of an `anyOf` into the parameter
        schema, and the `enum` it renders is whatever that branch carries -- so
        the string branch leading is the dropdown, not a detail of the document.
        """
        schema = create_api_app(sciencebeam_parser=sciencebeam_parser_mock).openapi()
        first_branches = [
            parameter['schema']['anyOf'][0]
            for operations in schema['paths'].values()
            for operation in operations.values()
            for parameter in operation.get('parameters', [])
            if parameter['name'] == 'profile'
        ]
        assert first_branches
        assert all(branch.get('type') == 'string' for branch in first_branches)

    def test_should_still_serve_the_documentation(
        self, test_client: TestClient
    ):
        assert test_client.get('/openapi.json').status_code == 200
