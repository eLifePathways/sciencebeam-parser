import logging
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from sciencebeam_parser.models.data import DEFAULT_APP_FEATURES_CONTEXT
from sciencebeam_parser.service.api.routers.models import create_models_router
from tests.processors.fulltext.model_mocks import MockFullTextModels


LOGGER = logging.getLogger(__name__)


@pytest.fixture(name='mock_fulltext_models')
def _mock_fulltext_models() -> MockFullTextModels:
    return MockFullTextModels()


@pytest.fixture(name='test_client')
def _test_client(mock_fulltext_models: MockFullTextModels) -> TestClient:
    sciencebeam_parser_mock = MagicMock(name='sciencebeam_parser')
    sciencebeam_parser_mock.fulltext_models = mock_fulltext_models
    sciencebeam_parser_mock.app_features_context = DEFAULT_APP_FEATURES_CONTEXT
    app = FastAPI()
    app.include_router(create_models_router(sciencebeam_parser_mock))
    return TestClient(app)


class TestGetFeatureNames:
    def test_segmentation_returns_feature_names(self, test_client: TestClient):
        response = test_client.get('/models/segmentation/feature-names')
        assert response.status_code == 200
        data = response.json()
        assert 'feature_names' in data
        feature_names = data['feature_names']
        assert feature_names[0] == 'token_text'
        assert 'whole_line_text' in feature_names

    def test_header_returns_feature_names(self, test_client: TestClient):
        response = test_client.get('/models/header/feature-names')
        assert response.status_code == 200
        feature_names = response.json()['feature_names']
        assert feature_names[0] == 'token_text'
        assert 'is_largest_font' in feature_names

    def test_citation_returns_feature_names(self, test_client: TestClient):
        response = test_client.get('/models/citation/feature-names')
        assert response.status_code == 200
        feature_names = response.json()['feature_names']
        assert feature_names[0] == 'token_text'
        assert 'sentence_token_relative_position' in feature_names
