import json
import logging
from contextlib import contextmanager
from typing import List, Tuple
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_crf_lines
from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagLabelFormats,
    TagOutputFormats,
    iter_format_tag_result
)

from sciencebeam_parser.app.profiles import ProfileBundle
from sciencebeam_parser.models.data import (
    DEFAULT_APP_FEATURES_CONTEXT,
    DocumentFeaturesContext
)
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.service.api.routers.models import (
    ModelResponseRouterFactory,
    create_models_router
)
from tests.processors.fulltext.model_mocks import MockFullTextModels


LOGGER = logging.getLogger(__name__)


@pytest.fixture(name='mock_fulltext_models')
def _mock_fulltext_models() -> MockFullTextModels:
    return MockFullTextModels()


@pytest.fixture(name='profile_bundle')
def _profile_bundle(mock_fulltext_models: MockFullTextModels) -> ProfileBundle:
    return ProfileBundle(
        name='profile1',
        fulltext_models=mock_fulltext_models,
        fulltext_processor_config=FullTextProcessorConfig(),
        models_digest='profile1-digest'
    )


@pytest.fixture(name='sciencebeam_parser_mock')
def _sciencebeam_parser_mock(profile_bundle: ProfileBundle) -> MagicMock:
    mock = MagicMock(name='sciencebeam_parser')
    mock.app_features_context = DEFAULT_APP_FEATURES_CONTEXT
    mock.profile_registry.get_bundle.return_value = profile_bundle
    return mock


@pytest.fixture(name='test_client')
def _test_client(sciencebeam_parser_mock: MagicMock) -> TestClient:
    app = FastAPI()
    app.state.sciencebeam_parser = sciencebeam_parser_mock
    app.include_router(create_models_router(sciencebeam_parser_mock))
    return TestClient(app)


class TestHandlePostSpan:
    def test_should_name_the_document_and_the_model(
        self, monkeypatch: pytest.MonkeyPatch, profile_bundle: ProfileBundle
    ):
        """This endpoint does not go through the parser's own document span.

        It reads `source.source_path` directly, so without a span of its own the
        model calls underneath it would belong to no document.
        """
        recorded: List[Tuple[str, dict]] = []

        @contextmanager
        def recording_span(name: str, attributes=None, **_kwargs):
            recorded.append((name, dict(attributes or {})))
            yield MagicMock(name='span')

        monkeypatch.setattr(
            'sciencebeam_parser.service.api.routers.models.span', recording_span
        )
        factory = ModelResponseRouterFactory(
            name='citation',
            sequence_model_name='citation',
            pdfalto_wrapper=MagicMock(name='pdfalto_wrapper'),
            app_features_context=DEFAULT_APP_FEATURES_CONTEXT,
            model_name='citation'
        )
        monkeypatch.setattr(
            factory, '_handle_post', lambda *args, **kwargs: 'the response'
        )
        source = MagicMock(name='source')
        source.source_name = 'the-uploaded-name.pdf'
        source.source_media_type = 'application/pdf'

        assert factory.handle_post(
            source, 'json', profile_bundle
        ) == 'the response'
        assert recorded == [(
            'process_document',
            {
                'sciencebeam.document.name': 'the-uploaded-name.pdf',
                'sciencebeam.document.source_media_type': 'application/pdf',
                'sciencebeam.model.name': 'citation',
                'sciencebeam.model.output_format': 'json',
                'sciencebeam.profile.name': 'profile1',
            }
        )]


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


class TestRaggedTagResultFormatting:
    """The json formatter calls np.array(texts), which fails on a ragged list.

    `load_data_crf_lines` returns dtype=object arrays that survive that call, so
    the router has to pass those arrays through rather than a nested list — one
    `.tolist()` too early breaks /api/models/citation?output_format=json for any
    document with references of differing lengths, on any engine.
    """
    def test_should_format_sequences_of_differing_lengths_as_json(self):
        data_lines = ['a f x', 'b f x', '', 'c f x', '', 'd f x', 'e f x']
        texts, features = load_data_crf_lines(data_lines)
        tag_result = [[(token, 'O') for token in sequence] for sequence in texts.tolist()]
        content = ''.join(iter_format_tag_result(
            tag_result,
            output_format=TagOutputFormats.JSON,
            label_format=TagLabelFormats.GROBID,
            expected_tag_result=None,
            texts=texts,
            features=features,
            model_name='citation'
        ))
        assert [len(sequence) for sequence in json.loads(content)['texts']] == [2, 1, 2]

    def test_should_fail_on_a_nested_list_rather_than_an_object_array(self):
        data_lines = ['a f x', 'b f x', '', 'c f x']
        texts, features = load_data_crf_lines(data_lines)
        tag_result = [[(token, 'O') for token in sequence] for sequence in texts.tolist()]
        with pytest.raises(ValueError, match='inhomogeneous'):
            ''.join(iter_format_tag_result(
                tag_result,
                output_format=TagOutputFormats.JSON,
                label_format=TagLabelFormats.GROBID,
                expected_tag_result=None,
                texts=texts.tolist(),
                features=features,
                model_name='citation'
            ))


class TestModelsRouterProfile:
    def test_should_serve_the_deployment_profile_when_nothing_is_named(
        self, test_client: TestClient, sciencebeam_parser_mock: MagicMock
    ):
        response = test_client.get('/models/segmentation/feature-names')
        assert response.status_code == 200
        sciencebeam_parser_mock.profile_registry.get_bundle.assert_called_with(None)

    def test_should_honour_the_named_profile(
        self, test_client: TestClient, sciencebeam_parser_mock: MagicMock
    ):
        response = test_client.get(
            '/models/segmentation/feature-names', params={'profile': 'profile2'}
        )
        assert response.status_code == 200
        sciencebeam_parser_mock.profile_registry.get_bundle.assert_called_with('profile2')

    def test_should_read_the_model_from_the_requested_bundle(
        self, test_client: TestClient, sciencebeam_parser_mock: MagicMock
    ):
        """The router holds a name; which instance it names is per request."""
        other_models = MockFullTextModels()
        sciencebeam_parser_mock.profile_registry.get_bundle.return_value = ProfileBundle(
            name='profile2',
            fulltext_models=other_models,
            fulltext_processor_config=FullTextProcessorConfig(),
            models_digest='profile2-digest'
        )
        response = test_client.get(
            '/models/header/feature-names', params={'profile': 'profile2'}
        )
        assert response.status_code == 200
        assert response.json()['feature_names'] == (
            other_models.header_model.get_data_generator(
                DocumentFeaturesContext(
                    app_features_context=DEFAULT_APP_FEATURES_CONTEXT
                )
            ).feature_names
        )
