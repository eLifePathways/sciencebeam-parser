import logging

import pytest
from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_crf_lines
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.config.config import AppConfig, get_download_dir
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.external.wapiti.wrapper import LazyWapitiBinaryWrapper
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.delft_model_impl import DelftModelImpl
from sciencebeam_parser.models.header.data import HeaderDataGenerator
from sciencebeam_parser.models.model import iter_data_lines_for_model_data_iterables
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE


LOGGER = logging.getLogger(__name__)


# The biorxiv_elife header model: TF-era `model_weights.hdf5`, and no word embeddings,
# so loading it needs nothing from the embedding registry.
HEADER_MODEL_URL = (
    'https://github.com/eLifePathways/sciencebeam-models/releases/download'
    '/v0.0.1/2020-10-04-delft-grobid-header-biorxiv-no-word-embedding.tar.gz'
)

HEADER_LINE_TOKEN_TEXTS = [
    ['A', 'Study', 'of', 'Something'],
    ['Jane', 'Doe', 'and', 'John', 'Smith']
]


@pytest.fixture(name='app_context', scope='module')
def _app_context() -> AppContext:
    app_config = AppConfig.load_yaml(DEFAULT_CONFIG_FILE)
    download_manager = DownloadManager(download_dir=get_download_dir(app_config))
    return AppContext(
        app_config=app_config,
        download_manager=download_manager,
        lazy_wapiti_binary_wrapper=LazyWapitiBinaryWrapper(
            download_manager=download_manager
        )
    )


def _get_layout_document() -> LayoutDocument:
    return LayoutDocument.for_blocks([
        LayoutBlock(lines=[
            LayoutLine([LayoutToken(text) for text in line_token_texts])
            for line_token_texts in HEADER_LINE_TOKEN_TEXTS
        ])
    ])


@pytest.mark.slow
class TestDelftModelImpl:
    def test_should_load_tensorflow_era_model_and_tag_every_token(
        self, app_context: AppContext
    ):
        model_impl = DelftModelImpl(HEADER_MODEL_URL, app_context)
        data_generator = HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
        model_data_list = list(
            data_generator.iter_model_data_for_layout_document(_get_layout_document())
        )
        data_lines = list(iter_data_lines_for_model_data_iterables([model_data_list]))
        texts, features = load_data_crf_lines(data_lines)
        tag_result = model_impl.predict_labels(
            texts=texts.tolist(),
            features=features.tolist(),
            output_format=None
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert len(tag_result) == 1
        assert [token for token, _ in tag_result[0]] == list(texts[0])
        preprocessor = model_impl.model.p
        assert preprocessor is not None
        assert preprocessor.indice_tag is not None
        assert {label for _, label in tag_result[0]} <= set(preprocessor.indice_tag.values())
