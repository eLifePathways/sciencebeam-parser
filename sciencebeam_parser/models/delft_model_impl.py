import logging
from typing import Optional, List, Tuple

from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager
from sciencebeam_trainer_delft.sequence_labelling.wrapper import Sequence

from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.models.model_impl import ModelImpl
from sciencebeam_parser.utils.lazy import LazyLoaded


LOGGER = logging.getLogger(__name__)


REGISTRY_REGISTRY_PATH = 'delft/resources-registry.json'


class DelftModelImpl(ModelImpl):
    def __init__(self, model_url: str, app_context: AppContext):
        self.model_url = model_url
        self.app_context = app_context
        self._lazy_model = LazyLoaded[Sequence](self._load_model)

    def __repr__(self) -> str:
        return '%s(%r, loaded=%r)' % (
            type(self).__name__, self.model_url, self._lazy_model.is_loaded
        )

    def _load_model(self) -> Sequence:
        embedding_manager = EmbeddingManager(
            path=REGISTRY_REGISTRY_PATH,
            download_manager=self.app_context.download_manager
        )
        model = Sequence(
            'dummy-model',
            embedding_manager=embedding_manager
        )
        model.load_from(self.model_url)
        LOGGER.info('loaded delft model: %r', self.model_url)
        return model

    @property
    def model(self) -> Sequence:
        return self._lazy_model.get()

    def preload(self):
        self._lazy_model.get()

    def predict_labels(
        self,
        texts: List[List[str]],
        features: List[List[List[str]]],
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        model = self.model
        return model.tag(texts, features=features, output_format=output_format)
