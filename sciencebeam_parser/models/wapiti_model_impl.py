import os
import logging
import threading
from typing import Iterable, Optional, List, Tuple, cast

import numpy as np

from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import WapitiWrapper
from sciencebeam_trainer_delft.utils.io import copy_file
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_adapters import (
    WapitiModelAdapter
)

from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.models.model_impl import ModelImpl
from sciencebeam_parser.utils.download import download_if_url_from_alternatives
from sciencebeam_parser.utils.lazy import LazyLoaded


LOGGER = logging.getLogger(__name__)


class WapitiServiceModelAdapter(WapitiModelAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._wapiti_timeout = 20.0
        self._wapiti_timeout_counter = 0
        self._wapiti_trial_count = 10

    @staticmethod
    def load_from(
        model_path: str,
        download_manager: DownloadManager,
        wapiti_binary_path: Optional[str] = None
    ) -> 'WapitiModelAdapter':
        # overriding method to return WapitiServiceModelAdapter
        model_file_path = os.path.join(model_path, 'model.wapiti.gz')
        model_file_paths = [model_file_path, os.path.splitext(model_file_path)[0]]
        LOGGER.debug('checking for existing local model files: %r', model_file_paths)
        local_model_file_path = download_if_url_from_alternatives(
            download_manager=download_manager,
            alternative_file_url_or_path_list=model_file_paths
        )
        LOGGER.debug('local_model_file_path: %s', local_model_file_path)
        if local_model_file_path.endswith('.gz'):
            local_uncompressed_file_path = os.path.splitext(local_model_file_path)[0]
            copy_file(local_model_file_path, local_uncompressed_file_path, overwrite=False)
            local_model_file_path = local_uncompressed_file_path
        return WapitiServiceModelAdapter(
            WapitiWrapper(
                wapiti_binary_path=wapiti_binary_path
            ),
            model_file_path=local_model_file_path,
            model_path=model_path
        )

    def stop(self):
        wapiti_model = self._wapiti_model
        if wapiti_model is None:
            return
        self._wapiti_model = None
        LOGGER.info('stopping wapiti process: %s', wapiti_model.process.pid)
        wapiti_model.process.kill()

    def on_wapiti_timeout(self):
        self._wapiti_timeout_counter += 1
        LOGGER.info(
            'wapiti timeout (%s, counter=%d)',
            self._wapiti_timeout, self._wapiti_timeout_counter
        )
        self.stop()

    def _get_tag_results_with_timeout(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        prev_wapiti_timeout_counter = self._wapiti_timeout_counter
        timer = threading.Timer(self._wapiti_timeout, self.on_wapiti_timeout)
        timer.start()
        result = list(self.iter_tag_using_model(x, features, output_format))
        timer.cancel()
        if self._wapiti_timeout_counter != prev_wapiti_timeout_counter:
            raise TimeoutError('wapiti timeout received during processing')
        return result

    def _get_tag_results_with_timeout_and_retry(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        attempt = 0
        while True:
            try:
                return self._get_tag_results_with_timeout(x, features, output_format)
            except Exception as exc:  # pylint: disable=broad-except
                attempt += 1
                LOGGER.warning(
                    'received error processing data: %r, attempt=%d/%d, texts=%r',
                    exc, attempt, self._wapiti_trial_count, list(x), exc_info=True
                )
                if attempt >= self._wapiti_trial_count:
                    LOGGER.warning('final attempt, re-raising exception')
                    raise

    def iter_tag(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> Iterable[List[Tuple[str, str]]]:
        # by default, WapitiModelAdapter will run the binary for each call
        # using "iter_tag_using_model" will result in a wapiti process
        # that we communicate with via stdin / stdout
        with self._lock:
            yield from self._get_tag_results_with_timeout_and_retry(x, features, output_format)


class WapitiModelImpl(ModelImpl):
    def __init__(self, model_url: str, app_context: AppContext):
        self.model_url = model_url
        self.app_context = app_context
        self._lazy_model = LazyLoaded[WapitiModelAdapter](self._load_model)

    def __repr__(self) -> str:
        return '%s(%r, loaded=%r)' % (
            type(self).__name__, self.model_url, self._lazy_model.is_loaded
        )

    def _load_model(self) -> WapitiModelAdapter:
        model = WapitiServiceModelAdapter.load_from(
            self.model_url,
            wapiti_binary_path=self.app_context.lazy_wapiti_binary_wrapper.get_binary_path(),
            download_manager=cast(DownloadManager, self.app_context.download_manager)
        )
        LOGGER.info('loaded wapiti model: %r', self.model_url)
        return model

    @property
    def model(self) -> WapitiModelAdapter:
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
        result = model.tag(
            np.array(texts, dtype=object),
            features=np.array(features, dtype=object),
            output_format=output_format
        )
        token_count = sum(len(text) for text in texts)
        LOGGER.info(
            'predicted labels using wapiti model (document count: %d, token count: %d)',
            len(texts), token_count
        )
        return result
