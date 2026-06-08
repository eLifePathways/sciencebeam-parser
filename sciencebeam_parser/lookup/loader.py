import logging
import os
from pathlib import Path
from typing import Optional

from sciencebeam_parser.app.context import DownloadManagerProtocol
from sciencebeam_parser.lookup import MergedTextLookUp, SimpleTextLookUp, TextLookUp
from sciencebeam_parser.lookup.xml_lookup import load_xml_lookup_from_file


LOGGER = logging.getLogger(__name__)


def is_xml_filename(path: str):
    name = os.path.basename(path)
    return name.lower().endswith('.xml') or name.lower().endswith('.xml.gz')


def is_wf_filename(path: str) -> bool:
    return os.path.basename(path).lower().endswith('.wf')


def load_lookup_from_text_file(path: str) -> TextLookUp:
    return SimpleTextLookUp(
        set(Path(path).read_text(encoding='utf-8').splitlines()) - {''}
    )


def load_lookup_from_wf_file(path: str) -> TextLookUp:
    # GROBID multext wordforms format: first tab-separated column is the word form
    words: set = set()
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        if line:
            word = line.split('\t', 1)[0]
            if word:
                words.add(word)
    return SimpleTextLookUp(words)


def load_lookup_from_path(
    path: Optional[str],
    download_manager: DownloadManagerProtocol
) -> Optional[TextLookUp]:
    if not path:
        return None
    local_path = download_manager.download_if_url(path)
    LOGGER.info('loading lookup from: %r (%r)', path, local_path)
    if is_xml_filename(path):
        return load_xml_lookup_from_file(local_path)
    if is_wf_filename(path):
        return load_lookup_from_wf_file(local_path)
    return load_lookup_from_text_file(local_path)


def load_lookup_from_config(
    config: Optional[dict],
    download_manager: DownloadManagerProtocol
) -> Optional[TextLookUp]:
    if not config:
        return None
    paths = config.get('paths', [])
    LOGGER.info('loading lookup from: %r', paths)
    return MergedTextLookUp([
        load_lookup_from_path(path, download_manager=download_manager)
        for path in paths
    ])
