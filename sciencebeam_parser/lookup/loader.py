import logging
import os
import re
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


def load_lookup_from_text_file(path: str, split_on_hyphen: bool = False) -> TextLookUp:
    words: set = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line:
            # GROBID uses StringTokenizer(line, "\t\n-") and takes the first token,
            # so compound names like "El-Am" contribute only "El" (→ "el") to the set.
            word = re.split(r'[\t-]', line)[0] if split_on_hyphen else line
            if word:
                words.add(word)
    return SimpleTextLookUp(words)


def load_lookup_from_wf_file(path: str) -> TextLookUp:
    # GROBID multext wordforms format: first tab-separated column is the word form.
    # GROBID stores words as-is (preserving case) but test_common() lowercases the query,
    # so capitalised wordforms (e.g. 'British', 'October') are effectively unreachable.
    # We replicate this by only storing already-lowercase wordforms.
    words: set = set()
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        if line:
            word = line.split('\t', 1)[0]
            if word and word == word.lower():
                words.add(word)
    return SimpleTextLookUp(words)


def load_lookup_from_path(
    path: Optional[str],
    download_manager: DownloadManagerProtocol,
    split_on_hyphen: bool = False
) -> Optional[TextLookUp]:
    if not path:
        return None
    local_path = download_manager.download_if_url(path)
    LOGGER.info('loading lookup from: %r (%r)', path, local_path)
    if is_xml_filename(path):
        return load_xml_lookup_from_file(local_path)
    if is_wf_filename(path):
        return load_lookup_from_wf_file(local_path)
    return load_lookup_from_text_file(local_path, split_on_hyphen=split_on_hyphen)


def load_lookup_from_config(
    config: Optional[dict],
    download_manager: DownloadManagerProtocol
) -> Optional[TextLookUp]:
    if not config:
        return None
    paths = config.get('paths', [])
    split_on_hyphen = config.get('split_on_hyphen', False)
    LOGGER.info('loading lookup from: %r', paths)
    return MergedTextLookUp([
        load_lookup_from_path(path, download_manager=download_manager,
                              split_on_hyphen=split_on_hyphen)
        for path in paths
    ])
