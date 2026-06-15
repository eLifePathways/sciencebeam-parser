from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.service.server import get_app_config
from sciencebeam_parser.utils.feature_importance import find_important_data
from sciencebeam_parser.utils.model_data_diff import format_model_data_diff

LOGGER = logging.getLogger(__name__)

GROBID_DEFAULT_URL = 'http://localhost:8070'
PARSER_DEFAULT_URL = 'http://localhost:8080'


def _check_service(url: str, name: str) -> None:
    try:
        r = httpx.get(f'{url}/api/isalive', timeout=5)
        r.raise_for_status()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        sys.exit(f'Error: {name} not reachable at {url}: {exc}')


def _fetch_grobid_model_data(
    pdf_path: Path,
    model_name: str,
    grobid_url: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f'{grobid_url}/api/processFulltextDocument',
            data={'debugMode': 'true', 'models': model_name},
            files={'input': (pdf_path.name, pdf_path.read_bytes(), 'application/pdf')},
        )
        r.raise_for_status()
    lines = [
        line.replace('\t', ' ')
        for line in r.text.splitlines()
        if not line.startswith('=== model:')
    ]
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _fetch_parser_model_data(
    pdf_path: Path,
    model_name: str,
    parser_url: str,
    data_path: Path,
    feature_names_path: Path,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f'{parser_url}/api/models/{model_name}',
            params={'output_format': 'data'},
            headers={'accept': 'application/json'},
            files={'input': (pdf_path.name, pdf_path.read_bytes(), 'application/pdf')},
        )
        r.raise_for_status()
        data_path.write_bytes(r.content)

        r2 = client.get(f'{parser_url}/api/models/{model_name}/feature-names')
        r2.raise_for_status()
        feature_names_path.write_bytes(r2.content)


def _load_sbparser_models(model_chain: List[str]) -> Dict[str, object]:
    config = get_app_config()
    sb_parser = ScienceBeamParser.from_config(config)
    models: Dict[str, object] = {}
    for model_name in model_chain:
        sb_name = model_name.replace('-', '_')
        try:
            models[model_name] = sb_parser.fulltext_models.get_sequence_model_by_name(sb_name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            sys.exit(f'Failed to load model {model_name!r}: {exc}')
    return models


def _write_diff(
    parser_data: Path,
    grobid_data: Path,
    feature_names_file: Path,
    diff_out: Path,
) -> None:
    feature_names = json.loads(
        feature_names_file.read_text(encoding='utf-8')
    )['feature_names']
    diff_text = format_model_data_diff(
        parser_data.read_text(encoding='utf-8'),
        grobid_data.read_text(encoding='utf-8'),
        feature_names,
    )
    diff_out.write_text(diff_text, encoding='utf-8')


def _analyze_doc_model(
    record_id: str,
    model_name: str,
    model,
    pdf_path: Path,
    grobid_url: str,
    parser_url: str,
    doc_dir: Path,
) -> Optional[dict]:
    """Fetch model data and run find-important for one doc/model pair. Returns JSON data or None."""
    grobid_data = doc_dir / 'grobid' / f'{model_name}.data'
    parser_data = doc_dir / 'sciencebeam-parser' / f'{model_name}.data'
    feature_names_file = doc_dir / 'sciencebeam-parser' / f'{model_name}.feature_names.json'
    json_out = doc_dir / f'{model_name}.find_important.json'
    diff_out = doc_dir / f'{model_name}.diff'

    if json_out.exists():
        LOGGER.info('Using cached result for %s / %s', record_id, model_name)
        if not diff_out.exists() and parser_data.exists() and grobid_data.exists():
            LOGGER.info('Generating missing diff for %s / %s', record_id, model_name)
            _write_diff(parser_data, grobid_data, feature_names_file, diff_out)
        return json.loads(json_out.read_text(encoding='utf-8'))

    if not pdf_path.exists():
        LOGGER.warning('PDF not found: %s', pdf_path)
        return None

    LOGGER.info('Fetching GROBID model data for %s / %s', record_id, model_name)
    _fetch_grobid_model_data(pdf_path, model_name, grobid_url, grobid_data)

    LOGGER.info('Fetching parser model data for %s / %s', record_id, model_name)
    _fetch_parser_model_data(pdf_path, model_name, parser_url, parser_data, feature_names_file)

    LOGGER.info('Running find-important for %s / %s', record_id, model_name)
    result = find_important_data(
        str(parser_data), str(grobid_data), str(feature_names_file), model
    )

    json_out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    _write_diff(parser_data, grobid_data, feature_names_file, diff_out)
    return result
