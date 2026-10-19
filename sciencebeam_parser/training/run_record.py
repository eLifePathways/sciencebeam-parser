"""What one assembly run read, and what it did with it.

The per-document numbers go to the assembly quality record. This holds what that
file cannot: which inputs the run was pointed at and at what revision, and the
totals over them. A corpus read from someone else's checkout is only
reproducible if the checkout is named.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sciencebeam_parser import __version__ as parser_version
from sciencebeam_parser.training.quality.assembly import AssembledDocumentRecord


LOGGER = logging.getLogger(__name__)

RUN_RECORD_SUFFIX = '.run.json'


def get_run_record_path_for_delft_output_path(delft_output_path: str) -> str:
    return delft_output_path + RUN_RECORD_SUFFIX


def get_run_record(
    model_name: str,
    tei_source_path: str,
    raw_source_path: Optional[str],
    assembled_records: Sequence[AssembledDocumentRecord],
    source_name: Optional[str] = None,
    source_revision: Optional[str] = None,
    include_extra_columns: bool = False,
) -> Dict[str, Any]:
    refused: List[Dict[str, Any]] = [
        {
            'document_id': record.document_id,
            'reasons': list(record.verdict.exclusion_reasons),
            'matched_row_ratio': (
                round(record.matched_row_ratio, 4)
                if record.matched_row_ratio is not None else None
            ),
        }
        for record in assembled_records
        if record.verdict is not None and record.verdict.is_excluded
    ]
    unrecognised: Dict[str, int] = {}
    for record in assembled_records:
        for path, count in (record.unrecognised_path_counts or {}).items():
            unrecognised[path] = unrecognised.get(path, 0) + count
    matched = sum(record.matched_row_count for record in assembled_records)
    unmatched = sum(record.unmatched_row_count for record in assembled_records)
    return {
        'model': model_name,
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'parser_version': parser_version,
        'source': {
            'name': source_name,
            'revision': source_revision,
            'tei_source_path': tei_source_path,
            'raw_source_path': raw_source_path,
        },
        'include_extra_columns': include_extra_columns,
        'documents': {
            'read': len(assembled_records),
            'refused': len(refused),
        },
        'rows': {
            'matched': matched,
            'unmatched': unmatched,
            'uncovered': sum(
                record.uncovered_row_count for record in assembled_records
            ),
            'matched_ratio': round(matched / (matched + unmatched), 4)
            if matched + unmatched else None,
        },
        # Zero for a TEI written here, so anything else says the corpus being
        # read uses element paths this project does not.
        'unrecognised_element_paths': dict(sorted(unrecognised.items())),
        'refused_documents': refused,
    }


def write_run_record(run_record_path: str, run_record: Dict[str, Any]) -> None:
    with open(run_record_path, 'w', encoding='utf-8') as fp:
        json.dump(run_record, fp, indent=2, sort_keys=True)
        fp.write('\n')
    LOGGER.info('wrote the run record to %r', run_record_path)
