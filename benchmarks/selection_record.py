"""What a run selected, recorded beside its output.

Two runs with the same digest read the same documents, with the same bytes, from
the same pinned sources.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping

RECORD_KEYS = ("record_id", "repo_id", "xml_sha256", "pdf_sha256")


def selection_record(
    sources: Mapping[str, Any], records: Iterable[Mapping[str, str]]
) -> Dict[str, Any]:
    documents: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        documents.setdefault(record["corpus"], []).append(
            {key: record[key] for key in RECORD_KEYS if key in record}
        )
    payload = json.dumps({"sources": sources, "documents": documents}, sort_keys=True)
    return {
        "documents": documents,
        "selection_digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
