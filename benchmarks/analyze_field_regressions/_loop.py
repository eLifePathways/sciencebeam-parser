from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ._fetch import _analyze_doc_model
from ._types import RegressionCase

LOGGER = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 0  # 0 = auto: max(2, cpu_count)


def _resolve_concurrency(concurrency: int) -> int:
    if concurrency == 0:
        return max(2, os.cpu_count() or 2)
    return concurrency


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.1f}h'
    if seconds >= 60:
        return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'
    return f'{seconds:.0f}s'


class _AnalysisProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self._done = 0
        self._lock = threading.Lock()
        self._t_start = time.monotonic()

    def record(
        self, corpus: str, record_id: str, model_name: str, elapsed_ms: int, ok: bool
    ) -> None:
        with self._lock:
            self._done += 1
            done = self._done
        elapsed = time.monotonic() - self._t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - done
        eta = _format_eta(remaining / rate) if rate > 0 else '?'
        print(
            f'[{done}/{self.total}] {corpus}/{record_id} {model_name}'
            f' {"ok" if ok else "err"} {elapsed_ms}ms'
            f' | {rate:.1f}/s | ~{eta} left',
            file=sys.stderr,
        )


def _run_analysis_loop(  # pylint: disable=too-many-locals
    cases: List[RegressionCase],
    model_chain: List[str],
    sbparser_models: Dict[str, object],
    grobid_url: str,
    parser_url: str,
    data_dir: Path,
    split: str,
    out_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Dict[str, List[Tuple[str, Optional[dict]]]]:
    by_doc_dir = out_dir / 'by-doc'
    concurrency = _resolve_concurrency(concurrency)
    work = [
        (case, model_name)
        for case in cases
        for model_name in model_chain
    ]
    print(
        f'Analyzing {len(work)} tasks (concurrency={concurrency})',
        file=sys.stderr,
    )
    progress = _AnalysisProgress(total=len(work))

    def _run_one(
        case: RegressionCase, model_name: str
    ) -> Tuple[Optional[dict], int]:
        t0 = time.monotonic()
        result = None
        model = sbparser_models.get(model_name)
        if model is not None:
            pdf_path = data_dir / split / case.corpus / f'{case.record_id}.pdf'
            doc_dir = by_doc_dir / case.corpus / case.record_id
            try:
                result = _analyze_doc_model(
                    case.record_id, model_name, model,
                    pdf_path, grobid_url, parser_url,
                    doc_dir=doc_dir,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    'Failed %s/%s model=%s: %s',
                    case.corpus, case.record_id, model_name, exc,
                )
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        progress.record(case.corpus, case.record_id, model_name, elapsed_ms, result is not None)
        return result, elapsed_ms

    keyed: Dict[Tuple[str, str, str], Optional[dict]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(_run_one, case, model_name): (case, model_name)
            for case, model_name in work
        }
        for future in as_completed(future_map):
            case, model_name = future_map[future]
            result, _ = future.result()
            keyed[(model_name, case.corpus, case.record_id)] = result

    return {
        model_name: [
            (case.record_id, keyed.get((model_name, case.corpus, case.record_id)))
            for case in cases
        ]
        for model_name in model_chain
    }
