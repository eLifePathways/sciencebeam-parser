"""Renders pages of a PDF through a worker process.

One worker serves one document and is reused across its pages, so the
document is opened once. How many live at a time is bounded: a worker holds
a document open for as long as the request it serves, and a process
answering many at once would otherwise hold one for each.
"""

import json
import logging
import subprocess
import sys
import threading
from time import monotonic
from typing import Any, Dict, Optional, Tuple


LOGGER = logging.getLogger(__name__)


WORKER_MODULE = 'sciencebeam_parser.external.pdfium.render_worker'

DEFAULT_MAX_RENDER_WORKERS = 2
DEFAULT_RENDER_TIMEOUT = 120.0

T_Crop = Tuple[float, float, float, float]


class RenderWorkerLimit:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._semaphore = threading.BoundedSemaphore(max(1, max_workers))

    def acquire(self, timeout: float) -> bool:
        return self._semaphore.acquire(timeout=timeout)  # pylint: disable=consider-using-with

    def release(self):
        self._semaphore.release()


_limit = RenderWorkerLimit(DEFAULT_MAX_RENDER_WORKERS)
_limit_lock = threading.Lock()


def configure_max_render_workers(max_workers: int):
    """Sets how many render workers this process may have alive at once."""
    global _limit  # pylint: disable=global-statement
    with _limit_lock:
        if max_workers != _limit.max_workers:
            _limit = RenderWorkerLimit(max_workers)


def get_render_worker_limit() -> RenderWorkerLimit:
    return _limit


class PdfRenderWorker:
    def __init__(
        self,
        pdf_path: str,
        timeout: float = DEFAULT_RENDER_TIMEOUT,
        limit: Optional[RenderWorkerLimit] = None
    ):
        self.pdf_path = pdf_path
        self.timeout = timeout
        self.limit = limit if limit is not None else get_render_worker_limit()
        self._process: Optional[subprocess.Popen] = None
        self._has_slot = False
        self._failed = False

    def __enter__(self) -> 'PdfRenderWorker':
        return self

    def __exit__(self, *args):
        self.close()

    def _start(self) -> Optional[subprocess.Popen]:
        if self._process is not None:
            return self._process
        if self._failed:
            return None
        if not self._has_slot:
            wait_start = monotonic()
            if not self.limit.acquire(timeout=self.timeout):
                LOGGER.warning(
                    'no render worker available within %.0fs for %r',
                    self.timeout, self.pdf_path
                )
                self._failed = True
                return None
            self._has_slot = True
            waited = monotonic() - wait_start
            if waited > 1:
                LOGGER.info('waited %.1fs for a render worker', waited)
        self._process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, '-m', WORKER_MODULE, self.pdf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        return self._process

    def _stop_process(self):
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass

    def close(self):
        self._stop_process()
        if self._has_slot:
            self._has_slot = False
            self.limit.release()

    def __del__(self):
        # a slot held for the life of a worker wedges the process where a
        # caller forgets to give it back, so collection is the backstop
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    def _request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        process = self._start()
        if process is None or process.stdin is None or process.stdout is None:
            return None
        reply_holder: Dict[str, Any] = {}

        def _exchange():
            try:
                assert process.stdin and process.stdout
                process.stdin.write(json.dumps(request) + '\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if line:
                    reply_holder['reply'] = json.loads(line)
            except Exception as exc:  # pylint: disable=broad-except
                reply_holder['error'] = exc

        thread = threading.Thread(target=_exchange, daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            LOGGER.warning(
                'render worker did not answer within %.0fs for %r',
                self.timeout, self.pdf_path
            )
            self._failed = True
            self._stop_process()
            return None
        if 'error' in reply_holder or 'reply' not in reply_holder:
            LOGGER.warning(
                'render worker failed for %r: %r',
                self.pdf_path, reply_holder.get('error')
            )
            self._failed = True
            self._stop_process()
            return None
        return reply_holder['reply']

    def render(
        self,
        page_number: int,
        scale: float,
        image_path: str,
        crop: Optional[T_Crop] = None
    ) -> Optional[str]:
        request: Dict[str, Any] = {
            'page_number': page_number,
            'scale': scale,
            'image_path': image_path
        }
        if crop is not None:
            request['crop'] = list(crop)
        reply = self._request(request)
        if reply is None:
            return None
        if not reply.get('ok'):
            LOGGER.warning(
                'unable to render page %d of %r: %s',
                page_number, self.pdf_path, reply.get('error')
            )
            return None
        return reply['image_path']
