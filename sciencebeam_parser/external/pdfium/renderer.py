"""Renders pages of a PDF through a worker process.

One worker serves one document and is reused across its pages, so the
document is opened once. A `RenderWorkerLimit` bounds how many live at a
time, since a worker holds a document open for as long as the request it
serves; whoever owns the process passes one in, and a caller rendering for
itself needs none.
"""

import logging
import multiprocessing.connection
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from time import monotonic
from typing import Any, Dict, Iterator, Optional, Tuple


LOGGER = logging.getLogger(__name__)


WORKER_MODULE = 'sciencebeam_parser.external.pdfium.render_worker'

DEFAULT_MAX_RENDER_WORKERS = 2
DEFAULT_RENDER_TIMEOUT = 120.0

T_Crop = Tuple[float, float, float, float]


class PdfRenderWorker:
    def __init__(
        self,
        pdf_path: str,
        timeout: float = DEFAULT_RENDER_TIMEOUT
    ):
        self.pdf_path = pdf_path
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._connection: Optional[multiprocessing.connection.Connection] = None
        self._failed = False

    def __enter__(self) -> 'PdfRenderWorker':
        return self

    def __exit__(self, *args):
        self.close()

    def _start(self) -> bool:
        if self._connection is not None:
            return True
        if self._failed:
            return False
        parent_socket, child_socket = socket.socketpair()
        try:
            self._process = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    sys.executable, '-m', WORKER_MODULE,
                    self.pdf_path, str(child_socket.fileno())
                ],
                pass_fds=(child_socket.fileno(),),
                stderr=subprocess.PIPE,
                text=True
            )
        except OSError:
            LOGGER.warning(
                'unable to start a render worker for %r', self.pdf_path, exc_info=True
            )
            parent_socket.close()
            child_socket.close()
            self._failed = True
            return False
        finally:
            child_socket.close()
        self._connection = multiprocessing.connection.Connection(parent_socket.detach())
        return True

    def _stop(self):
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.kill()
            # nothing else reads these, the protocol having a socket of its own
            _, error_output = process.communicate(timeout=5)
        except Exception:  # pylint: disable=broad-except
            return
        if error_output and error_output.strip():
            LOGGER.warning(
                'render worker for %r said: %s',
                self.pdf_path, error_output.strip().splitlines()[-1]
            )

    def close(self):
        self._stop()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    def _request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._start():
            return None
        connection = self._connection
        assert connection is not None
        try:
            connection.send(request)
            if not connection.poll(self.timeout):
                LOGGER.warning(
                    'render worker did not answer within %.0fs for %r',
                    self.timeout, self.pdf_path
                )
                self._failed = True
                self._stop()
                return None
            return connection.recv()
        except (EOFError, OSError) as exc:
            LOGGER.warning('render worker failed for %r: %r', self.pdf_path, exc)
            self._failed = True
            self._stop()
            return None

    def get_page_count(self) -> Optional[int]:
        reply = self._request({'page_count': True})
        if reply is None or not reply.get('ok'):
            return None
        return reply.get('page_count')

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
        return reply.get('image_path')


class PdfRenderWorkerPool:
    """Hands out render workers, no more than `max_workers` of them at a time.

    A worker holds a document open for as long as it serves one, so what is
    bounded is documents in flight. Asking when none is free waits; waiting
    longer than `timeout` gives no worker rather than no end of waiting.
    """

    def __init__(
        self,
        max_workers: int,
        timeout: float = DEFAULT_RENDER_TIMEOUT
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self._semaphore = threading.BoundedSemaphore(max(1, max_workers))

    @contextmanager
    def worker(self, pdf_path: str) -> Iterator[Optional[PdfRenderWorker]]:
        wait_start = monotonic()
        if not self._semaphore.acquire(timeout=self.timeout):  # noqa pylint: disable=consider-using-with
            LOGGER.warning(
                'no render worker free within %.0fs for %r', self.timeout, pdf_path
            )
            yield None
            return
        waited = monotonic() - wait_start
        if waited > 1:
            LOGGER.info('waited %.1fs for a render worker', waited)
        worker = PdfRenderWorker(pdf_path, timeout=self.timeout)
        try:
            yield worker
        finally:
            worker.close()
            self._semaphore.release()


@contextmanager
def open_render_worker(
    pdf_path: str,
    timeout: float = DEFAULT_RENDER_TIMEOUT,
    pool: Optional[PdfRenderWorkerPool] = None
) -> Iterator[Optional[PdfRenderWorker]]:
    """A worker from `pool` where there is one, otherwise one of its own."""
    if pool is not None:
        with pool.worker(pdf_path) as worker:
            yield worker
        return
    worker = PdfRenderWorker(pdf_path, timeout=timeout)
    try:
        yield worker
    finally:
        worker.close()
