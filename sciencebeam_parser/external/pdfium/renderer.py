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


class PdfRenderWorker:
    def __init__(
        self,
        pdf_path: str,
        timeout: float = DEFAULT_RENDER_TIMEOUT,
        limit: Optional[RenderWorkerLimit] = None
    ):
        self.pdf_path = pdf_path
        self.timeout = timeout
        self.limit = limit
        self._process: Optional[subprocess.Popen] = None
        self._connection: Optional[multiprocessing.connection.Connection] = None
        self._has_slot = False
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
        if self.limit is not None and not self._has_slot:
            wait_start = monotonic()
            if not self.limit.acquire(timeout=self.timeout):
                LOGGER.warning(
                    'no render worker available within %.0fs for %r',
                    self.timeout, self.pdf_path
                )
                self._failed = True
                return False
            self._has_slot = True
            waited = monotonic() - wait_start
            if waited > 1:
                LOGGER.info('waited %.1fs for a render worker', waited)
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
        if self._has_slot and self.limit is not None:
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
