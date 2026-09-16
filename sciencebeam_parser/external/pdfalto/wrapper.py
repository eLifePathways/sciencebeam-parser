import logging
import os
import stat
from typing import Optional

import pdfalto

from sciencebeam_parser.utils.background_process import exec_with_logging


LOGGER = logging.getLogger(__name__)


PDFALTO_VERSION = f'v{pdfalto.__version__}'


def get_default_pdfalto_binary_path() -> str:
    """Path of the pdfalto executable shipped by the ``pdfalto`` wheel.

    The wheel is platform specific (linux/mac, x86_64/arm64), so this replaces
    the per-platform GitHub release download. The ``pdfalto`` package honours
    ``PDFALTO_BINARY`` for pointing at a locally built executable instead.
    """
    return str(pdfalto.binary_path())


class PdfAltoWrapper:
    def __init__(self, binary_path: str):
        self.binary_path = binary_path

    def ensure_executable(self):
        if os.access(self.binary_path, os.X_OK):
            return
        st = os.stat(self.binary_path)
        os.chmod(self.binary_path, st.st_mode | stat.S_IEXEC)

    def get_command(
        self,
        pdf_path: str,
        output_path: str,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ):
        command = [
            self.binary_path,
            '-noImageInline',
            '-fullFontName',
            '-noLineNumbers',
            '-discardClippedText'
        ]
        if first_page:
            command.extend(['-f', str(first_page)])
        if last_page:
            command.extend(['-l', str(last_page)])
        command.extend([
            pdf_path,
            output_path
        ])
        return command

    def convert_pdf_to_pdfalto_xml(self,  *args, **kwargs):
        command = self.get_command(*args, **kwargs)
        LOGGER.info('command: %s', command)
        LOGGER.info('command str: %s', ' '.join(command))
        with exec_with_logging(command, logging_prefix='pdfalto') as _:
            pass
