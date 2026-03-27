import logging
import os
import platform
import stat
import sys
from typing import Optional

from sciencebeam_parser.utils.background_process import exec_with_logging


LOGGER = logging.getLogger(__name__)


PDFALTO_VERSION = 'v0.6.0'
_PDFALTO_BASE_URL = (
    f'https://github.com/kermitt2/pdfalto/releases/download/{PDFALTO_VERSION}'
)


def get_default_pdfalto_url() -> str:
    machine = platform.machine().lower()
    if sys.platform == 'darwin':
        os_name = 'mac'
    elif sys.platform.startswith('linux'):
        os_name = 'linux'
    else:
        raise RuntimeError(f'Unsupported platform: {sys.platform!r}')
    arch = 'arm64' if machine in ('arm64', 'aarch64') else '64'
    zip_name = f'pdfalto-bin-{os_name}-{arch}.zip'
    internal_path = f'pdfalto/{os_name}/{arch}/pdfalto'
    return f'{_PDFALTO_BASE_URL}/{zip_name}!/{internal_path}'


class PdfAltoWrapper:
    def __init__(self, binary_path: str):
        self.binary_path = binary_path

    def ensure_executable(self):
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
            '-noLineNumbers'
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
