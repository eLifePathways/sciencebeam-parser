from pathlib import Path

import pytest

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_parser.config.config import get_download_dir
from sciencebeam_parser.external.pdfalto.wrapper import (
    PdfAltoWrapper,
    get_default_pdfalto_binary_path
)
from sciencebeam_parser.utils.download import download_with_zip_path_support


EXAMPLE_PDF_PATH = 'test-data/minimal-example.pdf'


@pytest.fixture(name='pdfalto_wrapper', scope='session')
def _pdfalto_wrapper(sciencebeam_parser_config: dict) -> PdfAltoWrapper:
    configured_path = sciencebeam_parser_config['pdfalto'].get('path')
    if configured_path:
        download_manager = DownloadManager(download_dir=get_download_dir(
            sciencebeam_parser_config
        ))
        binary_path = download_with_zip_path_support(download_manager, configured_path)
    else:
        binary_path = get_default_pdfalto_binary_path()
    pdfalto_wrapper = PdfAltoWrapper(binary_path)
    pdfalto_wrapper.ensure_executable()
    return pdfalto_wrapper


class TestPdfAltoWrapper:
    def test_should_convert_example_document(
        self,
        pdfalto_wrapper: PdfAltoWrapper,
        tmp_path: Path
    ):
        output_path = tmp_path / 'test.lxml'
        pdfalto_wrapper.convert_pdf_to_pdfalto_xml(
            EXAMPLE_PDF_PATH,
            str(output_path)
        )
        assert output_path.exists()
