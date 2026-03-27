from unittest.mock import patch

from sciencebeam_parser.external.pdfalto.wrapper import (
    PDFALTO_VERSION,
    get_default_pdfalto_url
)


class TestGetDefaultPdfaltoUrl:
    def test_linux_x86_64(self):
        with patch('sys.platform', 'linux'), \
                patch('platform.machine', return_value='x86_64'):
            url = get_default_pdfalto_url()
        assert url == (
            f'https://github.com/kermitt2/pdfalto/releases/download/{PDFALTO_VERSION}'
            f'/pdfalto-bin-linux-64.zip!/pdfalto/linux/64/pdfalto'
        )

    def test_linux_aarch64(self):
        with patch('sys.platform', 'linux'), \
                patch('platform.machine', return_value='aarch64'):
            url = get_default_pdfalto_url()
        assert url == (
            f'https://github.com/kermitt2/pdfalto/releases/download/{PDFALTO_VERSION}'
            f'/pdfalto-bin-linux-arm64.zip!/pdfalto/linux/arm64/pdfalto'
        )

    def test_macos_x86_64(self):
        with patch('sys.platform', 'darwin'), \
                patch('platform.machine', return_value='x86_64'):
            url = get_default_pdfalto_url()
        assert url == (
            f'https://github.com/kermitt2/pdfalto/releases/download/{PDFALTO_VERSION}'
            f'/pdfalto-bin-mac-64.zip!/pdfalto/mac/64/pdfalto'
        )

    def test_macos_arm64(self):
        with patch('sys.platform', 'darwin'), \
                patch('platform.machine', return_value='arm64'):
            url = get_default_pdfalto_url()
        assert url == (
            f'https://github.com/kermitt2/pdfalto/releases/download/{PDFALTO_VERSION}'
            f'/pdfalto-bin-mac-arm64.zip!/pdfalto/mac/arm64/pdfalto'
        )
