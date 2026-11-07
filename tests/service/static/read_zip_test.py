import base64
import json
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import pytest


HARNESS_FILE = Path(__file__).parent / 'read_zip.mjs'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the archive reader that the page runs in a browser'
)


def get_zip_bytes(
    entries: Dict[str, str],
    compression: int = zipfile.ZIP_STORED,
    directories: Optional[list] = None
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=compression) as zip_file:
        for name in directories or []:
            zip_file.writestr(name, '')
        for name, content in entries.items():
            zip_file.writestr(name, content)
    return buffer.getvalue()


def read_zip_entries(zip_bytes: bytes) -> dict:
    completed_process = subprocess.run(
        ['node', str(HARNESS_FILE)],
        input=base64.b64encode(zip_bytes).decode('ascii'),
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(completed_process.stdout)


def get_text_by_name(zip_bytes: bytes) -> Dict[str, str]:
    result = read_zip_entries(zip_bytes)
    assert 'error' not in result, result.get('error')
    return {entry['name']: entry['text'] for entry in result['entries']}


class TestReadZipEntries:
    def test_should_read_a_stored_entry(self):
        assert get_text_by_name(get_zip_bytes({'jats.xml': '<article/>'})) == {
            'jats.xml': '<article/>'
        }

    def test_should_read_a_deflated_entry(self):
        content = '<article>' + ('a lot of repeated text ' * 200) + '</article>'
        assert get_text_by_name(
            get_zip_bytes({'jats.xml': content}, compression=zipfile.ZIP_DEFLATED)
        ) == {'jats.xml': content}

    def test_should_read_every_entry_the_archive_carries(self):
        text_by_name = get_text_by_name(get_zip_bytes({
            'jats.xml': '<article/>',
            'image1.png': 'first',
            'image2.png': 'second'
        }))
        assert text_by_name == {
            'jats.xml': '<article/>',
            'image1.png': 'first',
            'image2.png': 'second'
        }

    def test_should_read_an_archive_that_carries_a_comment(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            zip_file.comment = b'a comment on the archive'
            zip_file.writestr('jats.xml', '<article/>')
        assert get_text_by_name(buffer.getvalue()) == {'jats.xml': '<article/>'}

    def test_should_leave_out_the_directories(self):
        text_by_name = get_text_by_name(get_zip_bytes(
            {'assets/image1.png': 'first'}, directories=['assets/']
        ))
        assert text_by_name == {'assets/image1.png': 'first'}

    def test_should_read_a_name_that_is_not_ascii(self):
        assert get_text_by_name(get_zip_bytes({'figüre.png': 'content'})) == {
            'figüre.png': 'content'
        }

    def test_should_say_what_is_wrong_with_something_that_is_not_a_zip(self):
        result = read_zip_entries(b'not a zip at all')
        assert 'error' in result
        assert 'not a zip' in result['error']

    def test_should_refuse_an_encrypted_entry_by_name(self):
        zip_bytes = bytearray(get_zip_bytes({'jats.xml': '<article/>'}))
        central_directory = zip_bytes.rfind(b'PK\x01\x02')
        zip_bytes[central_directory + 8] |= 1
        result = read_zip_entries(bytes(zip_bytes))
        assert result.get('error') == 'jats.xml is encrypted'
