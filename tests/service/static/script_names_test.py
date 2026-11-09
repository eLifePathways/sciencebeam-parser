"""Every name the page's script uses is one it has.

`node --check` reads the syntax and stops there, so a name left behind by a
rename parses and only fails when the line runs -- which, for a line that runs
once a document has been converted, is in front of whoever converted it.
"""
import re
from pathlib import Path
from typing import Set

import pytest

from sciencebeam_parser.service.assets import STATIC_DIRECTORY


SCRIPT_FILES = sorted(Path(STATIC_DIRECTORY).glob('*.js'))

KNOWN_GLOBALS = {
    'AbortController', 'Array', 'Blob', 'DOMParser', 'DataView',
    'DecompressionStream', 'Error', 'FormData', 'Infinity', 'JSON', 'Map',
    'Math', 'Number', 'Object', 'Promise', 'Request', 'Response', 'Set',
    'String', 'TextDecoder', 'TransformStream', 'URL', 'URLSearchParams',
    'Uint8Array', 'console', 'document', 'fetch', 'navigator', 'window'
}

DECLARATION_PATTERNS = [
    r'\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)',
    r'\bfunction\s+([A-Za-z_$][\w$]*)',
    r'\bclass\s+([A-Za-z_$][\w$]*)',
    r'\bcatch\s*\(\s*([A-Za-z_$][\w$]*)',
]

BINDING_LIST_PATTERNS = [
    # destructuring, import lists and every shape of parameter list
    r'\b(?:const|let|var)\s*\{([^}]*)\}',
    r'\bimport\s*\{([^}]*)\}',
    r'\bfunction\s*[A-Za-z_$\w$]*\s*\(([^)]*)\)',
    r'\(([^()]*)\)\s*=>',
    r'(?:^|[^\w.$])([A-Za-z_$][\w$]*)\s*=>',
]

USE_PATTERN = r'(?:^|[^\w.$\'"])([a-z][\w$]*)\s*\.'


def get_declared_names(script: str) -> Set[str]:
    names = set()
    for pattern in DECLARATION_PATTERNS:
        names.update(re.findall(pattern, script))
    for pattern in BINDING_LIST_PATTERNS:
        for group in re.findall(pattern, script):
            names.update(re.findall(r'[A-Za-z_$][\w$]*', group))
    return names


STRING_PATTERN = r"'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\""

TEMPLATE_PATTERN = r'`(?:\\.|[^`\\])*`'


def get_code(script: str) -> str:
    """The script with its prose taken out.

    Comments and strings carry English, and English is full of words that
    look like a name in front of a full stop.
    """
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', script, flags=re.DOTALL)
    # a template's own text goes, what it interpolates stays
    code = re.sub(
        TEMPLATE_PATTERN,
        lambda match: ' '.join(re.findall(r'\$\{([^}]*)\}', match.group(0))),
        code
    )
    return re.sub(STRING_PATTERN, ' ', code)


def get_undeclared_names(script: str) -> Set[str]:
    code = get_code(script)
    used = set(re.findall(USE_PATTERN, code))
    return used - get_declared_names(code) - KNOWN_GLOBALS


@pytest.mark.parametrize(
    'script_file', SCRIPT_FILES, ids=lambda path: path.name
)
class TestScriptNames:
    def test_should_use_no_name_it_does_not_have(self, script_file: Path):
        assert get_undeclared_names(script_file.read_text(encoding='utf-8')) == set()


class TestGetUndeclaredNames:
    def test_should_find_a_name_a_rename_left_behind(self):
        assert get_undeclared_names(
            'function show() {\n  archiveNote.textContent = "";\n}'
        ) == {'archiveNote'}

    def test_should_accept_a_name_declared_anywhere_in_the_file(self):
        assert get_undeclared_names(
            'const archiveNote = document.getElementById("x");\n'
            'function show() {\n  archiveNote.textContent = "";\n}'
        ) == set()

    def test_should_accept_a_parameter_and_a_destructured_binding(self):
        assert get_undeclared_names(
            'function show(entry, { previewRow }) {\n'
            '  entry.read();\n  previewRow.hidden = true;\n}'
        ) == set()

    def test_should_not_read_a_name_out_of_prose(self):
        assert get_undeclared_names(
            'function show() {\n'
            "  caption.textContent = 'Reading it. The zip output.';\n"
            '  // a comment mentioning archiveNote.textContent\n'
            '}\n'
            'const caption = 1;'
        ) == set()

    def test_should_still_read_a_name_a_template_interpolates(self):
        assert get_undeclared_names(
            'function show() {\n'
            '  return `a size of ${missingThing.size} bytes`;\n'
            '}'
        ) == {'missingThing'}
