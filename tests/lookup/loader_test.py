from pathlib import Path

from sciencebeam_parser.lookup import SimpleTextLookUp
from sciencebeam_parser.lookup.loader import (
    is_wf_filename,
    load_lookup_from_text_file,
    load_lookup_from_wf_file,
)


class TestIsWfFilename:
    def test_should_return_true_for_wf_extension(self):
        assert is_wf_filename('english.wf') is True

    def test_should_return_true_for_path_with_wf_extension(self):
        assert is_wf_filename('/some/path/english.wf') is True

    def test_should_return_false_for_txt_extension(self):
        assert is_wf_filename('words.txt') is False

    def test_should_return_false_for_xml_extension(self):
        assert is_wf_filename('data.xml') is False


class TestLoadLookupFromWfFile:
    def test_should_load_word_from_first_column(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text('innovation\t=\tNcns-\n', encoding='latin-1')
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert isinstance(lookup, SimpleTextLookUp)
        assert lookup.contains('innovation') is True

    def test_should_ignore_remaining_columns(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text('battery\t=\tNcns-\n', encoding='latin-1')
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert lookup.contains('battery\t=\tNcns-') is False
        assert lookup.contains('battery') is True

    def test_should_be_case_insensitive(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text('Innovation\t=\tNcns-\n', encoding='latin-1')
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert lookup.contains('Innovation') is True
        assert lookup.contains('innovation') is True

    def test_should_load_multiple_word_forms(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text(
            'recycling\t=\tNcnsi\n'
            'challenges\tchallenge\tNcnp-\n'
            'navigating\tnavigate\tVvpp--\n',
            encoding='latin-1'
        )
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert lookup.contains('recycling') is True
        assert lookup.contains('challenges') is True
        assert lookup.contains('navigating') is True

    def test_should_return_false_for_absent_word(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text('battery\t=\tNcns-\n', encoding='latin-1')
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert lookup.contains('Patil') is False

    def test_should_skip_blank_lines(self, tmp_path: Path):
        wf_file = tmp_path / 'english.wf'
        wf_file.write_text('battery\t=\tNcns-\n\ninnovation\t=\tNcns-\n', encoding='latin-1')
        lookup = load_lookup_from_wf_file(str(wf_file))
        assert lookup.contains('battery') is True
        assert lookup.contains('innovation') is True


class TestLoadLookupFromTextFile:
    def test_should_load_plain_name(self, tmp_path: Path):
        names_file = tmp_path / 'names.family'
        names_file.write_text('Smith\nJones\n', encoding='utf-8')
        lookup = load_lookup_from_text_file(str(names_file))
        assert lookup.contains('Smith') is True
        assert lookup.contains('Jones') is True

    def test_without_split_on_hyphen_stores_whole_line(self, tmp_path: Path):
        names_file = tmp_path / 'names.family'
        names_file.write_text('El-Am\n', encoding='utf-8')
        lookup = load_lookup_from_text_file(str(names_file), split_on_hyphen=False)
        assert lookup.contains('El-Am') is True
        assert lookup.contains('El') is False

    def test_split_on_hyphen_takes_first_component(self, tmp_path: Path):
        # Matches GROBID's StringTokenizer(line, "\t\n-") behaviour:
        # "El-Am" → first token "El" → stored as "el" (SimpleTextLookUp lowercases).
        names_file = tmp_path / 'names.family'
        names_file.write_text('El-Am\n', encoding='utf-8')
        lookup = load_lookup_from_text_file(str(names_file), split_on_hyphen=True)
        assert lookup.contains('El') is True
        assert lookup.contains('el') is True
        assert lookup.contains('El-Am') is False
        assert lookup.contains('Am') is False

    def test_split_on_hyphen_handles_tab_separator(self, tmp_path: Path):
        names_file = tmp_path / 'names.family'
        names_file.write_text('Smith\tother\n', encoding='utf-8')
        lookup = load_lookup_from_text_file(str(names_file), split_on_hyphen=True)
        assert lookup.contains('Smith') is True
        assert lookup.contains('other') is False
