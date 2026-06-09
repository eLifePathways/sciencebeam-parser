from sciencebeam_parser.lookup import MergedTextLookUp, SimpleTextLookUp


class TestSimpleTextLookUp:
    def test_contains_exact_match(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains('word') is True

    def test_not_contains_unknown(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains('other') is False

    def test_contains_case_insensitive(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains('Word') is True
        assert lookup.contains('WORD') is True

    def test_contains_entry_stored_uppercase(self):
        lookup = SimpleTextLookUp({'WORD'})
        assert lookup.contains('word') is True

    def test_not_contains_trailing_comma(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains('word,') is False

    def test_leading_punctuation_not_matched(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains(',word') is False

    def test_empty_string(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains('') is False

    def test_punctuation_only(self):
        lookup = SimpleTextLookUp({'word'})
        assert lookup.contains(',') is False


class TestMergedTextLookUp:
    def test_contains_when_first_lookup_matches(self):
        lookup = MergedTextLookUp([SimpleTextLookUp({'a'}), SimpleTextLookUp({'b'})])
        assert lookup.contains('a') is True

    def test_contains_when_second_lookup_matches(self):
        lookup = MergedTextLookUp([SimpleTextLookUp({'a'}), SimpleTextLookUp({'b'})])
        assert lookup.contains('b') is True

    def test_not_contains_when_none_match(self):
        lookup = MergedTextLookUp([SimpleTextLookUp({'a'}), SimpleTextLookUp({'b'})])
        assert lookup.contains('c') is False

    def test_ignores_none_entries(self):
        lookup = MergedTextLookUp([None, SimpleTextLookUp({'word'})])
        assert lookup.contains('word') is True

    def test_empty_lookup_list(self):
        lookup = MergedTextLookUp([])
        assert lookup.contains('word') is False
