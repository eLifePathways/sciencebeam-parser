from sciencebeam_parser.lookup.phrase_match import SequencePhraseMatch


def _match(phrases, tokens):
    return SequencePhraseMatch(phrases).match_token_indices(tokens)


class TestSequencePhraseMatchSingleWord:
    def test_single_word_phrase_matches_exact_token(self):
        assert 0 in _match(['model'], ['model', 'city'])

    def test_single_word_phrase_is_case_insensitive(self):
        assert 0 in _match(['Model'], ['MODEL', 'city'])

    def test_single_word_phrase_does_not_match_different_token(self):
        assert _match(['model'], ['city']) == set()

    def test_single_word_at_end_of_stream_is_matched(self):
        assert 1 in _match(['model'], ['city', 'model'])

    def test_only_matched_token_is_in_result(self):
        result = _match(['model'], ['city', 'model', 'data'])
        assert result == {1}


class TestSequencePhraseMatchMultiWord:
    def test_two_word_phrase_matches_consecutive_tokens(self):
        result = _match(['New York'], ['New', 'York', 'city'])
        assert result == {0, 1}

    def test_two_word_phrase_is_case_insensitive(self):
        result = _match(['New York'], ['new', 'york', 'city'])
        assert result == {0, 1}

    def test_two_word_phrase_does_not_match_partial(self):
        # Only 'New' present, 'York' missing → no match
        result = _match(['New York'], ['new', 'city'])
        assert result == set()

    def test_delimiter_between_phrase_words_is_skipped(self):
        # GROBID skips delimiter tokens (like ',') when matching
        result = _match(['Model Colorado'], ['Model', ',', 'Colorado'])
        # span [0,2] inclusive: comma at position 1 is also marked
        assert result == {0, 1, 2}


class TestSequencePhraseMatchOverlapping:
    def test_shorter_and_longer_phrase_both_recorded(self):
        # 'Model' is a standalone entry AND 'Model Town' is a two-word entry.
        result = _match(['Model', 'Model Town'], ['Model', 'Town', 'data'])
        # 'Model' alone → [0,0]; 'Model Town' → [0,1]
        assert 0 in result
        assert 1 in result

    def test_multiple_independent_matches(self):
        result = _match(['Method', 'model'], ['Method', 'based', 'on', 'model'])
        assert result == {0, 3}


class TestSequencePhraseMatchDelimiters:
    def test_delimiter_only_tokens_are_not_matched(self):
        result = _match([','], [',', '.', ';'])
        assert result == set()

    def test_delimiter_between_non_matching_words_does_not_create_match(self):
        result = _match(['foo bar'], ['foo', ',', 'baz'])
        assert result == set()

    def test_standalone_on_uppercased_entry_matches_lowercase(self):
        # 'ON' is in location.txt; matching must be case-insensitive
        result = _match(['ON'], ['based', 'on', 'a'])
        assert result == {1}
