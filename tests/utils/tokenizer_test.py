from sciencebeam_parser.utils.tokenizer import (
    get_subdigit_tokenized_tokens,
    iter_tokenized_tokens
)


class TestGetSubdigitTokenizedTokens:
    def test_letter_then_digit_splits(self):
        # e.g. DOI suffix 'e1006572' → ['e', '1006572']
        assert get_subdigit_tokenized_tokens('e1006572') == ['e', '1006572']

    def test_digit_then_letter_splits(self):
        # e.g. '295X' → ['295', 'X']
        assert get_subdigit_tokenized_tokens('295X') == ['295', 'X']

    def test_letter_then_digit_then_letter_splits_twice(self):
        # e.g. 'j8sxz' → ['j', '8', 'sxz']
        assert get_subdigit_tokenized_tokens('j8sxz') == ['j', '8', 'sxz']

    def test_all_letters_unchanged(self):
        assert get_subdigit_tokenized_tokens('Lancet') == ['Lancet']

    def test_all_digits_unchanged(self):
        assert get_subdigit_tokenized_tokens('2020') == ['2020']

    def test_xge_prefix_splits(self):
        assert get_subdigit_tokenized_tokens('xge0000511') == ['xge', '0000511']


class TestIterTokenizedTokens:
    def test_should_split_on_regular_space(self):
        assert (
            list(iter_tokenized_tokens('token1 token2'))
            == ['token1', 'token2']
        )

    def test_should_split_on_thin_space(self):
        assert (
            list(iter_tokenized_tokens('token1\u2009token2'))
            == ['token1', 'token2']
        )

    def test_should_split_on_line_feed(self):
        assert (
            list(iter_tokenized_tokens('token1\ntoken2'))
            == ['token1', 'token2']
        )

    def test_should_preserve_space(self):
        assert (
            list(iter_tokenized_tokens('token1 token2', keep_whitespace=True))
            == ['token1', ' ', 'token2']
        )

    def test_should_preserve_line_feed(self):
        assert (
            list(iter_tokenized_tokens('token1\ntoken2', keep_whitespace=True))
            == ['token1', '\n', 'token2']
        )
