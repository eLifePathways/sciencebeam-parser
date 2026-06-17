from __future__ import annotations

from benchmarks.analyze_gold_failures._screenshots import _text_similarity


class TestTextSimilarity:
    def test_exact_match_scores_1(self):
        assert _text_similarity('abc', 'abc') == 1.0

    def test_substring_match_scores_1_regardless_of_element_length(self):
        # Gold appears verbatim at the start of a much longer element text.
        # The element also contains a translation and affiliation, but that
        # should not reduce the score.
        gold = 'debateonthepaperbynamoardealmeidafilho'
        element = gold + 'debatesobreoartigodenaomardealme' + 'laboratoriodelascienciassociales'
        assert _text_similarity(gold, element) == 1.0

    def test_gold_as_suffix_scores_1(self):
        assert _text_similarity('world', 'hello world') == 1.0

    def test_absent_text_scores_0(self):
        assert _text_similarity('acknowledgements', 'introduction methods results') == 0.0

    def test_empty_gold_scores_0(self):
        assert _text_similarity('', 'some element text') == 0.0

    def test_empty_element_scores_0(self):
        assert _text_similarity('some gold value', '') == 0.0
