from benchmarks.matched_property_scoring import (
    MATCHED_PROPERTY_SCORING_TYPE,
    PROPERTY_SEPARATOR,
    iter_matched_properties,
)


PORTUGUESE_1 = 'Resumo Objetivo avaliar o consumo alimentar de adolescentes praticantes'
ENGLISH_1 = 'Objective to evaluate food consumption of adolescents who practise judo'
ENGLISH_2 = 'Background the incidence of the disease has risen steadily over the decade'


def _value(prop: str, text: str) -> str:
    return prop + PROPERTY_SEPARATOR + text


class TestIterMatchedProperties:
    def test_should_pair_by_the_text_rather_than_by_position(self):
        assert iter_matched_properties(
            [_value('pt', PORTUGUESE_1), _value('en', ENGLISH_1)],
            [_value('en', ENGLISH_1), _value('pt', PORTUGUESE_1)]
        ) == [('pt', 'pt'), ('en', 'en')]

    def test_should_leave_out_a_gold_value_declaring_no_property(self):
        assert iter_matched_properties(
            [_value('', PORTUGUESE_1), _value('en', ENGLISH_1)],
            [_value('pt', PORTUGUESE_1), _value('en', ENGLISH_1)]
        ) == [('en', 'en')]

    def test_should_report_an_empty_property_where_the_value_was_not_found(self):
        assert iter_matched_properties(
            [_value('pt', PORTUGUESE_1)],
            [_value('en', ENGLISH_2)]
        ) == [('pt', '')]

    def test_should_report_an_empty_property_where_nothing_was_predicted(self):
        assert iter_matched_properties([_value('pt', PORTUGUESE_1)], []) == [('pt', '')]

    def test_should_report_an_empty_property_where_the_prediction_declares_none(self):
        assert iter_matched_properties(
            [_value('pt', PORTUGUESE_1)],
            [_value('', PORTUGUESE_1)]
        ) == [('pt', '')]


class TestMatchedPropertyScoringType:
    def _score(self, expected, actual):
        return MATCHED_PROPERTY_SCORING_TYPE.score(expected, actual, measures=['exact'])['exact']

    def test_should_score_a_matching_property(self):
        score = self._score([_value('pt', PORTUGUESE_1)], [_value('pt', PORTUGUESE_1)])
        assert score['true_positive'] == 1
        assert score['n_matched_pairs'] == 1

    def test_should_not_score_a_property_the_gold_does_not_declare(self):
        score = self._score([_value('', PORTUGUESE_1)], [_value('pt', PORTUGUESE_1)])
        assert score['n_matched_pairs'] == 0
        assert score['n_expected_properties'] == 0

    def test_should_count_a_wrong_property_against_the_prediction(self):
        score = self._score([_value('pt', PORTUGUESE_1)], [_value('es', PORTUGUESE_1)])
        assert score['true_positive'] == 0

    def test_should_credit_each_pair_rather_than_the_document_as_a_whole(self):
        score = self._score(
            [_value('pt', PORTUGUESE_1), _value('en', ENGLISH_1)],
            [_value('pt', PORTUGUESE_1), _value('de', ENGLISH_1)]
        )
        assert score['true_positive'] == 1
        assert score['false_positive'] == 1
        assert score['score'] == 0.5

    def test_should_count_pairs_rather_than_documents(self):
        score = self._score(
            [_value('pt', PORTUGUESE_1), _value('en', ENGLISH_1)],
            [_value('pt', PORTUGUESE_1), _value('en', ENGLISH_1)]
        )
        assert score['n_matched_pairs'] == 2
        assert score['n_expected_properties'] == 2
