import json

from sciencebeam_parser.training.quality.review import (
    ReviewStanding,
    get_review_standing,
    get_verdict_file_path,
    read_review_standing
)


DOCUMENT_ID_1 = 'document1'
HASH_1 = 'v1:sha256:aaaa'
HASH_2 = 'v1:sha256:bbbb'


def _write_record_and_verdict(tmp_path, verdict_json_dict=None, model='segmentation'):
    model_path = tmp_path / 'train' / 'ore' / model
    record_file_path = model_path / 'quality' / f'{DOCUMENT_ID_1}.{model}.quality.json'
    record_file_path.parent.mkdir(parents=True)
    record_file_path.write_text('{}', encoding='utf-8')
    if verdict_json_dict is not None:
        verdict_file_path = model_path / 'verdict' / f'{DOCUMENT_ID_1}.{model}.verdict.json'
        verdict_file_path.parent.mkdir(parents=True)
        verdict_file_path.write_text(
            json.dumps(verdict_json_dict), encoding='utf-8'
        )
    return str(record_file_path)


class TestGetVerdictFilePath:
    def test_should_find_the_verdict_beside_the_record(self):
        assert get_verdict_file_path(
            '/data/train/ore/segmentation/quality/document1.segmentation.quality.json'
        ) == (
            '/data/train/ore/segmentation/verdict/document1.segmentation.verdict.json'
        )


class TestGetReviewStanding:
    def test_should_report_a_verdict_about_the_content_the_document_holds(self):
        assert get_review_standing(
            {'verdict': 'accepted', 'labelled_content_hash': HASH_1}, HASH_1
        ) == ReviewStanding.ACCEPTED

    def test_should_report_a_rejection_the_same_way(self):
        assert get_review_standing(
            {'verdict': 'rejected', 'labelled_content_hash': HASH_1}, HASH_1
        ) == ReviewStanding.REJECTED

    def test_should_lapse_a_verdict_the_document_has_moved_under(self):
        # Regenerating a document moves the pin; what was judged is no longer
        # what is there.
        assert get_review_standing(
            {'verdict': 'accepted', 'labelled_content_hash': HASH_1}, HASH_2
        ) == ReviewStanding.UNREVIEWED

    def test_should_report_a_document_with_no_pin_as_unreviewed(self):
        # A model with no span record has nothing a verdict could be about.
        assert get_review_standing(
            {'verdict': 'accepted', 'labelled_content_hash': HASH_1}, None
        ) == ReviewStanding.UNREVIEWED

    def test_should_report_comments_without_a_decision_as_unreviewed(self):
        assert get_review_standing(
            {'comments': [{'text': 'wrong'}], 'labelled_content_hash': HASH_1}, HASH_1
        ) == ReviewStanding.UNREVIEWED


class TestReadReviewStanding:
    def test_should_read_the_verdict_beside_a_record(self, tmp_path):
        record_file_path = _write_record_and_verdict(
            tmp_path, {'verdict': 'rejected', 'labelled_content_hash': HASH_1}
        )
        assert read_review_standing(
            record_file_path, HASH_1
        ) == ReviewStanding.REJECTED

    def test_should_report_a_corpus_with_no_verdicts_as_unreviewed(self, tmp_path):
        record_file_path = _write_record_and_verdict(tmp_path)
        assert read_review_standing(
            record_file_path, HASH_1
        ) == ReviewStanding.UNREVIEWED

    def test_should_report_a_verdict_it_cannot_read_as_unreviewed(self, tmp_path):
        record_file_path = _write_record_and_verdict(tmp_path, {'verdict': 'accepted'})
        verdict_file_path = (
            tmp_path / 'train' / 'ore' / 'segmentation' / 'verdict'
            / f'{DOCUMENT_ID_1}.segmentation.verdict.json'
        )
        verdict_file_path.write_text('not json', encoding='utf-8')
        assert read_review_standing(
            record_file_path, HASH_1
        ) == ReviewStanding.UNREVIEWED
