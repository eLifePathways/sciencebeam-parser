from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.coverage import CoverageResult, check_coverage
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames


def _make_annotated(*field_names: str) -> JatsAnnotatedLayoutDocument:
    tokens = [LayoutToken(text=f'token_{i}') for i in range(len(field_names))]
    line = LayoutLine(tokens=tokens)
    doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
    annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
    for token, field in zip(tokens, field_names):
        if field:
            annotated.set_token_label(token, field)
    return annotated


class TestCoverageResult:
    def test_passing_when_no_requirements(self):
        result = CoverageResult()
        assert result.is_passing

    def test_failing_when_required_field_missing(self):
        result = CoverageResult(required_fields_missing={JatsFieldNames.TITLE})
        assert not result.is_passing

    def test_failing_when_matching_field_not_aligned(self):
        result = CoverageResult(
            required_matching_fields_missing={JatsFieldNames.ABSTRACT}
        )
        assert not result.is_passing

    def test_str_ok(self):
        assert str(CoverageResult()) == 'OK'

    def test_str_shows_missing_field(self):
        result = CoverageResult(required_fields_missing={'title'})
        assert 'title' in str(result)


class TestCheckCoverage:
    def test_required_field_present_and_aligned_passes(self):
        annotated = _make_annotated(JatsFieldNames.TITLE)
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={JatsFieldNames.TITLE: True},
            required_fields=[JatsFieldNames.TITLE],
            require_matching_fields=[],
        )
        assert result.is_passing
        assert JatsFieldNames.TITLE in result.required_fields_present

    def test_required_field_absent_from_jats_fails(self):
        annotated = _make_annotated()
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={},
            required_fields=[JatsFieldNames.TITLE],
            require_matching_fields=[],
        )
        assert not result.is_passing
        assert JatsFieldNames.TITLE in result.required_fields_missing

    def test_required_field_present_but_not_aligned_fails(self):
        annotated = _make_annotated()  # no labels assigned
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={JatsFieldNames.TITLE: True},
            required_fields=[JatsFieldNames.TITLE],
            require_matching_fields=[],
        )
        assert not result.is_passing
        assert JatsFieldNames.TITLE in result.required_fields_missing

    def test_require_matching_field_absent_in_jats_passes(self):
        annotated = _make_annotated()
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={},  # field not present in JATS
            required_fields=[],
            require_matching_fields=[JatsFieldNames.ABSTRACT],
        )
        assert result.is_passing

    def test_require_matching_field_present_but_not_aligned_fails(self):
        annotated = _make_annotated()  # no labels
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={JatsFieldNames.ABSTRACT: True},
            required_fields=[],
            require_matching_fields=[JatsFieldNames.ABSTRACT],
        )
        assert not result.is_passing
        assert JatsFieldNames.ABSTRACT in result.required_matching_fields_missing

    def test_require_matching_field_present_and_aligned_passes(self):
        annotated = _make_annotated(JatsFieldNames.ABSTRACT)
        result = check_coverage(
            annotated=annotated,
            field_values_by_field={JatsFieldNames.ABSTRACT: True},
            required_fields=[],
            require_matching_fields=[JatsFieldNames.ABSTRACT],
        )
        assert result.is_passing
        assert JatsFieldNames.ABSTRACT in result.required_matching_fields_matched
