from __future__ import annotations

from benchmarks.analyze_gold_failures._report import (
    _classify_near_miss,
    _format_sim,
    render_report,
)
from benchmarks.analyze_gold_failures._types import (
    DocumentSummary,
    FailureMode,
    FailureModeAggregate,
    FieldFailureSummary,
    GoldValueResult,
    PipelineAttribution,
)


def _make_summary(mode_counts: dict, is_online: bool = False) -> FieldFailureSummary:
    total = sum(mode_counts.values())
    mode_aggregates = []
    for mode in FailureMode:
        count = mode_counts.get(mode, 0)
        examples = [
            ('biorxiv', 'doc1', GoldValueResult(
                value=f'value-{i}', mode=mode, in_raw=True, in_sb_field=True
            ))
            for i in range(count)
        ]
        mode_aggregates.append(FailureModeAggregate(
            mode=mode, total_values=count, docs_affected=1 if count else 0, examples=examples,
        ))
    return FieldFailureSummary(
        field='body_section_titles',
        run_sb='benchmarks/runs/train',
        total_docs=1,
        total_gold=total,
        mode_aggregates=mode_aggregates,
        attribution_aggregates=[],
        recommended_action='Test recommendation.',
        is_online=is_online,
    )


def _make_doc(results: list) -> DocumentSummary:
    return DocumentSummary(
        corpus='biorxiv', record_id='doc1', score_sb=0.75, results=results,
    )


class TestRenderReport:
    def test_report_contains_field_name(self):
        summary = _make_summary({FailureMode.CORRECT: 5})
        doc = _make_doc([
            GoldValueResult(value='v', mode=FailureMode.CORRECT, in_raw=True, in_sb_field=True)
        ] * 5)
        report = render_report(summary, [doc])
        assert 'body_section_titles' in report

    def test_report_contains_summary_table(self):
        summary = _make_summary({
            FailureMode.NOT_IN_RAW_TEXT: 2,
            FailureMode.EXTRACTION_FAILED: 3,
            FailureMode.CORRECT: 5,
        })
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert 'Not found in raw text' in report
        assert 'Extraction failed' in report
        assert 'Correct' in report

    def test_report_shows_not_in_raw_section_when_present(self):
        summary = _make_summary({FailureMode.NOT_IN_RAW_TEXT: 1, FailureMode.CORRECT: 4})
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert '## Not found in raw text' in report

    def test_report_omits_not_in_raw_section_when_zero(self):
        summary = _make_summary({FailureMode.CORRECT: 5})
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert '## Not found in raw text' not in report

    def test_report_shows_offline_attribution_prompt(self):
        summary = _make_summary({FailureMode.EXTRACTION_FAILED: 3}, is_online=False)
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert '--parser-url' in report

    def test_report_shows_attribution_summary_when_online(self):
        summary = _make_summary({FailureMode.EXTRACTION_FAILED: 1}, is_online=True)
        result = GoldValueResult(
            value='value-0', mode=FailureMode.EXTRACTION_FAILED, in_raw=True, in_sb_field=False
        )
        summary.mode_aggregates[int(FailureMode.EXTRACTION_FAILED)].examples = [
            ('biorxiv', 'doc1', result)
        ]
        doc = _make_doc([result])
        doc.attributions['value-0'] = PipelineAttribution(
            correct_models=[],
            failed_models=['fulltext'],
            recommended_action='',
            first_failed_model='fulltext',
            predicted_label='<paragraph>',
            expected_label='<section>',
        )
        report = render_report(summary, [doc])
        assert 'Attribution summary' in report
        assert 'fulltext' in report

    def test_report_contains_presence_footnote(self):
        summary = _make_summary({FailureMode.CORRECT: 1})
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert 'Presence checks use whitespace-normalised substring matching' in report

    def test_report_contains_doc_table(self):
        summary = _make_summary({FailureMode.CORRECT: 1})
        doc = _make_doc([
            GoldValueResult(value='v', mode=FailureMode.CORRECT, in_raw=True, in_sb_field=True)
        ])
        report = render_report(summary, [doc])
        assert 'doc1' in report
        assert 'biorxiv' in report

    def test_sim_column_present_in_online_extraction_failed_table(self):
        summary = _make_summary({FailureMode.EXTRACTION_FAILED: 1}, is_online=True)
        doc = _make_doc([])
        report = render_report(summary, [doc])
        assert 'Sim' in report

    def test_partial_wrong_section_shows_similarity(self):
        summary = _make_summary({FailureMode.PARTIAL_WRONG: 1})
        result = GoldValueResult(
            value='Results', mode=FailureMode.PARTIAL_WRONG,
            in_raw=True, in_sb_field=True,
            best_sb_match='Results and Discussion', best_sb_similarity=0.54,
        )
        summary.mode_aggregates[int(FailureMode.PARTIAL_WRONG)].examples = [
            ('biorxiv', 'doc1', result)
        ]
        doc = _make_doc([result])
        report = render_report(summary, [doc])
        assert 'Partial/wrong match' in report
        assert '0.54' in report


class TestClassifyNearMiss:
    def test_case_only(self):
        assert _classify_near_miss('Results', 'results') == 'Case only'

    def test_dash_variant(self):
        assert _classify_near_miss('PKR‒eIF2', 'PKR-eIF2') == 'Dash/hyphen variant'

    def test_case_and_dash(self):
        assert _classify_near_miss('PKR‒EIF2', 'PKR-eif2') == 'Case + dash variant'

    def test_trailing_punctuation(self):
        assert _classify_near_miss('Methods.', 'Methods') == 'Trailing punctuation'

    def test_single_char_encoding(self):
        assert _classify_near_miss('pCT scan', 'µCT scan') == 'Single-char (encoding)'

    def test_other(self):
        assert _classify_near_miss('Introduction', 'Results') == 'Other'


class TestFormatSim:
    def test_none_candidate_returns_dash(self):
        assert _format_sim('anything', None) == '—'

    def test_identical_strings_return_one(self):
        assert _format_sim('hello world', 'hello world') == '1.00'

    def test_single_char_diff_returns_high_ratio(self):
        result = _format_sim('X-ray (pCT) scan', 'X-ray (µCT) scan')
        ratio = float(result)
        assert 0.9 < ratio < 1.0
