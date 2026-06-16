from __future__ import annotations

from benchmarks.analyze_gold_failures._attribution import (
    _check_model_labels,
    attribute_failures,
)


# Segmentation model token line format (simplified):
# [0] first_word  [1] second_word  [2] lowercase  [3-6] char_prefixes
# [7-32] layout/font features  [33:-1] full block text  [-1] label
def _make_seg_line(block_text: str, label: str) -> list:
    words = block_text.split()
    first = words[0] if words else ''
    second = words[1] if len(words) > 1 else first
    fixed_features = [
        first, second, first.lower(),
        first[:1], first[:2], first[:3], first[:4],
        'BLOCKSTART', 'PAGEIN', 'SAMEFONT', 'SAMEFONTSIZE',
        '1', '0', 'NOCAPS', 'NODIGIT', '0', '0', '1',
        '0', '0', '0', '0', '0', '0', '0',
        'no', '0', '10', '0', '1', '0', '0', '1',
    ]
    return fixed_features + words + [label]


class TestCheckModelLabels:
    def test_sliding_window_finds_single_word_correct(self):
        lines = [_make_seg_line('Introduction', '<header>')]
        matched, predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            'Introduction', lines, frozenset({'<header>'})
        )
        assert matched is True
        assert predicted is None
        assert candidate is None

    def test_sliding_window_finds_multi_line_span(self):
        # Word-level models: each line is one token, gold spans multiple lines
        lines = [
            ['Introduction', '<header>'],
            ['Methods', '<header>'],
        ]
        matched, _predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'IntroductionMethods', lines, frozenset({'<header>'})
        )
        assert matched is True

    def test_block_fallback_finds_multiword_title_in_single_block(self):
        # Multi-word section title lives entirely within one segmentation block.
        # parts[0] = "Potassium" only — sliding window alone cannot match the full title.
        gold = 'Potassium efflux is observed in target cells'
        lines = [_make_seg_line(gold, '<body>')]
        matched, predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            gold, lines, frozenset({'<body>'})
        )
        assert matched is True
        assert predicted is None
        assert candidate is None

    def test_block_fallback_case_insensitive(self):
        # JATS gold "Materials and methods" vs. data "Materials and Methods".
        # _normalize() lowercases both sides, so this should match.
        lines = [_make_seg_line('Materials and Methods', '<annex>')]
        matched, predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'Materials and methods', lines, frozenset({'<body>'})
        )
        assert matched is False
        assert predicted == '<annex>'

    def test_block_fallback_unicode_dash_normalization(self):
        # Gold uses U+2012 FIGURE DASH (‒); data uses ASCII hyphen (-).
        # _normalize() maps both to '-', so they should match.
        lines = [_make_seg_line('PKR-eIF2alpha pathway', '<body>')]
        matched, predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            'PKR‒eIF2alpha pathway', lines, frozenset({'<body>'})
        )
        assert matched is True
        assert predicted is None
        assert candidate is None

    def test_block_fallback_returns_predicted_label_when_wrong_model(self):
        gold = 'Cell death pathway'
        lines = [_make_seg_line(gold, '<header>')]
        matched, predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            gold, lines, frozenset({'<body>'})
        )
        assert matched is False
        assert predicted == '<header>'

    def test_block_fallback_no_false_positive_when_next_char_is_alnum(self):
        # "Supplemental Figures" must NOT match a block starting
        # "Supplemental Figure S1 B" because after the gold prefix
        # "supplementalfigures" the next character is '1' (alnum) — not a word boundary.
        lines = [_make_seg_line('Supplemental Figure S1 B full description', '<body>')]
        matched, _predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'Supplemental Figures', lines, frozenset({'<body>'})
        )
        assert matched is False

    def test_block_fallback_matches_heading_that_prefixes_longer_block(self):
        # "Transgenic mouse assays" should match a block whose text is
        # "Transgenic mouse assays. Sample sizes were selected..." because
        # after the gold prefix the next char is '.' (not alnum) — word boundary.
        block = 'Transgenic mouse assays. Sample sizes were selected empirically'
        lines = [_make_seg_line(block, '<body>')]
        matched, _predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'Transgenic mouse assays', lines, frozenset({'<body>'})
        )
        assert matched is True

    def test_fuzzy_block_match_returns_candidate_text(self):
        # One character difference (µ vs p): exact match fails, fuzzy match finds
        # the block and returns its raw text as candidate_text.
        lines = [_make_seg_line('X-ray tomography (µCT) results', '<body>')]
        matched, _predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            'X-ray tomography (pCT) results', lines, frozenset({'<body>'})
        )
        assert matched is False
        assert candidate is not None
        assert 'µCT' in candidate

    def test_block_fallback_skipped_for_short_gold_values(self):
        # "xy" is the SECOND word of the block, so parts[0]="intro" and the
        # sliding window cannot find it.  Block text "introxymethods" ≠ "xy"
        # (exact match fails) and len("xy") < 8 skips fuzzy match — no result.
        lines = [_make_seg_line('intro xy methods', '<header>')]
        matched, _predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'xy', lines, frozenset({'<header>'})
        )
        assert matched is False

    def test_no_match_returns_false_none(self):
        lines = [_make_seg_line('Introduction', '<header>')]
        matched, _predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            'Discussion', lines, frozenset({'<body>'})
        )
        assert matched is False
        assert candidate is None

    def test_empty_token_lines(self):
        matched, predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            'Introduction', [], frozenset({'<header>'})
        )
        assert matched is False
        assert predicted is None
        assert candidate is None

    def test_empty_gold_value(self):
        lines = [_make_seg_line('Introduction', '<header>')]
        matched, _predicted, candidate, _ctx, _ctx_bl = _check_model_labels(
            '', lines, frozenset({'<header>'})
        )
        assert matched is False
        assert candidate is None

    def test_bio_prefix_stripped_for_relevance_check(self):
        lines = [_make_seg_line('Introduction', 'B-<header>')]
        matched, _predicted, _candidate, _ctx, _ctx_bl = _check_model_labels(
            'Introduction', lines, frozenset({'B-<header>', 'I-<header>'})
        )
        assert matched is True


def _make_data_text(block_text: str, label: str) -> str:
    """Return minimal segmentation .data text for one block."""
    return '\t'.join(_make_seg_line(block_text, label))


class TestAttributeFailures:
    def test_text_found_with_wrong_label_is_attributed(self):
        # Segmentation data has the text labelled <annex> but we expect <body>.
        data_text = _make_data_text('Materials and Methods', '<annex>')
        result = attribute_failures(
            extraction_failed_values=['Materials and Methods'],
            model_chain=['segmentation'],
            relevant_labels={'segmentation': frozenset({'<body>'})},
            model_data={'segmentation': data_text},
        )
        assert 'Materials and Methods' in result
        attr = result['Materials and Methods']
        assert attr.first_failed_model == 'segmentation'
        assert attr.predicted_label == '<annex>'

    def test_text_not_found_carries_note(self):
        # Neither model can find the text → entry present with attribution_note, no model.
        data_seg = _make_data_text('Something unrelated', '<body>')
        data_ft = _make_data_text('Also unrelated', '<section>')
        result = attribute_failures(
            extraction_failed_values=['Supplemental Figures'],
            model_chain=['segmentation', 'fulltext'],
            relevant_labels={
                'segmentation': frozenset({'<body>'}),
                'fulltext': frozenset({'<section>'}),
            },
            model_data={'segmentation': data_seg, 'fulltext': data_ft},
        )
        assert 'Supplemental Figures' in result
        attr = result['Supplemental Figures']
        assert attr.first_failed_model is None
        assert attr.attribution_note == 'Text not found in any model data'

    def test_all_models_correct_carries_note(self):
        # Both models label the text correctly — entry present with attribution_note.
        data_seg = _make_data_text('Disruption of the pathway', '<body>')
        data_ft = _make_data_text('Disruption of the pathway', '<section>')
        result = attribute_failures(
            extraction_failed_values=['Disruption of the pathway'],
            model_chain=['segmentation', 'fulltext'],
            relevant_labels={
                'segmentation': frozenset({'<body>'}),
                'fulltext': frozenset({'<section>'}),
            },
            model_data={'segmentation': data_seg, 'fulltext': data_ft},
        )
        assert 'Disruption of the pathway' in result
        attr = result['Disruption of the pathway']
        assert attr.first_failed_model is None
        assert attr.attribution_note == 'All models correctly classify this text'

    def test_missing_data_carries_not_found_note(self):
        # If model data is absent we cannot blame that model; note is added.
        result = attribute_failures(
            extraction_failed_values=['Introduction'],
            model_chain=['segmentation'],
            relevant_labels={'segmentation': frozenset({'<body>'})},
            model_data={},  # no data available
        )
        assert 'Introduction' in result
        attr = result['Introduction']
        assert attr.first_failed_model is None
        assert attr.attribution_note == 'Text not found in any model data'

    def test_fuzzy_match_sets_candidate_text(self):
        # Near-miss block (µCT vs pCT) returns candidate_text.
        data_text = _make_data_text('X-ray tomography (µCT) results', '<body>')
        result = attribute_failures(
            extraction_failed_values=['X-ray tomography (pCT) results'],
            model_chain=['segmentation'],
            relevant_labels={'segmentation': frozenset({'<body>'})},
            model_data={'segmentation': data_text},
        )
        assert 'X-ray tomography (pCT) results' in result
        attr = result['X-ray tomography (pCT) results']
        assert attr.candidate_text is not None
        assert 'µCT' in attr.candidate_text

    def test_block_mismatch_populates_context_window(self):
        # Context window is captured for block-level mismatches.
        # Multi-word gold so the sliding window (parts[0] only) cannot match it,
        # forcing the block prefix strategy which sets context_is_block_level=True.
        data_text = '\n'.join([
            '\t'.join(_make_seg_line('Previous section text', '<body>')),
            '\t'.join(_make_seg_line('Materials and Methods', '<annex>')),
            '\t'.join(_make_seg_line('Following section text', '<body>')),
        ])
        result = attribute_failures(
            extraction_failed_values=['Materials and Methods'],
            model_chain=['segmentation'],
            relevant_labels={'segmentation': frozenset({'<body>'})},
            model_data={'segmentation': data_text},
        )
        attr = result['Materials and Methods']
        assert attr.context_window is not None
        assert attr.context_is_block_level is True
        assert any(is_match for _text, _label, is_match in attr.context_window)
