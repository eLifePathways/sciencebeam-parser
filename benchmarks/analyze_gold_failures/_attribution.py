from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Tuple

from ._types import PipelineAttribution

LOGGER = logging.getLogger(__name__)

PARSER_DEFAULT_URL = 'http://localhost:8080'

# Map Unicode dashes to ASCII hyphen and remove zero-width characters.
# Gold JATS often uses typographic dashes (e.g. U+2012 FIGURE DASH in "PKR‒eIF2α")
# while the model's feature data uses plain ASCII hyphens from PDF extraction.
_NORMALIZE_MAP = str.maketrans({
    0x2010: '-',   # HYPHEN
    0x2011: '-',   # NON-BREAKING HYPHEN
    0x2012: '-',   # FIGURE DASH
    0x2013: '-',   # EN DASH
    0x2014: '-',   # EM DASH
    0x00A0: ' ',   # NON-BREAKING SPACE → regular space (removed by \s+ below)
    0x200B: '',    # ZERO WIDTH SPACE
})


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_NORMALIZE_MAP)
    return re.sub(r'\s+', '', text)


def _strip_bio(label: str) -> str:
    return re.sub(r'^[BI]-', '', label)


def _parse_data_lines(data_text: str) -> List[List[str]]:
    """Parse model data text into token lines; each line is [token, feat..., label]."""
    result = []
    for line in data_text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            result.append(parts)
    return result


_CONTEXT_RADIUS_WORD = 5   # tokens either side of matched span
_CONTEXT_RADIUS_BLOCK = 3  # blocks either side of matched block

# Type alias for a context-window entry: (display_text, label, is_matched_span)
_ContextLine = Tuple[str, str, bool]


def _check_model_labels(  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements  # noqa: E501
    gold_value: str,
    token_lines: List[List[str]],
    relevant_labels: AbstractSet[str],
) -> Tuple[bool, Optional[str], Optional[str], Optional[List[_ContextLine]], bool]:
    """Return (matched, predicted_label, candidate_text, context_window, is_block_level).

    matched: True if a token span matching the gold value has a relevant label.
    predicted_label: label on the matching span when matched is False and the span
                     was found (so the caller can report what was predicted).
    candidate_text: raw block text from a near-miss fuzzy match when the normalized
                    forms differ by only a few characters (e.g. encoding variants
                    such as µCT vs pCT).  None when there is no close candidate.
    context_window: surrounding lines from model data at the point of mislabelling.
                    None when matched is True or no span was found.
    is_block_level: True when context_window entries come from the block-level
                    (segmentation) strategy rather than the token-level sliding window.
    """
    norm_gold = _normalize(gold_value)
    if not norm_gold or not token_lines:
        return False, None, None, None, False

    relevant_base = {_strip_bio(lbl) for lbl in relevant_labels}

    # --- Sliding window: word-level models (one token per line) ---
    for start in range(len(token_lines)):
        accumulated = ''
        for end in range(start, min(start + 200, len(token_lines))):
            accumulated += _normalize(token_lines[end][0])
            if accumulated == norm_gold:
                span_labels = {_strip_bio(token_lines[i][-1]) for i in range(start, end + 1)}
                if span_labels & relevant_base:
                    return True, None, None, None, False
                predicted = ', '.join(sorted(span_labels))
                raw_span = ' '.join(token_lines[i][0] for i in range(start, end + 1))
                ctx_s = max(0, start - _CONTEXT_RADIUS_WORD)
                ctx_e = min(len(token_lines) - 1, end + _CONTEXT_RADIUS_WORD)
                context: List[_ContextLine] = [
                    (token_lines[i][0], token_lines[i][-1], start <= i <= end)
                    for i in range(ctx_s, ctx_e + 1)
                ]
                return False, predicted, raw_span, context, False
            if len(accumulated) > len(norm_gold):
                break

    # Pre-index block-level lines so we can retrieve neighbours by position.
    block_lines: List[Tuple[int, List[str]]] = [
        (orig_i, lp) for orig_i, lp in enumerate(token_lines) if len(lp) > 34
    ]

    # --- Prefix-with-boundary block match: block-level models (segmentation) ---
    # Segmentation model lines have 33 fixed feature columns; the full block text
    # occupies columns [33:-1] before the label.  We accept the gold as a match
    # when the normalized block text STARTS WITH the normalized gold AND the
    # immediately following character (if any) is not alphanumeric — i.e. the gold
    # text ends at a word boundary such as a period, comma, or end-of-block.
    #
    # This handles two cases:
    #   • Exact: block = "Materials and Methods"  → no suffix → valid
    #   • Heading prefix: block = "Transgenic mouse assays. Sample..."  → suffix '.' → valid
    # And rejects:
    #   • False prefix: "Supplemental Figure S1 B" normalized starts with
    #     "supplementalfigures" but the next char is '1' (alnum) → invalid
    for bi, (_, line_parts) in enumerate(block_lines):
        label = _strip_bio(line_parts[-1])
        block_text = ''.join(_normalize(p) for p in line_parts[33:-1])
        if block_text.startswith(norm_gold):
            suffix_char = block_text[len(norm_gold):len(norm_gold) + 1]
            if not suffix_char or not suffix_char.isalnum():
                if label in relevant_base:
                    return True, None, None, None, True
                # Reconstruct raw words that form the matched prefix
                acc = ''
                n_prefix = 0
                for p in line_parts[33:-1]:
                    acc += _normalize(p)
                    n_prefix += 1
                    if len(acc) >= len(norm_gold):
                        break
                raw_prefix = ' '.join(line_parts[33:33 + n_prefix])
                ctx_s = max(0, bi - _CONTEXT_RADIUS_BLOCK)
                ctx_e = min(len(block_lines) - 1, bi + _CONTEXT_RADIUS_BLOCK)
                block_context: List[_ContextLine] = [
                    (' '.join(block_lines[j][1][33:-1]), block_lines[j][1][-1], j == bi)
                    for j in range(ctx_s, ctx_e + 1)
                ]
                return False, label, raw_prefix, block_context, True

    # --- Fuzzy block match: near-miss blocks for diagnostic display ---
    # Catches single-character encoding differences between gold and data
    # (e.g. µCT vs pCT).  Returns the raw block text so the report can show
    # what was actually found in the model data alongside the gold value.
    if len(norm_gold) >= 8:
        max_mismatches = max(1, len(norm_gold) // 30)
        best_bi: Optional[int] = None
        best_candidate: Optional[Tuple[str, str]] = None  # (label, raw_text)
        best_mismatches = max_mismatches + 1
        for bi, (_, line_parts) in enumerate(block_lines):
            label = _strip_bio(line_parts[-1])
            block_text = ''.join(_normalize(p) for p in line_parts[33:-1])
            if len(block_text) != len(norm_gold):
                continue
            mismatches = sum(a != b for a, b in zip(norm_gold, block_text))
            if 0 < mismatches <= max_mismatches and mismatches < best_mismatches:
                best_mismatches = mismatches
                raw_text = ' '.join(line_parts[33:-1])
                best_candidate = (label, raw_text)
                best_bi = bi
        if best_candidate is not None and best_bi is not None:
            cand_label, cand_raw = best_candidate
            ctx_s = max(0, best_bi - _CONTEXT_RADIUS_BLOCK)
            ctx_e = min(len(block_lines) - 1, best_bi + _CONTEXT_RADIUS_BLOCK)
            fuzzy_context: List[_ContextLine] = [
                (' '.join(block_lines[j][1][33:-1]), block_lines[j][1][-1], j == best_bi)
                for j in range(ctx_s, ctx_e + 1)
            ]
            return False, cand_label, cand_raw, fuzzy_context, True

    return False, None, None, None, False


def attribute_failures(  # pylint: disable=too-many-locals
    extraction_failed_values: List[str],
    model_chain: List[str],
    relevant_labels: Dict[str, frozenset],
    model_data: Dict[str, str],
) -> Dict[str, PipelineAttribution]:
    """Attribute EXTRACTION_FAILED values to the first model layer that lost them.

    A model is blamed only when the text is *found* in its data with a wrong label,
    or when a near-miss fuzzy block is found.  All values are included in the result;
    those with no definite attribution carry an attribution_note explaining why.
    """
    attributions: Dict[str, PipelineAttribution] = {}

    for value in extraction_failed_values:
        correct_models: List[str] = []
        failed_models: List[str] = []
        first_failed: Optional[str] = None
        first_failed_predicted: Optional[str] = None
        first_failed_expected: Optional[str] = None
        first_failed_candidate: Optional[str] = None
        first_failed_context: Optional[List] = None
        first_failed_context_block: bool = False

        for model_name in model_chain:
            data_text = model_data.get(model_name, '')
            if not data_text:
                failed_models.append(model_name)
                continue  # data unavailable, cannot determine pass/fail

            token_lines = _parse_data_lines(data_text)
            model_relevant = relevant_labels.get(model_name, frozenset())
            is_correct, predicted_label, candidate_text, ctx_window, ctx_block = (
                _check_model_labels(value, token_lines, model_relevant)
            )

            if is_correct:
                correct_models.append(model_name)
            elif predicted_label is not None or candidate_text is not None:
                # Text was found but with wrong label (or a near-miss was detected).
                # This model is the first definite failure point.
                failed_models.append(model_name)
                first_failed = model_name
                first_failed_predicted = predicted_label
                first_failed_candidate = candidate_text
                first_failed_context = ctx_window
                first_failed_context_block = ctx_block
                expected_bases = {_strip_bio(lbl) for lbl in model_relevant}
                first_failed_expected = (
                    ', '.join(sorted(expected_bases)) if expected_bases else None
                )
                break
            else:
                # Text not found at all in this model's data — can't blame it.
                failed_models.append(model_name)

        if first_failed is not None:
            attributions[value] = PipelineAttribution(
                correct_models=correct_models,
                failed_models=failed_models,
                recommended_action=f'Add training examples to `{first_failed}` model',
                first_failed_model=first_failed,
                predicted_label=first_failed_predicted,
                expected_label=first_failed_expected,
                candidate_text=first_failed_candidate,
                context_window=first_failed_context,
                context_is_block_level=first_failed_context_block,
            )
        else:
            # No definite attribution: include with an explanatory note.
            if correct_models:
                note = 'All models correctly classify this text'
            else:
                note = 'Text not found in any model data'
            attributions[value] = PipelineAttribution(
                correct_models=correct_models,
                failed_models=failed_models,
                recommended_action='',
                attribution_note=note,
            )

    return attributions


def fetch_model_data(
    pdf_path: Path,
    model_chain: List[str],
    parser_url: str,
    out_dir: Path,
) -> Dict[str, str]:
    """Fetch and cache ScienceBeam Parser token-level data for each model in the chain."""
    # pylint: disable-next=import-outside-toplevel
    from benchmarks.analyze_field_regressions._fetch import (
        _fetch_parser_model_data,
    )

    model_data: Dict[str, str] = {}

    for model_name in model_chain:
        data_path = out_dir / 'sciencebeam-parser' / f'{model_name}.data'
        feature_names_path = out_dir / 'sciencebeam-parser' / f'{model_name}.feature_names.json'

        if not data_path.exists():
            if not pdf_path.exists():
                LOGGER.warning('PDF not found for attribution: %s', pdf_path)
                continue
            try:
                LOGGER.info('Fetching parser model data for %s / %s', pdf_path.stem, model_name)
                _fetch_parser_model_data(
                    pdf_path, model_name, parser_url, data_path, feature_names_path
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning('Failed to fetch model data for %s: %s', model_name, exc)
                continue

        if data_path.exists():
            try:
                model_data[model_name] = data_path.read_text(encoding='utf-8', errors='replace')
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning('Failed to read model data for %s: %s', model_name, exc)

    return model_data
