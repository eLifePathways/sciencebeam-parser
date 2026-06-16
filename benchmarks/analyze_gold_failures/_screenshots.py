from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from xml.etree import ElementTree

# Register TEI as the default namespace so tostring() emits clean element names
# (e.g. <head> instead of <ns0:head>).
_TEI_NS = 'http://www.tei-c.org/ns/1.0'
ElementTree.register_namespace('', _TEI_NS)

_RENDER_DPI = 150
_PADDING_PT = 30.0
_HIGHLIGHT_COLOR = (255, 220, 0, 70)
_HIGHLIGHT_OUTLINE = (255, 140, 0, 200)
_SCREENSHOTS_SUBDIR = 'screenshots'
# Minimum similarity for a TEI element to be accepted as a match.
# score = len(gold) / len(element_text).  Elements below this threshold
# (e.g. large paragraphs that merely mention the gold word) are rejected,
# which avoids showing a misleading screenshot when no heading element exists.
_MIN_TEI_SIMILARITY = 0.3
# Context-aware matching: how much sibling text matching contributes to the score.
_CTX_WEIGHT = 0.25
# Only use the first N chars of a neighbour text as a prefix when checking siblings.
_CTX_PREFIX_LEN = 30
# Ignore context neighbour texts shorter than this (e.g. single word-level tokens).
_CTX_MIN_NEIGHBOR_LEN = 8

# Map GROBID/fulltext model predicted labels to the required TEI ancestor tag.
# When predicted_label is provided, only elements inside that TEI ancestor are
# considered — this prevents e.g. a section-title search from matching an abstract
# <hi> instead of the body <figure> where the fulltext model actually lost the label.
_GROBID_LABEL_TO_TEI_TAG: Dict[str, str] = {
    '<figure>': 'figure',
    '<table>': 'figure',
    '<paragraph>': 'p',
    '<section>': 'div',
}

# Mirrors _attribution._ContextLine: (display_text, label, is_matched_span)
_ContextLine = Tuple[str, str, bool]
# Item type for render_context_screenshots / collect_tei_snippets.
# 5-tuple: (corpus, record_id, gold_value, context_window, predicted_label)
# predicted_label (e.g. '<figure>') filters candidate TEI elements to the right ancestor.
_ScreenshotItem = Tuple[str, str, str, Optional[List[_ContextLine]], Optional[str]]


class TeiCoords(NamedTuple):
    page: int      # 1-based page number
    x: float       # PDF points from left edge
    y: float       # PDF points from top edge
    w: float       # width in PDF points
    h: float       # height in PDF points


def _value_slug(gold_value: str) -> str:
    h = hashlib.md5(gold_value.encode('utf-8')).hexdigest()[:8]
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', gold_value).strip('_')[:40]
    return f'{safe}_{h}'


def _norm_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', '', text.lower())


def _parse_tei_coords(coords_str: str) -> Optional[TeiCoords]:
    try:
        parts = [float(p) for p in coords_str.split(',')]
        if len(parts) >= 5:
            return TeiCoords(int(parts[0]), parts[1], parts[2], parts[3], parts[4])
    except (ValueError, IndexError):
        pass
    return None


def _iter_element_text(el: ElementTree.Element) -> str:
    return ''.join(el.itertext())


def _text_similarity(a: str, b: str) -> float:
    """Simple overlap ratio: shared characters / max length."""
    if not a or not b:
        return 0.0
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    # Score = proportion of the gold value covered by the element text.
    # An element whose normalised text equals the gold value scores 1.0;
    # an element containing it as a tiny substring scores close to 0.
    if shorter in longer:
        return len(shorter) / max(len(longer), 1)
    return 0.0


def _extract_context_neighbors(
    context_window: Optional[List[_ContextLine]],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (norm_pre, norm_post): normalised texts of the blocks immediately
    before and after the matched span in the context window.

    Returns (None, None) when context is unavailable or neighbour texts are too
    short to be useful for sibling matching (e.g. individual word-level tokens).
    """
    if not context_window:
        return None, None
    first_match = next((i for i, (_, _, m) in enumerate(context_window) if m), None)
    last_match = next(
        (i for i in range(len(context_window) - 1, -1, -1) if context_window[i][2]),
        None,
    )
    if first_match is None or last_match is None:
        return None, None

    def _usable(text: str) -> Optional[str]:
        n = _norm_text(text)
        return n if len(n) >= _CTX_MIN_NEIGHBOR_LEN else None

    norm_pre = _usable(context_window[first_match - 1][0]) if first_match > 0 else None
    norm_post = (
        _usable(context_window[last_match + 1][0])
        if last_match < len(context_window) - 1 else None
    )
    return norm_pre, norm_post


def _sibling_context_score(
    el: ElementTree.Element,
    parent_map: Dict[ElementTree.Element, ElementTree.Element],
    norm_pre: Optional[str],
    norm_post: Optional[str],
) -> float:
    """Score [0, 1] based on how well the element's TEI siblings match the context.

    Checks whether a prefix of norm_pre appears in the preceding sibling's text,
    and whether a prefix of norm_post appears in the following sibling's text.
    Returns 0.5 per direction that matches, summed and averaged.
    This helps distinguish a section heading from a bibliography journal title
    that happens to share the same text.
    """
    parent = parent_map.get(el)
    if parent is None:
        return 0.0
    siblings = list(parent)
    try:
        idx = siblings.index(el)
    except ValueError:
        return 0.0

    score = 0.0
    n = 0
    if norm_pre:
        n += 1
        prefix = norm_pre[:_CTX_PREFIX_LEN]
        if prefix and idx > 0:
            prev_norm = _norm_text(_iter_element_text(siblings[idx - 1]))
            if prefix in prev_norm:
                score += 1.0
    if norm_post:
        n += 1
        prefix = norm_post[:_CTX_PREFIX_LEN]
        if prefix and idx < len(siblings) - 1:
            next_norm = _norm_text(_iter_element_text(siblings[idx + 1]))
            if prefix in next_norm:
                score += 1.0
    return score / n if n > 0 else 0.0


def _nearest_coords_ancestor(
    el: ElementTree.Element,
    parent_map: Dict[ElementTree.Element, ElementTree.Element],
) -> Optional[ElementTree.Element]:
    """Return el if it has a coords attribute, else walk up to the nearest ancestor that does."""
    current: Optional[ElementTree.Element] = el
    while current is not None:
        if current.get('coords'):
            return current
        current = parent_map.get(current)
    return None


def _ancestor_tag_from_label(predicted_label: Optional[str]) -> Optional[str]:
    """Map a GROBID model label to the TEI ancestor tag it implies, or None."""
    if not predicted_label:
        return None
    return _GROBID_LABEL_TO_TEI_TAG.get(predicted_label)


def _has_ancestor_of_tag(
    el: ElementTree.Element,
    tag: str,
    parent_map: Dict[ElementTree.Element, ElementTree.Element],
) -> bool:
    """Return True if any ancestor of el has local-name == tag (TEI namespace or bare)."""
    tei_tag = f'{{{_TEI_NS}}}{tag}'
    current = parent_map.get(el)
    while current is not None:
        if current.tag in (tei_tag, tag):
            return True
        current = parent_map.get(current)
    return False


def _element_xml_path(
    el: ElementTree.Element,
    parent_map: Dict[ElementTree.Element, ElementTree.Element],
) -> str:
    """Build a slash-separated path of local tag names from root to el."""
    parts: List[str] = []
    current: Optional[ElementTree.Element] = el
    while current is not None:
        tag = current.tag
        if tag.startswith(f'{{{_TEI_NS}}}'):
            tag = tag[len(f'{{{_TEI_NS}}}'):]
        parts.append(tag)
        current = parent_map.get(current)
    return '/' + '/'.join(reversed(parts))


def _estimate_heading_height(
    coords_el: ElementTree.Element,
    coords: TeiCoords,
) -> Optional[float]:
    """Estimate heading height from median child-element height on the same page.

    Used to narrow the bounding box when text_el is an inline element (e.g. <hi>)
    whose nearest coords ancestor (coords_el) spans a whole paragraph.
    """
    child_heights = []
    for child in coords_el:
        child_coords = _parse_tei_coords(child.get('coords', ''))
        if child_coords and child_coords.page == coords.page and child_coords.h > 0:
            child_heights.append(child_coords.h)
    if not child_heights:
        return None
    sorted_h = sorted(child_heights)
    median_h = sorted_h[len(sorted_h) // 2]
    # A heading typically spans 2–3 lines at the reference line height.
    return median_h * 3.0


def _find_best_tei_element(  # pylint: disable=too-many-locals
    root: ElementTree.Element,
    norm_gold: str,
    norm_pre: Optional[str] = None,
    norm_post: Optional[str] = None,
    ancestor_tag: Optional[str] = None,
) -> Tuple[Optional[ElementTree.Element], Optional[ElementTree.Element]]:
    """Return (text_el, coords_el) for the best matching TEI element.

    text_el:   the element whose own text most closely matches the gold value
               (may be an inline element like <hi> that lacks coords).
    coords_el: text_el itself if it has coords, else the nearest ancestor that does.
               Used for screenshot bounding-box coordinates.

    Only elements with a gold-text similarity >= _MIN_TEI_SIMILARITY are
    considered, filtering out large blocks that merely mention the gold word.

    When norm_pre / norm_post are given (normalised texts of the preceding /
    following context blocks from model data), the score is boosted for elements
    whose coords ancestor's siblings match those texts.  This disambiguates e.g.
    a section heading from a bibliography journal title with the same short text.

    When ancestor_tag is given (e.g. 'figure'), only elements inside a TEI element
    with that local tag name are considered.  This prevents an abstract <hi> from
    beating the correct body <figure> element when the model predicted <figure>.
    """
    # Parent map is always needed: for ancestor-coords lookup and context scoring.
    parent_map: Dict[ElementTree.Element, ElementTree.Element] = {
        c: p for p in root.iter() for c in p
    }
    use_context = bool(norm_pre or norm_post)

    best_text_el: Optional[ElementTree.Element] = None
    best_coords_el: Optional[ElementTree.Element] = None
    best_score: float = _MIN_TEI_SIMILARITY - 0.001

    for el in root.iter():
        norm_el = _norm_text(_iter_element_text(el))
        if norm_gold not in norm_el:
            continue
        gold_score = _text_similarity(norm_gold, norm_el)
        if gold_score < _MIN_TEI_SIMILARITY:
            continue
        if ancestor_tag and not _has_ancestor_of_tag(el, ancestor_tag, parent_map):
            continue
        coords_el = _nearest_coords_ancestor(el, parent_map)
        if coords_el is None:
            continue  # no coords anywhere in ancestor chain
        ctx_score = (
            _sibling_context_score(coords_el, parent_map, norm_pre, norm_post)
            if use_context else 0.0
        )
        score = gold_score + _CTX_WEIGHT * ctx_score
        if score > best_score:
            best_text_el = el
            best_coords_el = coords_el
            best_score = score

    return best_text_el, best_coords_el


def _parse_context(
    context_window: Optional[List[_ContextLine]],
) -> Tuple[Optional[str], Optional[str]]:
    """Extract normalised context neighbour texts from a context window."""
    return _extract_context_neighbors(context_window)


def find_coords_for_value(
    tei_path: Path,
    gold_value: str,
    context_window: Optional[List[_ContextLine]] = None,
    predicted_label: Optional[str] = None,
) -> Optional[TeiCoords]:
    """Search TEI XML for the element best matching gold_value.

    Prefers elements whose full text is close to the gold value (heading blocks)
    over large containers (paragraphs) or tiny sub-word elements.
    Inline elements without coords (e.g. <hi rend="bold">) are matched by their
    text and highlighted via their nearest ancestor element that has coords.
    When context_window is given, sibling matching further disambiguates candidates.
    When predicted_label is given (e.g. '<figure>'), only elements inside the
    corresponding TEI ancestor are considered.
    If the matched element is inline (text_el != coords_el), the bounding-box height
    is narrowed using sibling element heights as a line-height estimate.
    """
    try:
        tree = ElementTree.parse(tei_path)
        root = tree.getroot()
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    norm_gold = _norm_text(gold_value)
    if not norm_gold:
        return None

    norm_pre, norm_post = _parse_context(context_window)
    ancestor_tag = _ancestor_tag_from_label(predicted_label)
    text_el, coords_el = _find_best_tei_element(root, norm_gold, norm_pre, norm_post, ancestor_tag)
    if coords_el is None:
        return None
    coords = _parse_tei_coords(coords_el.get('coords', ''))
    if coords is None:
        return None
    # Narrow bounding box when text_el is an inline element (no own coords).
    # The coords_el may span a whole paragraph; use child heights as proxy for
    # line height to crop to just the heading area.
    if text_el is not None and text_el is not coords_el:
        est_h = _estimate_heading_height(coords_el, coords)
        if est_h is not None and est_h < coords.h:
            coords = TeiCoords(coords.page, coords.x, coords.y, coords.w, est_h)
    return coords


def find_tei_snippet_for_value(
    tei_path: Path,
    gold_value: str,
    context_window: Optional[List[_ContextLine]] = None,
    predicted_label: Optional[str] = None,
) -> Optional[str]:
    """Return XML source of the TEI element best matching gold_value, or None.

    Prepends an XML comment with the full element path (e.g.
    <!-- /TEI/text/body/div/figure/head/hi -->) so the caller can see where in
    the document the match was found without reading the whole TEI file.
    """
    try:
        tree = ElementTree.parse(tei_path)
        root = tree.getroot()
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    norm_gold = _norm_text(gold_value)
    if not norm_gold:
        return None

    norm_pre, norm_post = _parse_context(context_window)
    ancestor_tag = _ancestor_tag_from_label(predicted_label)
    text_el, _ = _find_best_tei_element(root, norm_gold, norm_pre, norm_post, ancestor_tag)
    if text_el is None:
        return None
    parent_map: Dict[ElementTree.Element, ElementTree.Element] = {
        c: p for p in root.iter() for c in p
    }
    path = _element_xml_path(text_el, parent_map)
    xml_str = ElementTree.tostring(text_el, encoding='unicode')
    return f'<!-- {path} -->\n{xml_str}'


def render_region_screenshot(  # pylint: disable=too-many-locals
    pdf_path: Path,
    coords: TeiCoords,
    dpi: int = _RENDER_DPI,
    padding_pt: float = _PADDING_PT,
) -> Optional[bytes]:
    """Render a highlighted region from a PDF page. Returns PNG bytes or None."""
    try:
        # pylint: disable=import-outside-toplevel
        from pdf2image import convert_from_path  # type: ignore[import-untyped]
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    try:
        images = convert_from_path(
            str(pdf_path), dpi=dpi,
            first_page=coords.page, last_page=coords.page,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    if not images:
        return None

    img: Image.Image = images[0].copy()
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = dpi / 72.0
    x0 = int(coords.x * scale)
    y0 = int(coords.y * scale)
    x1 = int((coords.x + coords.w) * scale)
    y1 = int((coords.y + coords.h) * scale)

    draw.rectangle([x0, y0, x1, y1], fill=_HIGHLIGHT_COLOR, outline=_HIGHLIGHT_OUTLINE, width=2)

    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay).convert('RGB')

    pad = int(padding_pt * scale)
    img_w, img_h = img.size
    crop = img.crop((
        max(0, x0 - pad), max(0, y0 - pad),
        min(img_w, x1 + pad), min(img_h, y1 + pad),
    ))

    buf = io.BytesIO()
    crop.save(buf, format='PNG')
    return buf.getvalue()


def screenshot_relpath(corpus: str, record_id: str, gold_value: str) -> str:
    return f'{_SCREENSHOTS_SUBDIR}/{corpus}/{record_id}/{_value_slug(gold_value)}.png'


def render_context_screenshots(
    out_dir: Path,
    items: List[_ScreenshotItem],
) -> Dict[str, bytes]:
    """Generate PNG screenshots for the given items.

    Each item is (corpus, record_id, gold_value, context_window).
    Returns {relative_filepath: png_bytes} for successfully generated screenshots.
    Silently skips items where the TEI or PDF is not found or no element matches.
    """
    results: Dict[str, bytes] = {}
    for corpus, record_id, gold_value, ctx, predicted_label in items:
        doc_dir = out_dir / 'by-doc' / corpus / record_id
        tei_path = doc_dir / f'{record_id}.tei.xml'
        pdf_path = doc_dir / f'{record_id}.pdf'
        if not tei_path.exists() or not pdf_path.exists():
            continue
        coords = find_coords_for_value(tei_path, gold_value, ctx, predicted_label)
        if coords is None:
            continue
        png = render_region_screenshot(pdf_path, coords)
        if png is not None:
            relpath = screenshot_relpath(corpus, record_id, gold_value)
            results[relpath] = png
    return results


def collect_tei_snippets(
    out_dir: Path,
    items: List[_ScreenshotItem],
) -> Dict[str, str]:
    """Return {screenshot_relpath: tei_xml_snippet} for items with a matching TEI element.

    Uses the same element-selection logic as render_context_screenshots so the
    snippet always corresponds to the highlighted region in the screenshot.
    """
    results: Dict[str, str] = {}
    for corpus, record_id, gold_value, ctx, predicted_label in items:
        doc_dir = out_dir / 'by-doc' / corpus / record_id
        tei_path = doc_dir / f'{record_id}.tei.xml'
        if not tei_path.exists():
            continue
        snippet = find_tei_snippet_for_value(tei_path, gold_value, ctx, predicted_label)
        if snippet:
            relpath = screenshot_relpath(corpus, record_id, gold_value)
            results[relpath] = snippet
    return results
