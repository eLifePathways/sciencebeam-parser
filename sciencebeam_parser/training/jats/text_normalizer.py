import unicodedata

_LIGATURE_MAP = str.maketrans({
    'ﬀ': 'ff',
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
    'æ': 'ae',
    'œ': 'oe',
})

_DASH_MAP = str.maketrans({
    ch: '-'
    for ch in '‐‑‒–—―'
})

_QUOTE_MAP = str.maketrans({
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
})

_SOFT_HYPHEN = '­'


def normalize_text(text: str) -> str:
    """Light normalisation: ligatures, dashes, quotes, soft hyphens, NFC."""
    text = text.translate(_LIGATURE_MAP)
    text = text.translate(_DASH_MAP)
    text = text.translate(_QUOTE_MAP)
    text = text.replace(_SOFT_HYPHEN, '')
    return unicodedata.normalize('NFC', text)


def normalize_for_alignment(text: str) -> str:
    """Aggressive normalisation for match scoring: lowercase + collapsed whitespace."""
    text = normalize_text(text)
    return ' '.join(text.lower().split())
