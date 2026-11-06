import unicodedata
from typing import List

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


def starts_with_a_combining_mark(text: str) -> bool:
    """Whether a token opens with an accent belonging to the word before it.

    A font that carries the accent separately has pdfalto emit it as its own
    token, which then holds the mark and the rest of the word behind it.
    """
    return bool(text) and bool(unicodedata.combining(text[0]))


def without_combining_marks(text: str) -> str:
    """The text with its accents taken off, and a word broken by one closed up.

    An accent the page sets apart from its vowel cannot be composed back on,
    because the two are never in the same string, and a font that prints it on
    the wrong letter puts it somewhere no composing would reach.  Compared
    without their marks the page and the deposit say the same word.

    A mark standing on its own takes the spaces around it with it: a deposit
    carries the same broken spelling as the page it was made from -- `O` then
    the diaeresis then `zer` -- so dropping the mark alone would leave a gap
    on one side and not the other.  A mark written on its letter takes
    nothing, because the space behind it is the space between two words.
    """
    out: List[str] = []
    detached = False
    for character in unicodedata.normalize('NFD', text):
        if unicodedata.combining(character):
            if not out or out[-1].isspace():
                while out and out[-1].isspace():
                    out.pop()
                detached = True
            continue
        if detached:
            if character.isspace():
                continue
            detached = False
        out.append(character)
    return ''.join(out)


def normalize_for_alignment(text: str) -> str:
    """Aggressive normalisation for match scoring: lowercase + collapsed whitespace."""
    text = without_combining_marks(normalize_text(text))
    return ' '.join(text.lower().split())
