import re
from typing import Iterable, List


# delimters mostly copied from:
# https://github.com/kermitt2/delft/blob/v0.2.6/delft/utilities/Tokenizer.py
# and:
# https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/utilities/TextUtilities.java#L773-L948
# added: `@`, `#`, `\u2020`, ...
_COMMON_AFF_MARKERS = '\u2020\u2021\u00A7\u00B6\u204B\u01C2'
DELIMITERS = (
    "\n\r\t\f\u00A0([ •*,:;?.!/#)-−–‐\"“”‘’'`$]*\u2666\u2665\u2663\u2660\u00A0@"
    + _COMMON_AFF_MARKERS
)
DELIMITERS_REGEX = r'(' + r'|'.join(map(re.escape, DELIMITERS)) + r'|\s)'


def iter_tokenized_tokens(text: str, keep_whitespace: bool = False) -> Iterable[str]:
    for token in re.split(DELIMITERS_REGEX, text):
        if not keep_whitespace and not token.strip():
            continue
        yield token


def get_tokenized_tokens(text: str, **kwargs) -> List[str]:
    return list(iter_tokenized_tokens(text, **kwargs))


# Matches GROBID's GrobidDefaultAnalyzer.retokenizeSubdigitsFromLayoutToken:
# splits at letter→digit and digit→non-digit boundaries (e.g. e1006572 → e + 1006572).
_SUBDIGIT_SPLIT_PATTERN = re.compile(r'(?<=[^\W\d_])(?=\d)|(?<=\d)(?=\D)')


def get_subdigit_tokenized_tokens(text: str) -> List[str]:
    return _SUBDIGIT_SPLIT_PATTERN.split(text)
