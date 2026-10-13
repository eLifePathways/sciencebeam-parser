"""Which language a passage of prose is written in, by stopword frequency.

Reliable at abstract length over the languages the corpora use, and silent below it:
`detect_language` returns `None` rather than guess, so nothing declares a language the
text does not support.
"""

import re
from typing import AbstractSet, Mapping, Optional


# The most frequent function words of each language, which prose of any subject carries
# and which overlap little between these six.
STOPWORDS_BY_LANGUAGE: Mapping[str, str] = {
    'en': 'the of and to in a is that for was were with this are as be by on from at an it',
    'pt': 'de a o que e do da em para os as com uma um no na por se foi dos das como mais ao',
    'es': 'de la que el en y a los del se las por un para con no una su al es lo como mas',
    'fr': 'de la le et les des en un une du dans que pour qui par sur au avec est ne pas ce',
    'it': 'di e il la che in un a per sono con non una su le dei della come piu anche nel',
    'de': 'der die und in den von zu das mit sich des auf fur ist im dem nicht ein eine als'
}

STOPWORD_SET_BY_LANGUAGE: Mapping[str, AbstractSet[str]] = {
    language: frozenset(stopwords.split())
    for language, stopwords in STOPWORDS_BY_LANGUAGE.items()
}

MIN_LANGUAGE_DETECTION_WORD_COUNT = 20

# How far ahead the winner has to be, proportionally and absolutely. A passage that is
# mostly names or formulae fails the first; one written in two languages at once, such as
# a title block giving both, fails the second -- over a benchmark run the closest real
# abstract led by 7 and a bilingual block led by 3.
MIN_LANGUAGE_DETECTION_RATIO = 1.25
MIN_LANGUAGE_DETECTION_MARGIN = 5

WORD_PATTERN = re.compile(r'[^\W\d_]+', re.UNICODE)


def detect_language(text: str) -> Optional[str]:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    if len(words) < MIN_LANGUAGE_DETECTION_WORD_COUNT:
        return None
    count_by_language = {
        language: sum(1 for word in words if word in stopwords)
        for language, stopwords in STOPWORD_SET_BY_LANGUAGE.items()
    }
    best_language = max(count_by_language, key=lambda language: count_by_language[language])
    best_count = count_by_language[best_language]
    if not best_count:
        return None
    runner_up = max(
        count for language, count in count_by_language.items() if language != best_language
    )
    if best_count - runner_up < MIN_LANGUAGE_DETECTION_MARGIN:
        return None
    if best_count < MIN_LANGUAGE_DETECTION_RATIO * runner_up:
        return None
    return best_language
