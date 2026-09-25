from __future__ import annotations

from io import BytesIO

import pytest

from sciencebeam_judge.evaluation.scoring_types.scoring_types import resolve_scoring_type
from sciencebeam_judge.parsing.xml import parse_xml

from benchmarks.judge_setup import prepare_judge
from benchmarks.best_match_scoring import (
    BEST_MATCH_FROM_FIRST_SCORING_TYPE,
    BEST_MATCH_FROM_FIRST_SCORING_TYPE_NAME,
    BEST_MATCH_SCORING_TYPE_NAME,
    BEST_MATCH_SCORING_TYPE,
)

GOLD_JATS = b"""<article>
  <front>
    <article-meta>
      <title-group><article-title>The title</article-title></title-group>
      <abstract>the article's own abstract</abstract>
      <trans-abstract xml:lang="en">the translated abstract</trans-abstract>
      <abstract abstract-type="plain-language-summary">the summary</abstract>
    </article-meta>
  </front>
</article>"""


class TestPrepareJudge:
    def test_should_register_the_variants_scoring_type(self):
        prepare_judge()
        assert resolve_scoring_type(BEST_MATCH_SCORING_TYPE_NAME) is BEST_MATCH_SCORING_TYPE

    def test_should_register_the_best_match_from_first_scoring_type(self):
        prepare_judge()
        assert resolve_scoring_type(
            BEST_MATCH_FROM_FIRST_SCORING_TYPE_NAME
        ) is BEST_MATCH_FROM_FIRST_SCORING_TYPE

    def test_should_read_the_abstract_as_the_article_own(self):
        xml_mapping = prepare_judge()
        values = parse_xml(BytesIO(GOLD_JATS), xml_mapping, fields=["abstract"])
        assert values["abstract"] == ["the article's own abstract"]

    def test_should_read_the_abstracts_as_every_language(self):
        xml_mapping = prepare_judge()
        values = parse_xml(BytesIO(GOLD_JATS), xml_mapping, fields=["abstracts"])
        assert values["abstracts"] == [
            "the article's own abstract", "the translated abstract"
        ]

    def test_should_keep_the_fields_the_judge_maps_itself(self):
        xml_mapping = prepare_judge()
        values = parse_xml(BytesIO(GOLD_JATS), xml_mapping, fields=["title"])
        assert values["title"] == ["The title"]


ARTICLE_META_SHAPES = {
    "one abstract": "<abstract>only</abstract>",
    "a translation": (
        "<abstract>main</abstract><trans-abstract xml:lang='en'>translated</trans-abstract>"
    ),
    "a translation filed first": (
        "<trans-abstract xml:lang='en'>translated</trans-abstract><abstract>main</abstract>"
    ),
    "three languages": (
        "<abstract>main</abstract>"
        "<trans-abstract xml:lang='en'>english</trans-abstract>"
        "<trans-abstract xml:lang='es'>spanish</trans-abstract>"
    ),
    "a repeated abstract in another language": (
        "<abstract xml:lang='es' abstract-type='short'>spanish</abstract>"
        "<abstract xml:lang='en' abstract-type='short'>english</abstract>"
    ),
    "a plain-language summary": (
        "<abstract>main</abstract>"
        "<abstract abstract-type='plain-language-summary'>summary</abstract>"
    ),
    "no abstract": "<title-group><article-title>The title</article-title></title-group>",
}


class TestTheFirstAbstractIsTheArticleOwn:
    """The contract `variants` and `first_variant` rely on: whatever else `abstracts` holds,
    its first value is what `abstract` reads on its own."""

    @pytest.mark.parametrize("shape", sorted(ARTICLE_META_SHAPES))
    def test_should_hold_for_every_shape_the_corpora_carry(self, shape: str):
        xml_mapping = prepare_judge()
        article = (
            f"<article><front><article-meta>{ARTICLE_META_SHAPES[shape]}"
            "</article-meta></front></article>"
        ).encode("utf-8")
        values = parse_xml(BytesIO(article), xml_mapping, fields=["abstract", "abstracts"])
        assert values["abstracts"][:1] == values["abstract"]
