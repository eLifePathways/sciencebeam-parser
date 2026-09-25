from __future__ import annotations

from io import BytesIO

from sciencebeam_judge.evaluation.scoring_types.scoring_types import resolve_scoring_type
from sciencebeam_judge.parsing.xml import parse_xml

from benchmarks.judge_setup import prepare_judge
from benchmarks.variant_scoring import (
    MAIN_VARIANT_SCORING_TYPE,
    MAIN_VARIANT_SCORING_TYPE_NAME,
    SCORING_TYPE_NAME,
    VARIANTS_SCORING_TYPE,
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
        assert resolve_scoring_type(SCORING_TYPE_NAME) is VARIANTS_SCORING_TYPE

    def test_should_register_the_main_variant_scoring_type(self):
        prepare_judge()
        assert resolve_scoring_type(MAIN_VARIANT_SCORING_TYPE_NAME) is MAIN_VARIANT_SCORING_TYPE

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
