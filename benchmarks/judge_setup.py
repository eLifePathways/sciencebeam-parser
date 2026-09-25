"""The one place that prepares sciencebeam-judge for this benchmark.

Everything reading gold or predictions goes through `prepare_judge`, so the scorer, the
case viewer and the regression analysis cannot disagree about what a field's gold is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from sciencebeam_judge.evaluation.scoring_types.scoring_types import SCORING_TYPE_MAP
from sciencebeam_judge.parsing.xml import parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

from benchmarks.best_match_scoring import (
    BEST_MATCH_FROM_FIRST_SCORING_TYPE,
    BEST_MATCH_FROM_FIRST_SCORING_TYPE_NAME,
    BEST_MATCH_SCORING_TYPE_NAME,
    BEST_MATCH_SCORING_TYPE,
)
from benchmarks.variant_xpath import register_variant_functions

XML_MAPPING_OVERRIDE_PATH = str(Path(__file__).parent / "xml-mapping.conf")


def prepare_judge() -> Dict[str, Dict[str, str]]:
    """Registers what the judge does not ship with, and returns the mapping to parse with."""
    register_functions()
    register_variant_functions()
    SCORING_TYPE_MAP[BEST_MATCH_SCORING_TYPE_NAME] = BEST_MATCH_SCORING_TYPE
    SCORING_TYPE_MAP[BEST_MATCH_FROM_FIRST_SCORING_TYPE_NAME] = BEST_MATCH_FROM_FIRST_SCORING_TYPE
    xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)
    for section, entries in parse_xml_mapping(XML_MAPPING_OVERRIDE_PATH).items():
        xml_mapping.setdefault(section, {}).update(entries)
    return xml_mapping
