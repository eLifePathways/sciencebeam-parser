from lxml import etree

from sciencebeam_parser.document.tei.attribution import (
    APPLICATION_IDENT,
    DocumentAttribution,
    get_tei_application_element
)
from sciencebeam_parser.document.tei.common import TEI_NS_MAP


VERSION_1 = '1.2.3'
PROFILE_NAME_1 = 'profile1'
PROFILE_DIGEST_1 = 'digest1'


def _get_label_text_by_type(element: etree.ElementBase) -> dict:
    return {
        label.attrib['type']: label.text
        for label in element.xpath('tei:label', namespaces=TEI_NS_MAP)
    }


class TestGetTeiApplicationElement:
    def test_should_name_the_application_and_its_version(self):
        element = get_tei_application_element(DocumentAttribution(
            version=VERSION_1,
            profile_digest=PROFILE_DIGEST_1,
            profile_name=PROFILE_NAME_1
        ))
        assert element.attrib['ident'] == APPLICATION_IDENT
        assert element.attrib['version'] == VERSION_1

    def test_should_add_profile_name_and_digest_as_labels(self):
        element = get_tei_application_element(DocumentAttribution(
            version=VERSION_1,
            profile_digest=PROFILE_DIGEST_1,
            profile_name=PROFILE_NAME_1
        ))
        assert _get_label_text_by_type(element) == {
            'profile': PROFILE_NAME_1,
            'profile-digest': PROFILE_DIGEST_1
        }

    def test_should_add_digest_without_profile_name(self):
        element = get_tei_application_element(DocumentAttribution(
            version=VERSION_1,
            profile_digest=PROFILE_DIGEST_1
        ))
        assert _get_label_text_by_type(element) == {
            'profile-digest': PROFILE_DIGEST_1
        }
