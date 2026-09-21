# pylint: disable=too-many-lines
import logging
import os
from typing import List, Optional

from typing_extensions import Protocol

import pytest

from lxml import etree
from lxml.builder import ElementMaker

from sciencebeam_parser.utils.xml import get_text_content
from sciencebeam_parser.document.semantic_document import SemanticDocument
from sciencebeam_parser.document.tei.attribution import DocumentAttribution
from sciencebeam_parser.document.tei_document import get_tei_for_semantic_document
from sciencebeam_parser.transformers.xslt import XsltTransformerWrapper
from sciencebeam_parser.resources.xslt import XSLT_DIR


LOGGER = logging.getLogger(__name__)

DEFAULT_TEI_TO_JATS_XSLT_PATH = os.path.join(XSLT_DIR, 'tei-to-jats.xsl')

XML_NS = 'http://www.w3.org/XML/1998/namespace'
TEI_NS = 'http://www.tei-c.org/ns/1.0'

TEI_E = ElementMaker(namespace=TEI_NS, nsmap={'xml': 'xml', 'tei': TEI_NS})

XML_ID = '{%s}id' % XML_NS

VALUE_1 = 'value 1'
VALUE_2 = 'value 2'
VALUE_3 = 'value 3'

VERSION_1 = 'version 1'
PROFILE_NAME_1 = 'profile 1'
PROFILE_DIGEST_1 = 'digest 1'

FIRST_NAME_1 = 'first name 1'
LAST_NAME_1 = 'last name 1'
EMAIL_1 = 'email@me.org'

FIRST_NAME_2 = 'first name 2'
LAST_NAME_2 = 'last name 2'
EMAIL_2 = 'email@you.org'

AUTHOR_1 = {
    'first-name': FIRST_NAME_1,
    'last-name': LAST_NAME_1
}

AUTHOR_2 = {
    'first-name': FIRST_NAME_2,
    'last-name': LAST_NAME_2
}

AFFILIATION_1 = {
    'key': 'aff1',
    'department': 'Department of Science',
    'laboratory': 'Data Lab',
    'institution': 'Institute 1',
    'city': 'New London',
    'country': 'Country 1'
}

AFFILIATION_2 = {
    'key': 'aff2',
    'department': 'Department 2',
    'laboratory': 'Lab 2',
    'institution': 'Institute 2',
    'city': 'New New London',
    'country': 'Country 2'
}

ARTICLE_TITLE_1 = 'Article title 1'
COLLECTION_TITLE_1 = 'Collection title 1'

REFERENCE_1 = {
    'id': 'b0',
    'article_title': ARTICLE_TITLE_1,
    'journal_title': 'Journal 1',
    'year': '2018',
    'doi': '10.1234/doi1',
    'volume': 'volume1',
    'issue': 'issue1'
}

XLINK_NS = 'http://www.w3.org/1999/xlink'

NAMESPACES = {
    'xlink': XLINK_NS
}


def extend_dict(dict1: dict, **kwargs) -> dict:
    return {**dict1, **kwargs}


class T_TeiToJatsXsltFn(Protocol):
    def __call__(self, xml: str, template_arguments: Optional[dict] = None) -> str:
        pass


@pytest.fixture(name='tei_to_jats_xslt_fn', scope='session')
def _tei_to_jats_xslt_fn():
    transformer = XsltTransformerWrapper.from_template_file(DEFAULT_TEI_TO_JATS_XSLT_PATH)

    def wrapper(xml, *args, **kwargs):
        xml_str = etree.tostring(xml)
        LOGGER.debug('tei: %s', etree.tostring(xml, pretty_print=True))
        result = etree.tostring(transformer(etree.fromstring(xml_str), *args, **kwargs))
        LOGGER.debug('jats: %s', etree.tostring(etree.fromstring(result), pretty_print=True))
        return result
    return wrapper


def _tei(
    titleStmt: Optional[etree.ElementBase] = None,
    biblStruct: Optional[etree.ElementBase] = None,
    authors: Optional[List[etree.ElementBase]] = None,
    body: Optional[etree.ElementBase] = None,
    back: Optional[etree.ElementBase] = None,
    references: Optional[List[etree.ElementBase]] = None,
    application: Optional[etree.ElementBase] = None
) -> etree.ElementBase:
    if authors is None:
        authors = []
    fileDesc = TEI_E.fileDesc()
    if titleStmt is not None:
        fileDesc.append(titleStmt)
    if biblStruct is None:
        biblStruct = TEI_E.biblStruct(
            TEI_E.analytic(
                *authors
            )
        )
    tei_text = TEI_E.text()
    if body is not None:
        tei_text.append(body)
    if back is None:
        back = TEI_E.back()
    tei_text.append(back)
    if references is not None:
        back.append(
            TEI_E.div(
                TEI_E.listBibl(
                    *references
                )
            )
        )
    fileDesc.append(
        TEI_E.sourceDesc(
            biblStruct
        )
    )
    teiHeader = TEI_E.teiHeader(fileDesc)
    if application is not None:
        teiHeader.append(TEI_E.encodingDesc(TEI_E.appInfo(application)))
    return TEI_E.TEI(
        teiHeader,
        tei_text
    )


def _application(
    version: str = VERSION_1,
    profile_name: Optional[str] = PROFILE_NAME_1,
    profile_digest: Optional[str] = PROFILE_DIGEST_1
) -> etree.ElementBase:
    application = TEI_E.application(ident='sciencebeam-parser', version=version)
    if profile_name is not None:
        application.append(TEI_E.label(profile_name, type='profile'))
    if profile_digest is not None:
        application.append(TEI_E.label(profile_digest, type='profile-digest'))
    return application


def _author(forenames=None, surname=LAST_NAME_1, email=EMAIL_1, affiliation=None):
    if forenames is None:
        forenames = [FIRST_NAME_1]
    author = TEI_E.author()
    persName = TEI_E.persName()
    author.append(persName)
    for i, forename in enumerate(forenames):
        persName.append(TEI_E.forename(forename, type=(
            'first' if i == 0 else 'middle'
        )))
    if surname:
        persName.append(TEI_E.surname(surname))
    if email:
        author.append(TEI_E.email(email))
    if affiliation is not None:
        author.append(affiliation)
    return author


def _author_affiliation(**kwargs):
    props = kwargs
    affiliation = TEI_E.affiliation()
    if 'key' in props:
        affiliation.attrib['key'] = props['key']
    if 'department' in props:
        affiliation.append(TEI_E.orgName(props['department'], type='department'))
    if 'laboratory' in props:
        affiliation.append(TEI_E.orgName(props['laboratory'], type='laboratory'))
    if 'institution' in props:
        affiliation.append(TEI_E.orgName(props['institution'], type='institution'))
    address = TEI_E.address()
    affiliation.append(address)
    if 'city' in props:
        address.append(TEI_E.settlement(props['city']))
    if 'country' in props:
        address.append(TEI_E.country(props['country']))
    return affiliation


def _reference(**kwargs):  # pylint: disable=too-many-branches
    props = kwargs
    bibl_struct = TEI_E.biblStruct()
    if 'id' in props:
        bibl_struct.attrib['{xml}id'] = props['id']

    analytic = TEI_E.analytic()
    bibl_struct.append(analytic)
    monogr = TEI_E.monogr()
    bibl_struct.append(monogr)
    imprint = TEI_E.imprint()
    monogr.append(imprint)

    title_level = props.get('title_level', 'a')
    if props.get('article_title') is not None:
        analytic.append(
            TEI_E.title(props['article_title'], level=title_level, type='main')
        )
    if 'collection_title' in props:
        monogr.append(
            TEI_E.title(props['collection_title'], level=title_level, type='main')
        )
    if 'journal_title' in props:
        monogr.append(TEI_E.title(props['journal_title'], level='j'))
    if 'year' in props:
        when = props['year']
        if 'month' in props:
            when += '-%s' % props['month']
            if 'day' in props:
                when += '-%s' % props['day']
        imprint.append(TEI_E.date(type='published', when=when))
    if 'volume' in props:
        imprint.append(TEI_E.biblScope(props['volume'], unit='volume'))
    if 'issue' in props:
        imprint.append(TEI_E.biblScope(props['issue'], unit='issue'))
    if 'fpage' in props and 'lpage' in props:
        imprint.append(TEI_E.biblScope(
            {'unit': 'page', 'from': props['fpage'], 'to': props['lpage']}
        ))
    if 'page' in props:
        imprint.append(TEI_E.biblScope(props['page'], unit='page'))
    if 'doi' in props:
        monogr.append(TEI_E.idno(props['doi'], type='doi'))
    if 'article_authors' in props:
        for author in props['article_authors']:
            analytic.append(_author(
                forenames=[author['first-name']],
                surname=author['last-name'],
                email=None
            ))
    if 'collection_authors' in props:
        for author in props['collection_authors']:
            monogr.append(_author(
                forenames=[author['first-name']],
                surname=author['last-name'],
                email=None
            ))
    return bibl_struct


def _xpath(xml, xpath: str):
    return xml.xpath(xpath, namespaces=NAMESPACES)


def _get_item(xml, xpath: str):
    items = _xpath(xml, xpath)
    if not items:
        raise AssertionError('xpath %s did not match any elements in xml %s' % (
            xpath, etree.tostring(xml)
        ))
    assert len(items) == 1
    return items[0]


def _get_text(xml, xpath: str):
    item = _get_item(xml, xpath)
    try:
        return get_text_content(item)
    except AttributeError:
        return str(item)


def _get_custom_meta_value_by_name(jats) -> dict:
    return {
        get_text_content(_get_item(custom_meta, 'meta-name')):
            get_text_content(_get_item(custom_meta, 'meta-value'))
        for custom_meta in _xpath(
            jats, 'front/article-meta/custom-meta-group/custom-meta'
        )
    }


class TestTeiToJatsXslt:
    class TestAttribution:
        def test_should_translate_version_profile_and_digest(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(application=_application())
            ))
            assert _get_custom_meta_value_by_name(jats) == {
                'sciencebeam-parser-version': VERSION_1,
                'sciencebeam-parser-profile': PROFILE_NAME_1,
                'sciencebeam-parser-profile-digest': PROFILE_DIGEST_1
            }

        def test_should_translate_digest_without_profile_name(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(application=_application(profile_name=None))
            ))
            assert _get_custom_meta_value_by_name(jats) == {
                'sciencebeam-parser-version': VERSION_1,
                'sciencebeam-parser-profile-digest': PROFILE_DIGEST_1
            }

        def test_should_share_one_custom_meta_group_with_the_parameters(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(application=_application()),
                {'output_parameters': 'true'}
            ))
            assert len(_xpath(jats, 'front/article-meta/custom-meta-group')) == 1
            assert set(_get_custom_meta_value_by_name(jats)) == {
                'sciencebeam-parser-version',
                'sciencebeam-parser-profile',
                'sciencebeam-parser-profile-digest',
                'xslt-param-acknowledgement_target',
                'xslt-param-annex_target'
            }

        def test_should_not_add_empty_custom_meta_group_without_application(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei()))
            assert _xpath(jats, 'front/article-meta/custom-meta-group') == []

        def test_should_agree_with_the_tei_the_parser_produced(self, tei_to_jats_xslt_fn):
            attribution = DocumentAttribution(
                version=VERSION_1,
                profile_digest=PROFILE_DIGEST_1,
                profile_name=PROFILE_NAME_1
            )
            tei_document = get_tei_for_semantic_document(
                SemanticDocument(), attribution=attribution
            )
            jats = etree.fromstring(tei_to_jats_xslt_fn(tei_document.root))
            assert _get_custom_meta_value_by_name(jats) == {
                'sciencebeam-parser-version': attribution.version,
                'sciencebeam-parser-profile': attribution.profile_name,
                'sciencebeam-parser-profile-digest': attribution.profile_digest
            }

    class TestJournalTitle:
        def test_should_translate_journal_title(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(biblStruct=TEI_E.biblStruct(TEI_E.monogr(
                    TEI_E.title(VALUE_1)
                )))
            ))
            assert _get_text(
                jats, 'front/journal-meta/journal-title-group/journal-title'
            ) == VALUE_1

        def test_should_not_add_journal_title_if_not_in_tei(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei()
            ))
            assert jats.xpath(
                'front/journal-meta/journal-title-group/journal-title'
            ) == []

    class TestArticleTitle:
        def test_should_translate_title(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(VALUE_1)
                ))
            ))
            assert _get_text(
                jats, 'front/article-meta/title-group/article-title'
            ) == VALUE_1

        def test_should_not_include_title_attributes_in_transformed_title_value(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(VALUE_1, attrib1='other')
                ))
            ))
            assert _get_text(
                jats, 'front/article-meta/title-group/article-title'
            ) == VALUE_1

        def test_should_include_values_of_sub_elements(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(
                        TEI_E.before(VALUE_1),
                        VALUE_2,
                        TEI_E.after(VALUE_3)
                    )
                ))
            ))
            assert (
                _get_text(jats, 'front/article-meta/title-group/article-title') ==
                ''.join([VALUE_1, VALUE_2, VALUE_3])
            )

    class TestAuthor:
        def test_should_not_output_contribut_group_without_authors(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[])
            ))
            assert not jats.xpath('front/article-meta/contrib-group')

        def test_should_translate_single_author(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(
                        forenames=[FIRST_NAME_1],
                        surname=LAST_NAME_1,
                        email=EMAIL_1
                    )
                ])
            ))
            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert _get_text(person, './name/given-names') == FIRST_NAME_1
            assert _get_text(person, './name/surname') == LAST_NAME_1
            assert _get_text(person, './email') == EMAIL_1

        def test_should_include_middle_name_in_given_names(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(
                        forenames=[FIRST_NAME_1, FIRST_NAME_2],
                        surname=LAST_NAME_1,
                        email=EMAIL_1
                    )
                ])
            ))
            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert _get_text(
                person, './name/given-names'
            ) == '%s %s' % (FIRST_NAME_1, FIRST_NAME_2)

        def test_should_not_add_email_if_not_in_tei(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(email=None)
                ])
            ))
            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert person.xpath('./email') == []

        def test_should_add_contrib_type_person_attribute(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[_author()])
            ))
            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert person.attrib.get('contrib-type') == 'person'

        def test_should_add_content_type_author_attribute_to_contrib_group(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[_author()])
            ))
            person = _get_item(jats, 'front/article-meta/contrib-group')
            assert person.attrib.get('content-type') == 'author'

        def test_should_translate_multiple_authors(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(
                        forenames=[FIRST_NAME_1],
                        surname=LAST_NAME_1,
                        email=EMAIL_1
                    ),
                    _author(
                        forenames=[FIRST_NAME_2],
                        surname=LAST_NAME_2,
                        email=EMAIL_2
                    )
                ])
            ))
            persons = jats.xpath('front/article-meta/contrib-group/contrib')
            assert _get_text(persons[0], './name/surname') == LAST_NAME_1
            assert _get_text(persons[1], './name/surname') == LAST_NAME_2

    class TestAuthorAffiliation:
        def test_should_add_affiliation_of_single_author_with_xref(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(affiliation=_author_affiliation(**AFFILIATION_1))
                ])
            ))

            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert (
                _get_item(
                    person, './xref[@ref-type="aff"]'
                ).attrib.get('rid') == AFFILIATION_1['key']
            )

            aff = _get_item(jats, 'front/article-meta/aff')
            assert aff.attrib.get('id') == AFFILIATION_1['key']
            assert (
                _get_text(
                    aff, 'institution[@content-type="orgname"]'
                ) == AFFILIATION_1['institution']
            )
            assert (
                _get_text(
                    aff, 'institution[@content-type="orgdiv1"]'
                ) == AFFILIATION_1['department']
            )
            assert (
                _get_text(
                    aff, 'institution[@content-type="orgdiv2"]'
                ) == AFFILIATION_1['laboratory']
            )
            assert _get_text(aff, 'city') == AFFILIATION_1['city']
            assert _get_text(aff, 'country') == AFFILIATION_1['country']

        def test_should_not_add_affiliation_fields_not_in_tei(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(affiliation=_author_affiliation(
                        key=AFFILIATION_1['key']
                    ))
                ])
            ))

            aff = _get_item(jats, 'front/article-meta/aff')
            assert aff.xpath('institution[@content-type="orgname"]') == []
            assert aff.xpath('institution[@content-type="orgdiv1"]') == []
            assert aff.xpath('institution[@content-type="orgdiv2"]') == []
            assert aff.xpath('city') == []
            assert aff.xpath('country') == []

        def test_should_not_add_affiliation_if_not_in_tei(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[_author()])
            ))

            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert person.xpath('./xref[@ref-type="aff"]') == []

            assert jats.xpath('front/article-meta/aff') == []

        def test_should_add_multiple_affiliations(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(affiliation=_author_affiliation(**AFFILIATION_1)),
                    _author(affiliation=_author_affiliation(**AFFILIATION_2))
                ])
            ))

            persons = jats.xpath('front/article-meta/contrib-group/contrib')
            assert (
                _get_item(
                    persons[0], './xref[@ref-type="aff"]'
                ).attrib.get('rid') == AFFILIATION_1['key']
            )
            assert (
                _get_item(
                    persons[1], './xref[@ref-type="aff"]'
                ).attrib.get('rid') == AFFILIATION_2['key']
            )

            affs = jats.xpath('front/article-meta/aff')
            assert affs[0].attrib.get('id') == AFFILIATION_1['key']
            assert affs[1].attrib.get('id') == AFFILIATION_2['key']

    class TestBody:
        def test_should_add_body(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei()
            ))
            assert _get_item(jats, 'body') is not None

        def test_should_extract_head_and_p_divs(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(VALUE_1),
                    TEI_E.p(VALUE_2)
                )))
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert _get_text(jats, 'body/sec/p') == VALUE_2

        def test_should_add_italics_formatting_to_head_and_p_divs(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(TEI_E.hi({'rend': 'italic'}, VALUE_1)),
                    TEI_E.p(TEI_E.hi({'rend': 'italic'}, VALUE_2))
                ))),
                {'output_italic': 'true'}
            ))
            assert _get_text(jats, 'body/sec/title/i') == VALUE_1
            assert _get_text(jats, 'body/sec/p/i') == VALUE_2

        def test_should_not_add_italics_formatting_if_disabled(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(TEI_E.hi({'rend': 'italic'}, VALUE_1)),
                    TEI_E.p(TEI_E.hi({'rend': 'italic'}, VALUE_2))
                ))),
                {'output_italic': 'false'}
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert not jats.xpath('body/sec/title/i')
            assert _get_text(jats, 'body/sec/p') == VALUE_2
            assert not jats.xpath('body/sec/p/i')

        def test_should_add_bold_formatting_to_head_and_p_divs(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(TEI_E.hi({'rend': 'bold'}, VALUE_1)),
                    TEI_E.p(TEI_E.hi({'rend': 'bold'}, VALUE_2))
                ))),
                {'output_bold': 'true'}
            ))
            assert _get_text(jats, 'body/sec/title/b') == VALUE_1
            assert _get_text(jats, 'body/sec/p/b') == VALUE_2

        def test_should_not_add_bold_formatting_to_head_and_p_divs_if_disabled(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(TEI_E.hi({'rend': 'bold'}, VALUE_1)),
                    TEI_E.p(TEI_E.hi({'rend': 'bold'}, VALUE_2))
                ))),
                {'output_bold': 'false'}
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert not jats.xpath('body/sec/p/b')
            assert _get_text(jats, 'body/sec/p') == VALUE_2
            assert not jats.xpath('body/sec/p/b')

        def test_should_extract_figures_with_graphic_having_url(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.div(
                        TEI_E.p(TEI_E.ref('(Figure 1)', type='figure', target='#fig_0'))
                    ),
                    TEI_E.figure(
                        {XML_ID: 'fig_0'},
                        TEI_E.head('Figure 1'),
                        TEI_E.label('1'),
                        TEI_E.figDesc('Figure 1. This is the figure'),
                        TEI_E.graphic({
                            'url': 'image1.png'
                        })
                    )
                ))
            ))
            assert _get_text(jats, 'body/sec/fig/@id') == 'fig_0'
            assert _get_text(jats, 'body/sec/fig/object-id') == 'fig_0'
            assert _get_text(jats, 'body/sec/fig/label') == 'Figure 1'
            assert _get_text(jats, 'body/sec/fig/caption/title') == 'Figure 1'
            assert _get_text(jats, 'body/sec/fig/caption/p') == 'Figure 1. This is the figure'
            assert _get_item(jats, 'body/sec/fig/graphic') is not None
            assert _get_text(jats, 'body/sec/fig/graphic/@xlink:href ') == 'image1.png'
            assert _get_text(jats, 'body/sec/p/xref') == '(Figure 1)'
            assert _get_text(jats, 'body/sec/p/xref/@ref-type') == 'fig'
            assert _get_text(jats, 'body/sec/p/xref/@rid') == 'fig_0'
            assert not _xpath(jats, '//title/title')

        def test_should_extract_figures_with_graphic_not_having_url(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure(
                        TEI_E.graphic()
                    )
                ))
            ))
            assert _get_item(jats, 'body/sec/fig/graphic') is not None
            assert _xpath(jats, 'body/sec/fig/graphic/@xlink:href ') == []

        def test_should_create_empty_graphic_for_figures_without_graphic(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure({XML_ID: 'fig_0'})
                ))
            ))
            assert _get_text(jats, 'body/sec/fig/@id') == 'fig_0'
            assert _get_item(jats, 'body/sec/fig/graphic') is not None

        def test_should_extract_tables(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.div(
                        TEI_E.p(TEI_E.ref('(Table 1)', type='table', target='#tab_0'))
                    ),
                    TEI_E.figure(
                        {
                            'type': 'table',
                            XML_ID: 'tab_0'
                        },
                        TEI_E.head('Table 1'),
                        TEI_E.label('1'),
                        TEI_E.figDesc('Table 1. This is a table'),
                        TEI_E.table('Table content')
                    )
                ))
            ))
            assert _get_text(jats, 'body/sec/table-wrap/@id') == 'tab_0'
            assert _get_text(jats, 'body/sec/table-wrap/label') == 'Table 1'
            assert _get_text(jats, 'body/sec/table-wrap/caption/title') == 'Table 1'
            assert _get_text(jats, 'body/sec/table-wrap/caption/p') == 'Table 1. This is a table'
            assert _get_item(jats, 'body/sec/table-wrap/table') is not None
            assert _get_item(jats, 'body/sec/table-wrap/table/tbody') is not None
            assert _get_item(jats, 'body/sec/table-wrap/table/tbody/tr') is not None
            assert _get_item(jats, 'body/sec/table-wrap/table/tbody/tr/td') is not None
            assert _get_text(jats, 'body/sec/table-wrap/table/tbody/tr/td') == 'Table content'
            assert _get_text(jats, 'body/sec/p/xref') == '(Table 1)'
            assert _get_text(jats, 'body/sec/p/xref/@ref-type') == 'table'
            assert _get_text(jats, 'body/sec/p/xref/@rid') == 'tab_0'
            assert not _xpath(jats, '//title/title')

        def test_should_extract_bibr_ref(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.p(TEI_E.ref('Some ref', type='bibr', target='#b0'))
                )))
            ))
            assert _get_text(jats, 'body/sec/p') == 'Some ref'
            assert _get_text(jats, 'body/sec/p/xref') == 'Some ref'
            assert _get_text(jats, 'body/sec/p/xref/@ref-type') == 'bibr'
            assert _get_text(jats, 'body/sec/p/xref/@rid') == 'b0'

        def test_should_extract_bibr_ref_without_target_as_text(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.p(TEI_E.ref('Some ref', type='bibr'))
                )))
            ))
            assert _get_text(jats, 'body/sec/p') == 'Some ref'
            assert not jats.xpath('body/sec/p/xref')

        def test_should_extract_unknown_ref_as_text(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.p(TEI_E.ref('Some ref', type='other', target='#other'))
                )))
            ))
            assert _get_text(jats, 'body/sec/p') == 'Some ref'
            assert not jats.xpath('body/sec/p/xref')

    class TestBack:
        def test_should_add_back(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei()
            ))
            assert _get_item(jats, 'back') is not None

        def test_should_extract_acknowledgement_head_and_p_divs_as_ack(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'acknowledgement'},
                        TEI_E.div(
                            TEI_E.head(VALUE_1),
                            TEI_E.p(VALUE_2)
                        )
                    )
                )),
                {
                    'acknowledgement_target': 'ack'
                }
            ))
            assert _get_text(jats, 'back/ack/sec/title') == VALUE_1
            assert _get_text(jats, 'back/ack/sec/p') == VALUE_2

        def test_should_extract_acknowledgement_head_and_p_divs_as_body(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'acknowledgement'},
                        TEI_E.div(
                            TEI_E.head(VALUE_1),
                            TEI_E.p(VALUE_2)
                        )
                    )
                )),
                {
                    'acknowledgement_target': 'body'
                }
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert _get_text(jats, 'body/sec/p') == VALUE_2

        def test_should_extract_annex_head_and_p_divs_as_back_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.div(
                            TEI_E.head(VALUE_1),
                            TEI_E.p(VALUE_2)
                        )
                    )
                )),
                {
                    'annex_target': 'back'
                }
            ))
            assert _get_text(jats, 'back/sec/title') == VALUE_1
            assert _get_text(jats, 'back/sec/p') == VALUE_2

        def test_should_extract_annex_head_and_p_divs_as_body(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.div(
                            TEI_E.head(VALUE_1),
                            TEI_E.p(VALUE_2)
                        )
                    )
                )),
                {
                    'annex_target': 'body'
                }
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert _get_text(jats, 'body/sec/p') == VALUE_2

        def test_should_extract_annex_head_and_p_divs_as_app_group(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.div(
                            TEI_E.head(VALUE_1),
                            TEI_E.p(VALUE_2)
                        )
                    )
                )),
                {
                    'annex_target': 'app'
                }
            ))
            assert _get_text(jats, 'back/app-group/app/sec/title') == VALUE_1
            assert _get_text(jats, 'back/app-group/app/sec/p') == VALUE_2

        def test_should_extract_annex_figures_as_back_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            TEI_E.head('Figure 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Figure 1. This is the figure')
                        )
                    )
                )),
                {
                    'annex_target': 'back'
                }
            ))
            assert _get_text(jats, 'back/sec/fig/label') == 'Figure 1'
            assert _get_text(jats, 'back/sec/fig/caption/title') == 'Figure 1'
            assert _get_text(jats, 'back/sec/fig/caption/p') == 'Figure 1. This is the figure'

        def test_should_extract_annex_figures_as_body_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            TEI_E.head('Figure 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Figure 1. This is the figure')
                        )
                    )
                )),
                {
                    'annex_target': 'body'
                }
            ))
            assert _get_text(jats, 'body/sec/fig/label') == 'Figure 1'
            assert _get_text(jats, 'body/sec/fig/caption/title') == 'Figure 1'
            assert _get_text(jats, 'body/sec/fig/caption/p') == 'Figure 1. This is the figure'

        def test_should_extract_annex_figures_as_app_group(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            TEI_E.head('Figure 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Figure 1. This is the figure')
                        )
                    )
                )),
                {
                    'annex_target': 'app'
                }
            ))
            assert _get_text(jats, 'back/app-group/app/fig/label') == 'Figure 1'
            assert _get_text(jats, 'back/app-group/app/fig/caption/title') == 'Figure 1'
            assert _get_text(jats, 'back/app-group/app/fig/caption/p') == (
                'Figure 1. This is the figure'
            )

        def test_should_extract_annex_tables_as_back_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            {'type': 'table'},
                            TEI_E.head('Table 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Table 1. This is the table')
                        )
                    )
                )),
                {
                    'annex_target': 'back'
                }
            ))
            assert _get_text(jats, 'back/sec/table-wrap/label') == 'Table 1'
            assert _get_text(jats, 'back/sec/table-wrap/caption/title') == 'Table 1'
            assert _get_text(jats, 'back/sec/table-wrap/caption/p') == 'Table 1. This is the table'

        def test_should_extract_annex_tables_as_body_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            {'type': 'table'},
                            TEI_E.head('Table 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Table 1. This is the table')
                        )
                    )
                )),
                {
                    'annex_target': 'body'
                }
            ))
            assert _get_text(jats, 'body/sec/table-wrap/label') == 'Table 1'
            assert _get_text(jats, 'body/sec/table-wrap/caption/title') == 'Table 1'
            assert _get_text(jats, 'body/sec/table-wrap/caption/p') == 'Table 1. This is the table'

        def test_should_extract_annex_tables_as_app_group(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            {'type': 'table'},
                            TEI_E.head('Table 1'),
                            TEI_E.label('1'),
                            TEI_E.figDesc('Table 1. This is the table')
                        )
                    )
                )),
                {
                    'annex_target': 'app'
                }
            ))
            assert _get_text(jats, 'back/app-group/app/table-wrap/label') == 'Table 1'
            assert _get_text(jats, 'back/app-group/app/table-wrap/caption/title') == 'Table 1'
            assert _get_text(jats, 'back/app-group/app/table-wrap/caption/p') == (
                'Table 1. This is the table'
            )

    class TestReferences:
        def test_should_convert_single_reference(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**REFERENCE_1)])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert ref.attrib.get('id') == REFERENCE_1['id']
            assert element_citation.attrib.get('publication-type') == 'journal'
            assert _get_text(
                element_citation,
                'article-title'
            ) == REFERENCE_1['article_title']
            assert _get_text(element_citation, 'year') == REFERENCE_1['year']
            assert _get_text(
                element_citation, 'source'
            ) == REFERENCE_1['journal_title']
            assert _get_text(
                element_citation, 'volume'
            ) == REFERENCE_1['volume']
            assert _get_text(element_citation, 'issue') == REFERENCE_1['issue']
            assert _get_text(
                element_citation, 'pub-id[@pub-id-type="doi"]'
            ) == REFERENCE_1['doi']

        def test_should_fallback_to_collection_title_if_article_title_does_not_exist(
                self, tei_to_jats_xslt_fn):

            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, article_title=None, collection_title=COLLECTION_TITLE_1
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(
                element_citation, 'article-title'
            ) == COLLECTION_TITLE_1

        def test_should_only_return_article_title_even_if_collection_title_exists(
                self, tei_to_jats_xslt_fn):

            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, article_title=ARTICLE_TITLE_1, collection_title=COLLECTION_TITLE_1
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(
                element_citation,
                'article-title'
            ) == ARTICLE_TITLE_1

        @pytest.mark.parametrize('title_level', ['a', 'm'])
        def test_should_only_return_article_title_at_different_levels(
                self, tei_to_jats_xslt_fn, title_level):

            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, article_title=ARTICLE_TITLE_1, title_level=title_level
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(
                element_citation, 'article-title'
            ) == ARTICLE_TITLE_1

        def test_should_convert_page_range(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, fpage='fpage', lpage='lpage'
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(element_citation, 'fpage') == 'fpage'
            assert _get_text(element_citation, 'lpage') == 'lpage'

        def test_should_convert_single_page_no(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, page='page1'
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(element_citation, 'fpage') == 'page1'
            assert _get_text(element_citation, 'lpage') == 'page1'

        def test_should_convert_year_and_month(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, year='2001', month='02'
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(element_citation, 'year') == '2001'
            assert _get_text(element_citation, 'month') == '02'

        def test_should_convert_year_month_and_day(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, year='2001', month='02', day='03'
                ))])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')

            assert _get_text(element_citation, 'year') == '2001'
            assert _get_text(element_citation, 'month') == '02'
            assert _get_text(element_citation, 'day') == '03'

        def test_should_convert_multiple_article_authors_of_single_reference(
                self, tei_to_jats_xslt_fn):
            authors = [AUTHOR_1, AUTHOR_2]
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[
                    _reference(**extend_dict(
                        REFERENCE_1,
                        article_authors=authors
                    ))
                ])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')
            person_group = _get_item(element_citation, 'person-group')
            persons = person_group.xpath('name')
            assert len(persons) == 2

            for person, author in zip(persons, authors):
                assert _get_text(person, 'surname') == author['last-name']
                assert _get_text(person, 'given-names') == author['first-name']

        def test_should_convert_multiple_collection_authors_of_single_reference(
                self, tei_to_jats_xslt_fn):
            authors = [AUTHOR_1, AUTHOR_2]
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[
                    _reference(**extend_dict(
                        REFERENCE_1,
                        collection_authors=authors
                    ))
                ])
            ))

            ref_list = _get_item(jats, 'back/ref-list')
            ref = _get_item(ref_list, 'ref')
            element_citation = _get_item(ref, 'element-citation')
            person_group = _get_item(element_citation, 'person-group')
            persons = person_group.xpath('name')
            assert len(persons) == 2

            for person, author in zip(persons, authors):
                assert _get_text(person, 'surname') == author['last-name']
                assert _get_text(person, 'given-names') == author['first-name']
