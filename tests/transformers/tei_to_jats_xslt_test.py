# pylint: disable=too-many-lines
import logging
import os
from typing import Any, List, Optional

from typing_extensions import Protocol

import pytest

from lxml import etree
from lxml.builder import ElementMaker

from sciencebeam_parser.utils.xml import get_text_content
from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticCountry,
    SemanticDepartment,
    SemanticDocument,
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticGivenName,
    SemanticHeading,
    SemanticInstitution,
    SemanticLabel,
    SemanticMarker,
    SemanticJournal,
    SemanticNote,
    SemanticRawReferenceText,
    SemanticReference,
    SemanticSectionTypes,
    SemanticSurname,
    SemanticTitle,
    SemanticVolume
)
from sciencebeam_parser.document.tei.author import (
    get_tei_affiliation_for_semantic_affiliation_address_element
)
from sciencebeam_parser.document.tei.factories import DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
from sciencebeam_parser.document.tei.references import get_tei_reference_element
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
XML_LANG = '{%s}lang' % XML_NS

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

XML_LANG = '{http://www.w3.org/XML/1998/namespace}lang'

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


def _tei(  # pylint: disable=too-many-arguments
    titleStmt: Optional[etree.ElementBase] = None,
    biblStruct: Optional[etree.ElementBase] = None,
    authors: Optional[List[etree.ElementBase]] = None,
    body: Optional[etree.ElementBase] = None,
    back: Optional[etree.ElementBase] = None,
    references: Optional[List[etree.ElementBase]] = None,
    application: Optional[etree.ElementBase] = None,
    abstracts: Optional[List[Any]] = None,
    abstract_headings: Optional[List[Any]] = None
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
    if abstracts is not None:
        teiHeader.append(TEI_E.profileDesc(*[
            TEI_E.abstract(TEI_E.div(TEI_E.p(abstract[0])), **{XML_LANG: abstract[1]})
            if isinstance(abstract, tuple)
            else TEI_E.abstract(TEI_E.div(TEI_E.p(abstract)))
            for abstract in abstracts
        ]))
    if abstract_headings is not None:
        teiHeader.append(TEI_E.profileDesc(*[
            element
            for heading, abstract in abstract_headings
            for element in (
                [TEI_E.abstract(TEI_E.div(TEI_E.head(heading), TEI_E.p(abstract)))]
                if heading
                else [TEI_E.abstract(TEI_E.div(TEI_E.p(abstract)))]
            )
        ]))
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
    if 'raw' in props:
        label, text = props['raw']
        note = TEI_E.note(type='raw_affiliation')
        if label:
            note.append(TEI_E.label(label))
            note[-1].tail = text
        else:
            note.text = text
        affiliation.append(note)
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
    if 'label' in props:
        bibl_struct.insert(0, TEI_E.note(props['label'], type='label'))
    if 'doi' in props:
        analytic.append(TEI_E.idno(props['doi'], type='DOI'))
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


def _printed_reference(printed: str, *children) -> etree.ElementBase:
    return TEI_E.biblStruct(
        {'{http://www.w3.org/XML/1998/namespace}id': 'b0'},
        TEI_E.note(printed, type='raw_reference'),
        *children
    )


def _person(surname: str, *forenames: str) -> etree.ElementBase:
    return TEI_E.author(TEI_E.persName(
        TEI_E.surname(surname), *[TEI_E.forename(forename) for forename in forenames]
    ))


def _mixed_citation_of(tei_to_jats_xslt_fn, bibl_struct) -> str:
    jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[bibl_struct])))
    mixed_citation = _get_item(jats, 'back/ref-list/ref/mixed-citation')
    return etree.tostring(mixed_citation, encoding='unicode', with_tail=False).split('>', 1)[1]


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
            jats, 'processing-meta/custom-meta-group/custom-meta'
        )
    }


class TestTeiToJatsXslt:
    class TestDeclaration:
        def test_should_declare_jats_1_4_archiving(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei()))
            docinfo = jats.getroottree().docinfo
            assert docinfo.public_id == (
                '-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD'
                ' v1.4 20241031//EN'
            )
            assert docinfo.system_url == (
                'https://jats.nlm.nih.gov/archiving/1.4/JATS-archivearticle1-4.dtd'
            )
            assert jats.attrib.get('dtd-version') == '1.4'

        def test_should_name_the_tag_set_in_processing_meta_before_the_front(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei()))
            processing_meta = jats[0]
            assert processing_meta.tag == 'processing-meta'
            assert processing_meta.attrib == {
                'tagset-family': 'jats', 'base-tagset': 'archiving'
            }
            assert jats[1].tag == 'front'

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
            assert len(_xpath(jats, 'processing-meta/custom-meta-group')) == 1
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
            assert _xpath(jats, 'processing-meta/custom-meta-group') == []

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

    class TestAbstract:
        def test_should_translate_single_abstract(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(abstracts=[VALUE_1])))
            assert [
                get_text_content(node)
                for node in jats.xpath('front/article-meta/abstract')
            ] == [VALUE_1]

        def test_should_give_the_abstract_the_heading_noted_before_it(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(abstract_headings=[
                ('Abstract', VALUE_1), ('Plain language summary', VALUE_2)
            ])))
            assert [
                get_text_content(node)
                for node in jats.xpath('front/article-meta/abstract/title')
            ] == ['Abstract', 'Plain language summary']

        def test_should_add_no_title_where_no_heading_was_noted(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(abstract_headings=[
                (None, VALUE_1)
            ])))
            assert jats.xpath('front/article-meta/abstract/title') == []
            assert [
                get_text_content(node)
                for node in jats.xpath('front/article-meta/abstract')
            ] == [VALUE_1]

        def test_should_keep_the_empty_abstract_where_the_tei_has_none(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei()))
            assert [
                get_text_content(node)
                for node in jats.xpath('front/article-meta/abstract')
            ] == ['']

        def test_should_keep_the_language_of_each_abstract(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(abstracts=[(VALUE_1, 'en'), (VALUE_2, 'pt')])
            ))
            assert [
                node.attrib.get(XML_LANG)
                for node in jats.xpath('front/article-meta/abstract')
            ] == ['en', 'pt']

        def test_should_not_add_a_language_the_tei_does_not_declare(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(abstracts=[VALUE_1])))
            assert XML_LANG not in jats.xpath('front/article-meta/abstract')[0].attrib

        def test_should_translate_further_abstract_as_further_abstract_element(
                self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(abstracts=[VALUE_1, VALUE_2])
            ))
            assert [
                get_text_content(node)
                for node in jats.xpath('front/article-meta/abstract')
            ] == [VALUE_1, VALUE_2]

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

        def test_should_translate_a_further_title_as_a_translation(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(VALUE_1, **{XML_LANG: 'pt'}),
                    TEI_E.title(VALUE_2, **{XML_LANG: 'en'})
                ))
            ))
            assert _get_text(
                jats, 'front/article-meta/title-group/article-title'
            ) == VALUE_1
            trans_group = jats.find(
                'front/article-meta/title-group/trans-title-group'
            )
            assert trans_group.attrib[XML_LANG] == 'en'
            assert _get_text(trans_group, 'trans-title') == VALUE_2

        def test_should_translate_the_language_of_the_title(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(VALUE_1, **{XML_LANG: 'pt'})
                ))
            ))
            title = jats.find('front/article-meta/title-group/article-title')
            assert title.attrib[XML_LANG] == 'pt'

        def test_should_leave_the_language_off_a_title_without_one(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(titleStmt=TEI_E.titleStmt(
                    TEI_E.title(VALUE_1)
                ))
            ))
            title = jats.find('front/article-meta/title-group/article-title')
            assert XML_LANG not in title.attrib

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

        def test_should_add_contrib_type_author_attribute(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[_author()])
            ))
            person = _get_item(
                jats, 'front/article-meta/contrib-group/contrib'
            )
            assert person.attrib.get('contrib-type') == 'author'

        def test_should_not_add_content_type_to_contrib_group(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[_author()])
            ))
            contrib_group = _get_item(jats, 'front/article-meta/contrib-group')
            assert 'content-type' not in contrib_group.attrib

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

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
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

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
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

            assert jats.xpath('front/article-meta/contrib-group/aff') == []

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

            affs = jats.xpath('front/article-meta/contrib-group/aff')
            assert affs[0].attrib.get('id') == AFFILIATION_1['key']
            assert affs[1].attrib.get('id') == AFFILIATION_2['key']

        def test_should_place_affiliations_in_the_contrib_group_after_the_contributors(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(affiliation=_author_affiliation(**AFFILIATION_1)),
                    _author(affiliation=_author_affiliation(**AFFILIATION_2))
                ])
            ))

            contrib_group = _get_item(jats, 'front/article-meta/contrib-group')
            assert [child.tag for child in contrib_group] == [
                'contrib', 'contrib', 'aff', 'aff'
            ]
            assert jats.xpath('front/article-meta/aff') == []

        def test_should_add_a_shared_affiliation_once_with_an_xref_from_each_author(
            self, tei_to_jats_xslt_fn
        ):
            # the TEI writes an affiliation out under each author it is attached to
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(authors=[
                    _author(affiliation=_author_affiliation(**AFFILIATION_1)),
                    _author(affiliation=_author_affiliation(**AFFILIATION_1))
                ])
            ))

            persons = jats.xpath('front/article-meta/contrib-group/contrib')
            assert [
                _get_item(person, './xref[@ref-type="aff"]').attrib.get('rid')
                for person in persons
            ] == [AFFILIATION_1['key'], AFFILIATION_1['key']]

            affs = jats.xpath('front/article-meta/contrib-group/aff')
            assert [aff.attrib.get('id') for aff in affs] == [AFFILIATION_1['key']]

        def test_should_add_an_xref_for_each_affiliation_of_an_author(
            self, tei_to_jats_xslt_fn
        ):
            author = _author(affiliation=_author_affiliation(**AFFILIATION_1))
            author.append(_author_affiliation(**AFFILIATION_2))
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[author])))

            person = _get_item(jats, 'front/article-meta/contrib-group/contrib')
            assert [
                xref.attrib.get('rid')
                for xref in person.xpath('./xref[@ref-type="aff"]')
            ] == [AFFILIATION_1['key'], AFFILIATION_2['key']]

            affs = jats.xpath('front/article-meta/contrib-group/aff')
            assert [aff.attrib.get('id') for aff in affs] == [
                AFFILIATION_1['key'], AFFILIATION_2['key']
            ]

        def test_should_keep_the_printed_affiliation_with_its_parts_tagged_in_place(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(**AFFILIATION_1, raw=(
                    '1', ' Department of Science, Data Lab, Institute 1, New London, Country 1'
                )))
            ])))

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert _get_text(aff, 'label') == '1'
            assert get_text_content(aff) == (
                '1 Department of Science, Data Lab, Institute 1, New London, Country 1'
            )
            assert [
                (child.tag, child.attrib.get('content-type'), child.text)
                for child in aff
            ] == [
                ('label', None, '1'),
                ('institution', 'orgdiv1', 'Department of Science'),
                ('institution', 'orgdiv2', 'Data Lab'),
                ('institution', 'orgname', 'Institute 1'),
                ('city', None, 'New London'),
                ('country', None, 'Country 1')
            ]

        def test_should_tag_parts_in_printed_order(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    key='aff1', institution='Institute 1', city='New London',
                    raw=(None, 'New London: Institute 1')
                ))
            ])))

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert [child.tag for child in aff] == ['city', 'institution']
            assert get_text_content(aff) == 'New London: Institute 1'

        def test_should_tag_the_longer_part_where_two_start_together(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    key='aff1', institution='New London Hospital', city='New London',
                    raw=(None, 'New London Hospital, New London')
                ))
            ])))

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert [(child.tag, child.text) for child in aff] == [
                ('institution', 'New London Hospital'),
                ('city', 'New London')
            ]

        def test_should_keep_printed_text_that_no_part_accounts_for(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    key='aff1', institution='Institute One', country='Country 1',
                    raw=(None, 'Institute 1, Country 1')
                ))
            ])))

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert [child.tag for child in aff] == ['country']
            assert get_text_content(aff) == 'Institute 1, Country 1'

        def test_should_separate_parts_where_the_printed_text_is_missing(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    key='aff1', institution='Institute 1', country='Country 1'
                ))
            ])))

            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert get_text_content(aff) == 'Institute 1, Country 1'

        def test_should_give_the_author_xref_the_printed_label(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    **AFFILIATION_1, raw=('a', ' Institute 1')
                ))
            ])))

            person = _get_item(jats, 'front/article-meta/contrib-group/contrib')
            assert _get_text(person, './xref[@ref-type="aff"]') == 'a'

        def test_should_invent_no_label(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[
                _author(affiliation=_author_affiliation(
                    **AFFILIATION_1, raw=(None, 'Institute 1')
                ))
            ])))

            person = _get_item(jats, 'front/article-meta/contrib-group/contrib')
            assert _get_text(person, './xref[@ref-type="aff"]') == ''
            assert jats.xpath('front/article-meta/contrib-group/aff/label') == []

        def test_should_carry_the_affiliation_the_parser_produced(
            self, tei_to_jats_xslt_fn
        ):
            semantic_affiliation = SemanticAffiliationAddress(content_id='aff0')
            for semantic_content in [
                SemanticMarker(layout_block=LayoutBlock.for_text('1')),
                SemanticDepartment(layout_block=LayoutBlock.for_text('Department of Science')),
                SemanticNote(layout_block=LayoutBlock.for_text(',')),
                SemanticInstitution(layout_block=LayoutBlock.for_text('Institute 1')),
                SemanticNote(layout_block=LayoutBlock.for_text(',')),
                SemanticCountry(layout_block=LayoutBlock.for_text('Country 1'))
            ]:
                semantic_affiliation.add_content(semantic_content)
            author = _author()
            author.append(get_tei_affiliation_for_semantic_affiliation_address_element(
                semantic_affiliation, context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
            ))
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(authors=[author])))

            person = _get_item(jats, 'front/article-meta/contrib-group/contrib')
            assert _get_text(person, './xref[@ref-type="aff"]') == '1'
            aff = _get_item(jats, 'front/article-meta/contrib-group/aff')
            assert ' '.join(get_text_content(aff).split()) == (
                '1 Department of Science , Institute 1 , Country 1'
            )
            assert [child.tag for child in aff] == [
                'label', 'institution', 'institution', 'country'
            ]

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

        def test_should_add_the_printed_section_number_as_label(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(
                    TEI_E.head(VALUE_1, n='2.1'),
                    TEI_E.p(VALUE_2)
                )))
            ))
            sec = _get_item(jats, 'body/sec')
            assert [child.tag for child in sec] == ['label', 'title', 'p']
            assert _get_text(sec, 'label') == '2.1'
            assert _get_text(sec, 'title') == VALUE_1

        def test_should_not_add_a_label_where_no_number_was_printed(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.div(TEI_E.head(VALUE_1))))
            ))
            assert jats.xpath('body/sec/label') == []

        def test_should_carry_the_section_number_the_parser_produced(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            semantic_document = SemanticDocument()
            section = semantic_document.body_section.add_new_section()
            section.add_content(SemanticHeading([
                SemanticLabel(layout_block=LayoutBlock.for_text('3.')),
                SemanticNote(layout_block=LayoutBlock.for_text(VALUE_1))
            ]))
            tei_document = get_tei_for_semantic_document(semantic_document)
            jats = etree.fromstring(tei_to_jats_xslt_fn(tei_document.root))
            assert _get_text(jats, 'body/sec/label') == '3.'
            assert _get_text(jats, 'body/sec/title') == VALUE_1

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
            assert _get_text(jats, 'body/sec/title/italic') == VALUE_1
            assert _get_text(jats, 'body/sec/p/italic') == VALUE_2

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
            assert not jats.xpath('body/sec/title/italic')
            assert _get_text(jats, 'body/sec/p') == VALUE_2
            assert not jats.xpath('body/sec/p/italic')

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
            assert _get_text(jats, 'body/sec/title/bold') == VALUE_1
            assert _get_text(jats, 'body/sec/p/bold') == VALUE_2

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
            assert not jats.xpath('body/sec/title/bold')
            assert _get_text(jats, 'body/sec/p') == VALUE_2
            assert not jats.xpath('body/sec/p/bold')

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
                        TEI_E.note('Figure', type='<figure_head>'),
                        TEI_E.head('1'),
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
            assert jats.xpath('body/sec/fig/caption/title') == []
            assert _get_text(jats, 'body/sec/fig/caption/p') == 'Figure 1. This is the figure'
            assert _get_item(jats, 'body/sec/fig/graphic') is not None
            assert _get_text(jats, 'body/sec/fig/graphic/@xlink:href ') == 'image1.png'
            assert _get_text(jats, 'body/sec/p/xref') == '(Figure 1)'
            assert _get_text(jats, 'body/sec/p/xref/@ref-type') == 'fig'
            assert _get_text(jats, 'body/sec/p/xref/@rid') == 'fig_0'
            assert not _xpath(jats, '//title/title')

        def test_should_label_a_figure_as_printed(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure(
                        {XML_ID: 'fig_0'},
                        TEI_E.note('Figura', type='<figure_head>'),
                        TEI_E.head('3'),
                        TEI_E.label('3'),
                        TEI_E.note('.', type='<figure_head>'),
                        TEI_E.figDesc(VALUE_1)
                    )
                ))
            ))
            fig = _get_item(jats, 'body/sec/fig')
            assert _get_text(fig, 'label') == 'Figura 3.'
            assert [child.tag for child in _get_item(fig, 'caption')] == ['p']

        def test_should_not_label_a_figure_without_a_printed_label(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(TEI_E.figure({XML_ID: 'fig_0'}, TEI_E.figDesc(VALUE_1))))
            ))
            assert jats.xpath('body/sec/fig/label') == []

        def test_should_leave_out_a_graphic_whose_file_was_not_kept(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure(
                        TEI_E.graphic(type='svg')
                    )
                ))
            ))
            assert _get_item(jats, 'body/sec/fig') is not None
            assert jats.xpath('body/sec/fig/graphic') == []

        def test_should_add_no_graphic_to_a_figure_without_one(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure({XML_ID: 'fig_0'})
                ))
            ))
            assert _get_text(jats, 'body/sec/fig/@id') == 'fig_0'
            assert jats.xpath('body/sec/fig/graphic') == []

        def test_should_create_empty_graphic_for_figures_without_graphic_if_enabled(
            self,
            tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(body=TEI_E.body(
                    TEI_E.figure({XML_ID: 'fig_0'})
                )),
                {'output_empty_figure_graphic': 'true'}
            ))
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
                        TEI_E.note('Table', type='<figure_head>'),
                        TEI_E.head('1'),
                        TEI_E.label('1'),
                        TEI_E.figDesc('Table 1. This is a table'),
                        TEI_E.table('Table content')
                    )
                ))
            ))
            assert _get_text(jats, 'body/sec/table-wrap/@id') == 'tab_0'
            assert _get_text(jats, 'body/sec/table-wrap/label') == 'Table 1'
            assert jats.xpath('body/sec/table-wrap/caption/title') == []
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

        def test_should_extract_acknowledgement_head_and_p_without_inner_div_as_ack(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'acknowledgement'},
                        TEI_E.head(VALUE_1),
                        TEI_E.p(VALUE_2)
                    )
                )),
                {
                    'acknowledgement_target': 'ack'
                }
            ))
            assert _get_text(jats, 'back/ack/sec/title') == VALUE_1
            assert _get_text(jats, 'back/ack/sec/p') == VALUE_2

        def test_should_extract_acknowledgement_head_and_p_without_inner_div_as_body(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'acknowledgement'},
                        TEI_E.head(VALUE_1),
                        TEI_E.p(VALUE_2)
                    )
                )),
                {
                    'acknowledgement_target': 'body'
                }
            ))
            assert _get_text(jats, 'body/sec/title') == VALUE_1
            assert _get_text(jats, 'body/sec/p') == VALUE_2

        def test_should_extract_the_acknowledgement_the_parser_produced(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            semantic_document = SemanticDocument()
            section = semantic_document.back_section.add_new_section(
                SemanticSectionTypes.ACKNOWLEDGEMENT
            )
            section.add_heading_block(LayoutBlock.for_text(VALUE_1))
            section.add_new_paragraph().add_block_content(LayoutBlock.for_text(VALUE_2))
            tei_document = get_tei_for_semantic_document(semantic_document)
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                tei_document.root,
                {
                    'acknowledgement_target': 'ack'
                }
            ))
            assert _get_text(jats, 'back/ack/sec/title') == VALUE_1
            assert _get_text(jats, 'back/ack/sec/p') == VALUE_2

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
                            TEI_E.note('Figure', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('back/sec/fig/caption/title') == []
            assert _get_text(jats, 'back/sec/fig/caption/p') == 'Figure 1. This is the figure'

        def test_should_extract_annex_figures_as_body_section(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            TEI_E.note('Figure', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('body/sec/fig/caption/title') == []
            assert _get_text(jats, 'body/sec/fig/caption/p') == 'Figure 1. This is the figure'

        def test_should_put_annex_figures_before_annex_sections_in_the_app(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.div(TEI_E.head(VALUE_1), TEI_E.p(VALUE_2)),
                        TEI_E.figure(TEI_E.head('Figure 1'))
                    )
                )),
                {'annex_target': 'app'}
            ))
            app = _get_item(jats, 'back/app-group/app')
            assert [child.tag for child in app] == ['title', 'fig', 'sec']

        def test_should_extract_annex_figures_as_app_group(
            self, tei_to_jats_xslt_fn: T_TeiToJatsXsltFn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(back=TEI_E.back(
                    TEI_E.div(
                        {'type': 'annex'},
                        TEI_E.figure(
                            TEI_E.note('Figure', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('back/app-group/app/fig/caption/title') == []
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
                            TEI_E.note('Table', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('back/sec/table-wrap/caption/title') == []
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
                            TEI_E.note('Table', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('body/sec/table-wrap/caption/title') == []
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
                            TEI_E.note('Table', type='<figure_head>'),
                            TEI_E.head('1'),
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
            assert jats.xpath('back/app-group/app/table-wrap/caption/title') == []
            assert _get_text(jats, 'back/app-group/app/table-wrap/caption/p') == (
                'Table 1. This is the table'
            )

    class TestMixedCitation:
        def test_should_tag_each_part_where_it_is_printed(self, tei_to_jats_xslt_fn):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                'Hallgren P, Okafor N. Creep in beams. J Timber Eng. 2019;14(2):101-9.',
                TEI_E.analytic(
                    _person('Hallgren', 'P'),
                    _person('Okafor', 'N'),
                    TEI_E.title('Creep in beams', level='a', type='main')
                ),
                TEI_E.monogr(
                    TEI_E.title('J Timber Eng', level='j'),
                    TEI_E.imprint(
                        TEI_E.biblScope('14', unit='volume'),
                        TEI_E.biblScope('2', unit='issue'),
                        TEI_E.biblScope({'unit': 'page', 'from': '101', 'to': '9'}),
                        TEI_E.date('2019', type='published', when='2019')
                    )
                )
            )) == (
                '<person-group person-group-type="author">'
                '<string-name><surname>Hallgren</surname> <given-names>P</given-names>'
                '</string-name>, '
                '<string-name><surname>Okafor</surname> <given-names>N</given-names>'
                '</string-name></person-group>. '
                '<article-title>Creep in beams</article-title>. '
                '<source>J Timber Eng</source>. <year>2019</year>;<volume>14</volume>'
                '(<issue>2</issue>):<fpage>101</fpage>-<lpage>9</lpage>.</mixed-citation>'
            )

        def test_should_only_tag_a_part_printed_as_a_whole_word(self, tei_to_jats_xslt_fn):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                'A title. 2019; 2.',
                TEI_E.monogr(TEI_E.imprint(TEI_E.biblScope('2', unit='issue')))
            )) == 'A title. 2019; <issue>2</issue>.</mixed-citation>'

        def test_should_keep_text_no_part_accounts_for(self, tei_to_jats_xslt_fn):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                '. A title, in press.',
                TEI_E.analytic(TEI_E.title('A title', level='a', type='main'))
            )) == '. <article-title>A title</article-title>, in press.</mixed-citation>'

        def test_should_tag_initials_printed_together_as_one_given_name(
            self, tei_to_jats_xslt_fn
        ):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                'Broadhead MJ. A map. J Neurosci.',
                TEI_E.analytic(_person('Broadhead', 'M', 'J')),
                TEI_E.monogr(TEI_E.title('J Neurosci', level='j'))
            )) == (
                '<person-group person-group-type="author"><string-name>'
                '<surname>Broadhead</surname> <given-names>MJ</given-names>'
                '</string-name></person-group>. A map. <source>J Neurosci</source>.'
                '</mixed-citation>'
            )

        def test_should_leave_a_name_untagged_where_its_parts_are_printed_apart(
            self, tei_to_jats_xslt_fn
        ):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                'Smith and colleagues, edited by K.',
                TEI_E.analytic(_person('Smith', 'K'))
            )) == 'Smith and colleagues, edited by K.</mixed-citation>'

        def test_should_write_a_doi_and_url_without_the_spaces_a_line_break_left(
            self, tei_to_jats_xslt_fn
        ):
            assert _mixed_citation_of(tei_to_jats_xslt_fn, _printed_reference(
                'A title. doi: 10. 1000/xyz. Available from: https://example.org/ data',
                TEI_E.analytic(TEI_E.idno('10.1000/xyz', type='DOI')),
                TEI_E.ref('https://example.org/data', type='url')
            )) == (
                'A title. doi: <pub-id pub-id-type="doi">10.1000/xyz</pub-id>. Available from: '
                '<ext-link ext-link-type="uri"'
                ' xlink:href="https://example.org/data">https://example.org/data</ext-link>'
                '</mixed-citation>'
            )

        def test_should_keep_the_element_citation_without_the_printed_reference(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**REFERENCE_1)])
            ))
            assert [child.tag for child in _get_item(jats, 'back/ref-list/ref')] == [
                'element-citation'
            ]

        def test_should_write_an_empty_mixed_citation_where_nothing_is_known(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[
                TEI_E.biblStruct({'{http://www.w3.org/XML/1998/namespace}id': 'b0'})
            ])))
            ref = _get_item(jats, 'back/ref-list/ref')
            assert [child.tag for child in ref] == ['mixed-citation']
            assert len(ref[0]) == 0 and not ref[0].text

        def test_should_tag_the_reference_the_parser_produced(self, tei_to_jats_xslt_fn):
            semantic_reference = SemanticReference(content_id='b0')
            semantic_reference.add_content(SemanticRawReferenceText(
                layout_block=LayoutBlock.for_text('Smith J. A title. A Journal 12.')
            ))
            semantic_reference.add_content(SemanticAuthor([
                SemanticSurname(layout_block=LayoutBlock.for_text('Smith')),
                SemanticGivenName(layout_block=LayoutBlock.for_text('J'))
            ]))
            semantic_reference.add_content(
                SemanticTitle(layout_block=LayoutBlock.for_text('A title'))
            )
            semantic_reference.add_content(
                SemanticJournal(layout_block=LayoutBlock.for_text('A Journal'))
            )
            semantic_reference.add_content(
                SemanticVolume(layout_block=LayoutBlock.for_text('12'))
            )
            assert _mixed_citation_of(tei_to_jats_xslt_fn, get_tei_reference_element(
                semantic_reference, context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
            )) == (
                '<person-group person-group-type="author"><string-name>'
                '<surname>Smith</surname> <given-names>J</given-names></string-name>'
                '</person-group>. <article-title>A title</article-title>. '
                '<source>A Journal</source> <volume>12</volume>.</mixed-citation>'
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

        def test_should_carry_the_doi_the_parser_produced(self, tei_to_jats_xslt_fn):
            semantic_reference = SemanticReference(content_id='b0')
            semantic_reference.add_content(SemanticExternalIdentifier(
                layout_block=LayoutBlock.for_text('10.1234/doi1'),
                value='10.1234/doi1',
                external_identifier_type=SemanticExternalIdentifierTypes.DOI
            ))
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[
                get_tei_reference_element(
                    semantic_reference, context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
                )
            ])))

            assert _get_text(
                jats, 'back/ref-list/ref/element-citation/pub-id[@pub-id-type="doi"]'
            ) == '10.1234/doi1'

        def test_should_add_the_printed_reference_number_as_label(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(REFERENCE_1, label='12'))])
            ))
            ref = _get_item(jats, 'back/ref-list/ref')
            assert [child.tag for child in ref] == ['label', 'element-citation']
            assert _get_text(ref, 'label') == '12'

        def test_should_not_add_a_label_where_no_reference_number_was_printed(
            self, tei_to_jats_xslt_fn
        ):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**REFERENCE_1)])
            ))
            assert jats.xpath('back/ref-list/ref/label') == []

        def test_should_carry_the_reference_number_the_parser_produced(
            self, tei_to_jats_xslt_fn
        ):
            semantic_reference = SemanticReference(content_id='b0')
            semantic_reference.add_content(
                SemanticLabel(layout_block=LayoutBlock.for_text('[4]'))
            )
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[
                get_tei_reference_element(
                    semantic_reference, context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
                )
            ])))
            assert _get_text(jats, 'back/ref-list/ref/label') == '[4]'

        def test_should_write_the_authors_first(self, tei_to_jats_xslt_fn):
            jats = etree.fromstring(tei_to_jats_xslt_fn(
                _tei(references=[_reference(**extend_dict(
                    REFERENCE_1, article_authors=[AUTHOR_1]
                ))])
            ))
            element_citation = _get_item(jats, 'back/ref-list/ref/element-citation')
            assert element_citation[0].tag == 'person-group'

        def test_should_link_a_url_printed_in_angle_brackets(self, tei_to_jats_xslt_fn):
            bibl_struct = _reference(id='b0')
            bibl_struct.append(TEI_E.ref('<https://example.org/ data>', type='url'))
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[bibl_struct])))
            ext_link = _get_item(jats, 'back/ref-list/ref/element-citation/ext-link')
            assert ext_link.attrib['{http://www.w3.org/1999/xlink}href'] == (
                'https://example.org/data'
            )
            assert ext_link.text == '<https://example.org/ data>'

        def test_should_link_the_url_of_a_reference(self, tei_to_jats_xslt_fn):
            bibl_struct = _reference(id='b0')
            bibl_struct.append(TEI_E.ref(' https://example.org/data ', type='url'))
            jats = etree.fromstring(tei_to_jats_xslt_fn(_tei(references=[bibl_struct])))
            ext_link = _get_item(jats, 'back/ref-list/ref/element-citation/ext-link')
            assert ext_link.attrib == {
                'ext-link-type': 'uri',
                '{http://www.w3.org/1999/xlink}href': 'https://example.org/data'
            }
            assert ext_link.text == 'https://example.org/data'

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
