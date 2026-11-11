import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from lxml import etree
import pytest

from tests.service.static.jats_to_html_test import (
    TAG_ID_PATTERN,
    TAG_PATTERN,
    get_text_without_labels,
    get_xml_node,
    iter_paths,
    without_paths
)


VIEW_MODEL_HARNESS_FILE = Path(__file__).parent / 'get_tei_view_model.mjs'
RENDER_HARNESS_FILE = Path(__file__).parent / 'render_tei.mjs'

TEI_NS = 'http://www.tei-c.org/ns/1.0'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the reading view that the page runs in a browser'
)


def run_harness(harness_file: Path, payload: Any) -> str:
    completed_process = subprocess.run(
        ['node', str(harness_file)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True
    )
    return completed_process.stdout


def get_parsed_xml_node(tei_xml: str) -> Dict[str, Any]:
    return get_xml_node(etree.fromstring(tei_xml.encode('utf-8')))


def get_tei_view_model(tei_xml: str) -> dict:
    return json.loads(
        run_harness(VIEW_MODEL_HARNESS_FILE, get_parsed_xml_node(tei_xml))
    )


def get_rendered_tei(
    tei_xml: str, assets: Optional[Dict[str, str]] = None
) -> str:
    return run_harness(RENDER_HARNESS_FILE, {
        'node': get_parsed_xml_node(tei_xml),
        'assets': assets or {}
    })


def get_tei_xml(header: str = '', body: str = '', back: str = '') -> str:
    return (
        f'<TEI xmlns="{TEI_NS}">'
        f'<teiHeader>{header}</teiHeader>'
        f'<text><body>{body}</body><back>{back}</back></text>'
        '</TEI>'
    )


def get_paragraph_texts(paragraphs: List[dict]) -> List[str]:
    return [
        ''.join(run['text'] for run in paragraph['runs'])
        for paragraph in paragraphs
    ]


TITLE_STMT = (
    '<fileDesc><titleStmt><title level="a" type="main">The title</title>'
    '</titleStmt></fileDesc>'
)


class TestTitle:
    def test_should_read_the_article_title_and_the_element_it_came_from(self):
        view_model = get_tei_view_model(get_tei_xml(header=TITLE_STMT))
        assert without_paths(view_model['titles']) == [{
            'label': 'title', 'text': 'The title'
        }]

    def test_should_carry_no_title_where_the_document_states_none(self):
        assert get_tei_view_model(get_tei_xml())['titles'] == []

    def test_should_read_every_title_with_the_language_it_declares(self):
        view_model = get_tei_view_model(get_tei_xml(header=(
            '<fileDesc><titleStmt>'
            '<title level="a" type="main" xml:lang="en">The title</title>'
            '<title level="a" type="main" xml:lang="de">Der Titel</title>'
            '</titleStmt></fileDesc>'
        )))
        assert without_paths(view_model['titles']) == [
            {'label': 'title', 'text': 'The title', 'language': 'en'},
            {'label': 'title', 'text': 'Der Titel', 'language': 'de'}
        ]


AUTHORS_HEADER = (
    '<fileDesc><sourceDesc><biblStruct><analytic>'
    '<author><persName><forename type="first">Jane</forename>'
    '<forename type="middle">A</forename><surname>Smith</surname></persName>'
    '<affiliation key="aff1">'
    '<orgName type="institution">A University</orgName>'
    '<address><settlement>A City</settlement><country>A Country</country></address>'
    '</affiliation></author>'
    '<author><persName><forename type="first">Ravi</forename>'
    '<surname>Jones</surname></persName>'
    '<affiliation key="aff2">'
    '<orgName type="institution">An Institute</orgName></affiliation>'
    '</author>'
    '</analytic></biblStruct></sourceDesc></fileDesc>'
)


class TestAuthors:
    def test_should_read_each_forename_before_the_surname(self):
        view_model = get_tei_view_model(get_tei_xml(header=AUTHORS_HEADER))
        assert without_paths(view_model['authors'][0]['nameParts']) == [
            {'label': 'forename', 'text': 'Jane'},
            {'label': 'forename', 'text': 'A'},
            {'label': 'surname', 'text': 'Smith'}
        ]

    def test_should_show_the_group_the_authors_sit_in(self):
        view_model = get_tei_view_model(get_tei_xml(header=AUTHORS_HEADER))
        assert view_model['authorGroup']['label'] == 'analytic'

    def test_should_point_an_author_at_the_affiliation_it_carries(self):
        view_model = get_tei_view_model(get_tei_xml(header=AUTHORS_HEADER))
        assert [
            [ref['rid'] for ref in author['affiliationRefs']]
            for author in view_model['authors']
        ] == [['aff1'], ['aff2']]

    def test_should_name_the_element_the_reference_is(self):
        view_model = get_tei_view_model(get_tei_xml(header=AUTHORS_HEADER))
        assert view_model['authors'][0]['affiliationRefs'][0]['label'] == (
            'affiliation'
        )


class TestAffiliations:
    def test_should_read_the_parts_from_the_smallest_outwards(self):
        view_model = get_tei_view_model(get_tei_xml(header=(
            '<fileDesc><sourceDesc><biblStruct><analytic><author>'
            '<affiliation key="aff1">'
            '<orgName type="institution">A University</orgName>'
            '<orgName type="department">A Department</orgName>'
            '<orgName type="laboratory">A Laboratory</orgName>'
            '<address><settlement>A City</settlement>'
            '<postCode>00000</postCode><country>A Country</country></address>'
            '</affiliation></author></analytic></biblStruct></sourceDesc></fileDesc>'
        )))
        assert [
            (part['label'], part['text'])
            for part in view_model['affiliations'][0]['parts']
        ] == [
            ('laboratory', 'A Laboratory'),
            ('department', 'A Department'),
            ('institution', 'A University'),
            ('settlement', 'A City'),
            ('postCode', '00000'),
            ('country', 'A Country')
        ]

    def test_should_keep_an_affiliation_each_author_carries(self):
        view_model = get_tei_view_model(get_tei_xml(header=AUTHORS_HEADER))
        assert [
            affiliation['id'] for affiliation in view_model['affiliations']
        ] == ['aff1', 'aff2']


class TestAbstract:
    def test_should_read_every_paragraph_of_the_abstract(self):
        view_model = get_tei_view_model(get_tei_xml(
            header='<profileDesc><abstract><p>First.</p><p>Second.</p></abstract></profileDesc>'
        ))
        assert get_paragraph_texts(
            view_model['abstracts'][0]['paragraphs']
        ) == ['First.', 'Second.']


class TestBody:
    def test_should_read_a_division_as_a_section_with_its_head(self):
        view_model = get_tei_view_model(get_tei_xml(
            body='<div><head>Methods</head><p>A paragraph.</p></div>'
        ))
        section = view_model['sections'][0]
        assert section['type'] == 'section'
        assert without_paths(section['title']) == {
            'label': 'head', 'text': 'Methods'
        }
        assert get_paragraph_texts(section['blocks']) == ['A paragraph.']

    def test_should_keep_a_nested_division_inside_its_parent(self):
        view_model = get_tei_view_model(get_tei_xml(
            body='<div><head>Outer</head><div><head>Inner</head></div></div>'
        ))
        assert [
            block['title']['text']
            for block in view_model['sections'][0]['blocks']
        ] == ['Inner']

    def test_should_read_the_marked_up_runs_of_a_paragraph(self):
        view_model = get_tei_view_model(get_tei_xml(
            body=(
                '<div><p>Plain <hi rend="bold">bold</hi> and '
                '<hi rend="italic">italic</hi>.</p></div>'
            )
        ))
        runs = view_model['sections'][0]['blocks'][0]['runs']
        assert [
            (run['text'], run.get('bold'), run.get('italic')) for run in runs
        ] == [
            ('Plain ', None, None),
            ('bold', True, None),
            (' and ', None, None),
            ('italic', None, True),
            ('.', None, None)
        ]

    def test_should_read_a_reference_in_a_paragraph_as_what_it_points_at(self):
        view_model = get_tei_view_model(get_tei_xml(
            body='<div><p>See <ref type="bibr" target="#b12">12</ref>.</p></div>'
        ))
        runs = view_model['sections'][0]['blocks'][0]['runs']
        assert [run.get('reference') for run in runs] == [None, 'b12', None]


FIGURE_BODY = (
    '<figure xml:id="fig_0"><head>A figure</head><label>1</label>'
    '<figDesc>What it shows.</figDesc>'
    '<graphic url="image1.png" type="image"/>'
    '<graphic url="image2.png" type="image"/></figure>'
    '<figure type="table" xml:id="tab_0"><head>A table</head><label>1</label>'
    '<figDesc>What it lists.</figDesc></figure>'
)


class TestFiguresAndTables:
    def test_should_read_a_figure_with_its_caption_and_every_graphic(self):
        view_model = get_tei_view_model(get_tei_xml(body=FIGURE_BODY))
        figure = view_model['sections'][0]
        assert figure['type'] == 'figure'
        assert figure['id'] == 'fig_0'
        assert without_paths(figure['heading']) == {'label': 'label', 'text': '1'}
        assert get_paragraph_texts(figure['paragraphs']) == ['What it shows.']
        assert [graphic['href'] for graphic in figure['graphics']] == [
            'image1.png', 'image2.png'
        ]

    def test_should_read_a_figure_that_says_it_is_a_table_as_one(self):
        view_model = get_tei_view_model(get_tei_xml(body=FIGURE_BODY))
        table = view_model['sections'][1]
        assert table['type'] == 'table'
        assert table['id'] == 'tab_0'
        assert table['rows'] == []

    def test_should_show_the_image_where_it_travels_beside_the_xml(self):
        html = get_rendered_tei(
            get_tei_xml(body=FIGURE_BODY), assets={'image1.png': 'blob:one'}
        )
        assert 'src="blob:one"' in html
        assert 'image2.png (in the zip output)' in html


class TestBackMatter:
    def test_should_read_an_annex_as_an_appendix(self):
        view_model = get_tei_view_model(get_tei_xml(
            back='<div type="annex"><div><head>Methods</head><p>More.</p></div></div>'
        ))
        assert [
            block['title']['text'] for block in view_model['appendices']
        ] == ['Methods']

    def test_should_read_an_acknowledgement_as_one(self):
        view_model = get_tei_view_model(get_tei_xml(
            back=(
                '<div type="acknowledgement"><div><head>Thanks</head>'
                '<p>To everyone.</p></div></div>'
            )
        ))
        assert [
            block['title']['text'] for block in view_model['acknowledgements']
        ] == ['Thanks']


REFERENCE_BACK = (
    '<div type="references"><listBibl>'
    '<biblStruct xml:id="b0">'
    '<analytic>'
    '<author><persName><forename type="first">J</forename>'
    '<surname>Smith</surname></persName></author>'
    '<author><persName><forename type="first">A</forename>'
    '<surname>Jones</surname></persName></author>'
    '<title level="a" type="main">The cited title</title>'
    '</analytic>'
    '<monogr><title level="j">A Journal</title>'
    '<imprint><date type="published" when="2020">2020</date>'
    '<biblScope unit="volume">12</biblScope>'
    '<biblScope unit="issue">3</biblScope>'
    '<biblScope unit="page" from="100" to="110"/></imprint></monogr>'
    '<idno type="DOI">10.1000/xyz</idno>'
    '</biblStruct></listBibl></div>'
)


class TestReferences:
    def test_should_read_what_a_citation_states(self):
        view_model = get_tei_view_model(get_tei_xml(back=REFERENCE_BACK))
        reference = without_paths(view_model['references'][0])
        assert reference['id'] == 'b0'
        assert reference['articleTitle'] == {
            'label': 'title', 'text': 'The cited title'
        }
        assert reference['source'] == {'label': 'title', 'text': 'A Journal'}
        assert reference['year'] == {'label': 'date', 'text': '2020'}
        assert reference['volume'] == {'label': 'biblScope', 'text': '12'}
        assert reference['issue'] == {'label': 'biblScope', 'text': '3'}
        assert reference['firstPage'] == {'label': 'page', 'text': '100'}
        assert reference['lastPage'] == {'label': 'page', 'text': '110'}
        assert reference['doi'] == {'label': 'idno', 'text': '10.1000/xyz'}

    def test_should_read_the_authors_of_a_citation(self):
        view_model = get_tei_view_model(get_tei_xml(back=REFERENCE_BACK))
        assert [
            [part['text'] for part in name]
            for name in view_model['references'][0]['authors']
        ] == [['J', 'Smith'], ['A', 'Jones']]

    def test_should_read_a_page_range_the_document_gives_as_one_value(self):
        view_model = get_tei_view_model(get_tei_xml(back=(
            '<div type="references"><listBibl><biblStruct xml:id="b0">'
            '<monogr><imprint><biblScope unit="page">42</biblScope>'
            '</imprint></monogr></biblStruct></listBibl></div>'
        )))
        reference = view_model['references'][0]
        assert reference['firstPage']['text'] == '42'
        assert reference['lastPage'] is None

    def test_should_read_a_citation_that_states_almost_nothing(self):
        view_model = get_tei_view_model(get_tei_xml(back=(
            '<div type="references"><listBibl>'
            '<biblStruct xml:id="b0"/></listBibl></div>'
        )))
        reference = view_model['references'][0]
        assert reference['id'] == 'b0'
        assert reference['authors'] == []
        assert reference['articleTitle'] is None


FULL_TEI_XML = get_tei_xml(
    header=TITLE_STMT + AUTHORS_HEADER + (
        '<profileDesc><abstract><p>An abstract.</p></abstract></profileDesc>'
    ),
    body='<div><head>Methods</head><p>A paragraph.</p></div>' + FIGURE_BODY,
    back=REFERENCE_BACK
)


class TestElementPaths:
    def test_should_say_where_in_the_document_a_value_was_found(self):
        view_model = get_tei_view_model(FULL_TEI_XML)
        assert view_model['titles'][0]['path'] == (
            '/TEI/teiHeader/fileDesc[1]/titleStmt/title'
        )

    def test_should_name_a_path_that_finds_the_one_element_it_came_from(self):
        view_model = get_tei_view_model(FULL_TEI_XML)
        paths = sorted(set(iter_paths(view_model)))
        assert len(paths) > 30
        tree = etree.ElementTree(etree.fromstring(FULL_TEI_XML.encode('utf-8')))
        namespaces = {'t': TEI_NS}
        found = {
            path: len(tree.xpath(
                re.sub(r'/(\w)', r'/t:\1', path), namespaces=namespaces
            ))
            for path in paths
        }
        assert {path: n for path, n in found.items() if n != 1} == {}


class TestRenderedTei:
    def test_should_mark_each_value_with_the_element_it_came_from(self):
        html = get_rendered_tei(FULL_TEI_XML)
        for label in [
            'title', 'analytic', 'author', 'forename', 'surname', 'affiliation',
            'institution', 'abstract', 'p', 'div', 'head', 'figure', 'label',
            'biblStruct'
        ]:
            assert f'data-jats="{label}"' in html

    def test_should_show_every_title_with_the_language_it_declares(self):
        html = get_rendered_tei(get_tei_xml(header=(
            '<fileDesc><titleStmt>'
            '<title level="a" type="main" xml:lang="en">The title</title>'
            '<title level="a" type="main" xml:lang="de">Der Titel</title>'
            '</titleStmt></fileDesc>'
        )))
        assert 'lang="en"' in html
        assert 'lang="de"' in html
        text_only = get_text_without_labels(html)
        assert 'The title' in text_only
        assert 'Der Titel' in text_only

    def test_should_read_as_the_article_without_the_names_shown(self):
        text_only = get_text_without_labels(get_rendered_tei(FULL_TEI_XML))
        assert 'The title' in text_only
        assert 'Jane A Smith' in text_only
        assert 'A University, A City, A Country' in text_only
        assert 'A paragraph.' in text_only
        assert 'J Smith, A Jones. The cited title. A Journal, 12(3), 100-110, 2020' in text_only

    def test_should_say_which_one_an_element_with_an_id_is(self):
        html = get_rendered_tei(FULL_TEI_XML)
        for element_id in ['fig_0', 'tab_0', 'b0', 'aff1']:
            assert f'<span class="jats-tag-id">#{element_id}</span>' in html

    def test_should_point_an_author_at_the_affiliation_it_carries(self):
        html = get_rendered_tei(FULL_TEI_XML)
        assert 'href="#aff1"' in html
        assert 'title="A University, A City, A Country"' in html
        assert TAG_PATTERN and TAG_ID_PATTERN
