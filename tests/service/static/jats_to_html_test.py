import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

from lxml import etree
import pytest


HARNESS_FILE = Path(__file__).parent / 'get_article_view_model.mjs'

XLINK_NS = 'http://www.w3.org/1999/xlink'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the reading view that the page runs in a browser'
)


def get_attribute_name(name: str) -> str:
    if not name.startswith('{'):
        return name
    namespace, local_name = name[1:].split('}', maxsplit=1)
    return f'xlink:{local_name}' if namespace == XLINK_NS else local_name


def get_xml_node(element: etree.ElementBase) -> Dict[str, Any]:
    children: List[Union[str, dict]] = []
    if element.text:
        children.append(element.text)
    for child in element:
        if isinstance(child.tag, str):
            children.append(get_xml_node(child))
        if child.tail:
            children.append(child.tail)
    return {
        'name': etree.QName(element).localname,
        'attributes': {
            get_attribute_name(name): value
            for name, value in element.attrib.items()
        },
        'children': children
    }


def get_article_view_model(jats_xml: str) -> dict:
    root = etree.fromstring(jats_xml.encode('utf-8'))
    completed_process = subprocess.run(
        ['node', str(HARNESS_FILE)],
        input=json.dumps(get_xml_node(root)),
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(completed_process.stdout)


def get_article_xml(front: str = '', body: str = '', back: str = '') -> str:
    return (
        '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<front><article-meta>{front}</article-meta></front>'
        f'<body>{body}</body>'
        f'<back>{back}</back>'
        '</article>'
    )


def get_paragraph_texts(paragraphs: List[dict]) -> List[str]:
    return [
        ''.join(run['text'] for run in paragraph['runs'])
        for paragraph in paragraphs
    ]


class TestTitle:
    def test_should_read_the_article_title(self):
        view_model = get_article_view_model(get_article_xml(
            front='<title-group><article-title>The title</article-title></title-group>'
        ))
        assert view_model['title'] == 'The title'

    def test_should_be_empty_without_a_title(self):
        assert get_article_view_model(get_article_xml())['title'] == ''


class TestAuthors:
    def test_should_read_the_name_in_reading_order(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<contrib-group><contrib contrib-type="person"><name>'
                '<surname>Smith</surname><given-names>Jane A</given-names>'
                '</name></contrib></contrib-group>'
            )
        ))
        assert [author['name'] for author in view_model['authors']] == ['Jane A Smith']

    def test_should_read_the_email_and_the_affiliations_referenced(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<contrib-group><contrib>'
                '<name><surname>Smith</surname></name>'
                '<email>jane@example.org</email>'
                '<xref ref-type="aff" rid="aff1"/>'
                '<xref ref-type="other" rid="other1"/>'
                '</contrib></contrib-group>'
            )
        ))
        author = view_model['authors'][0]
        assert author['email'] == 'jane@example.org'
        assert author['affiliationIds'] == ['aff1']


class TestAffiliations:
    def test_should_read_the_parts_from_the_smallest_outwards(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<aff id="aff1">'
                '<institution content-type="orgname">A University</institution>'
                '<institution content-type="orgdiv1">A Department</institution>'
                '<institution content-type="orgdiv2">A Laboratory</institution>'
                '<city>A City</city><country>A Country</country>'
                '</aff>'
            )
        ))
        assert view_model['affiliations'] == [{
            'id': 'aff1',
            'text': 'A Laboratory, A Department, A University, A City, A Country'
        }]

    def test_should_leave_out_the_parts_the_document_does_not_carry(self):
        view_model = get_article_view_model(get_article_xml(
            front='<aff id="aff1"><country>A Country</country></aff>'
        ))
        assert view_model['affiliations'][0]['text'] == 'A Country'


class TestAbstract:
    def test_should_read_every_paragraph_of_every_abstract(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<abstract><p>First.</p><p>Second.</p></abstract>'
                '<abstract><title>Summary</title><p>Other.</p></abstract>'
            )
        ))
        assert [
            get_paragraph_texts(abstract['paragraphs'])
            for abstract in view_model['abstracts']
        ] == [['First.', 'Second.'], ['Other.']]
        assert view_model['abstracts'][1]['title'] == 'Summary'


class TestSections:
    def test_should_keep_the_order_the_document_has(self):
        view_model = get_article_view_model(get_article_xml(
            body=(
                '<sec><title>Methods</title>'
                '<p>Before.</p>'
                '<fig id="fig_0"><label>Figure 1</label></fig>'
                '<p>After.</p>'
                '</sec>'
            )
        ))
        section = view_model['sections'][0]
        assert section['title'] == 'Methods'
        assert [block['type'] for block in section['blocks']] == [
            'paragraph', 'figure', 'paragraph'
        ]

    def test_should_keep_a_nested_section_inside_its_parent(self):
        view_model = get_article_view_model(get_article_xml(
            body='<sec><title>Outer</title><sec><title>Inner</title></sec></sec>'
        ))
        outer = view_model['sections'][0]
        assert [block['title'] for block in outer['blocks']] == ['Inner']

    def test_should_read_the_marked_up_runs_of_a_paragraph(self):
        view_model = get_article_view_model(get_article_xml(
            body='<sec><p>Plain <b>bold</b> and <i>italic</i>.</p></sec>'
        ))
        runs = view_model['sections'][0]['blocks'][0]['runs']
        assert [(run['text'], run.get('bold'), run.get('italic')) for run in runs] == [
            ('Plain ', None, None),
            ('bold', True, None),
            (' and ', None, None),
            ('italic', None, True),
            ('.', None, None)
        ]


class TestFiguresAndTables:
    def test_should_read_a_figure_with_its_caption_and_graphic(self):
        view_model = get_article_view_model(get_article_xml(
            body=(
                '<sec><fig id="fig_0"><label>Figure 1</label>'
                '<caption><title>Figure 1</title><p>What it shows.</p></caption>'
                '<graphic xlink:href="image1.png"/></fig></sec>'
            )
        ))
        figure = view_model['sections'][0]['blocks'][0]
        assert figure['label'] == 'Figure 1'
        assert get_paragraph_texts(figure['paragraphs']) == ['What it shows.']
        assert figure['graphicHref'] == 'image1.png'

    def test_should_read_a_table_row_by_row(self):
        view_model = get_article_view_model(get_article_xml(
            body=(
                '<sec><table-wrap id="tab_0"><label>Table 1</label>'
                '<table><tbody>'
                '<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>'
                '</tbody></table></table-wrap></sec>'
            )
        ))
        table = view_model['sections'][0]['blocks'][0]
        assert [get_paragraph_texts(row) for row in table['rows']] == [
            ['a', 'b'], ['c', 'd']
        ]


class TestBackMatter:
    def test_should_read_the_acknowledgements(self):
        view_model = get_article_view_model(get_article_xml(
            back='<ack><sec><title>Thanks</title><p>To everyone.</p></sec></ack>'
        ))
        assert [block['title'] for block in view_model['acknowledgements']] == ['Thanks']

    def test_should_read_an_appendix(self):
        view_model = get_article_view_model(get_article_xml(
            back='<app-group><app><title>Appendix 1</title><p>More.</p></app></app-group>'
        ))
        assert [appendix['title'] for appendix in view_model['appendices']] == [
            'Appendix 1'
        ]


class TestReferences:
    def test_should_read_what_a_citation_states(self):
        view_model = get_article_view_model(get_article_xml(
            back=(
                '<ref-list><ref id="b0"><element-citation publication-type="journal">'
                '<article-title>The cited title</article-title>'
                '<source>A Journal</source>'
                '<year>2020</year><volume>12</volume><issue>3</issue>'
                '<fpage>100</fpage><lpage>110</lpage>'
                '<pub-id pub-id-type="doi">10.1000/xyz</pub-id>'
                '<person-group person-group-type="author">'
                '<name><surname>Smith</surname><given-names>J</given-names></name>'
                '<name><surname>Jones</surname><given-names>A</given-names></name>'
                '</person-group>'
                '</element-citation></ref></ref-list>'
            )
        ))
        assert view_model['references'] == [{
            'id': 'b0',
            'authors': ['J Smith', 'A Jones'],
            'articleTitle': 'The cited title',
            'source': 'A Journal',
            'year': '2020',
            'volume': '12',
            'issue': '3',
            'firstPage': '100',
            'lastPage': '110',
            'doi': '10.1000/xyz'
        }]

    def test_should_read_a_citation_that_states_almost_nothing(self):
        view_model = get_article_view_model(get_article_xml(
            back='<ref-list><ref id="b0"><element-citation/></ref></ref-list>'
        ))
        reference = view_model['references'][0]
        assert reference['id'] == 'b0'
        assert reference['authors'] == []
        assert reference['articleTitle'] == ''
