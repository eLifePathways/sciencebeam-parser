import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lxml import etree
import pytest


VIEW_MODEL_HARNESS_FILE = Path(__file__).parent / 'get_article_view_model.mjs'
RENDER_HARNESS_FILE = Path(__file__).parent / 'render_article.mjs'

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


def run_harness(harness_file: Path, payload: Any) -> str:
    completed_process = subprocess.run(
        ['node', str(harness_file)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True
    )
    return completed_process.stdout


def get_parsed_xml_node(jats_xml: str) -> Dict[str, Any]:
    return get_xml_node(etree.fromstring(jats_xml.encode('utf-8')))


def get_article_view_model(jats_xml: str) -> dict:
    return json.loads(
        run_harness(VIEW_MODEL_HARNESS_FILE, get_parsed_xml_node(jats_xml))
    )


def get_rendered_article(
    jats_xml: str, assets: Optional[Dict[str, str]] = None
) -> str:
    return run_harness(RENDER_HARNESS_FILE, {
        'node': get_parsed_xml_node(jats_xml),
        'assets': assets or {}
    })


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
    def test_should_read_the_article_title_and_the_element_it_came_from(self):
        view_model = get_article_view_model(get_article_xml(
            front='<title-group><article-title>The title</article-title></title-group>'
        ))
        assert view_model['title'] == {'label': 'article-title', 'text': 'The title'}

    def test_should_carry_no_title_where_the_document_states_none(self):
        assert get_article_view_model(get_article_xml())['title'] is None


class TestAuthors:
    def test_should_read_the_name_in_reading_order_naming_each_part(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<contrib-group><contrib contrib-type="person"><name>'
                '<surname>Smith</surname><given-names>Jane A</given-names>'
                '</name></contrib></contrib-group>'
            )
        ))
        assert view_model['authors'][0]['nameParts'] == [
            {'label': 'given-names', 'text': 'Jane A'},
            {'label': 'surname', 'text': 'Smith'}
        ]

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
        assert author['email'] == {'label': 'email', 'text': 'jane@example.org'}
        assert author['affiliationRefs'] == [{'label': 'xref', 'rid': 'aff1'}]


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
            'label': 'aff',
            'id': 'aff1',
            'parts': [
                {'label': 'orgdiv2', 'text': 'A Laboratory'},
                {'label': 'orgdiv1', 'text': 'A Department'},
                {'label': 'orgname', 'text': 'A University'},
                {'label': 'city', 'text': 'A City'},
                {'label': 'country', 'text': 'A Country'}
            ]
        }]

    def test_should_leave_out_the_parts_the_document_does_not_carry(self):
        view_model = get_article_view_model(get_article_xml(
            front='<aff id="aff1"><country>A Country</country></aff>'
        ))
        assert view_model['affiliations'][0]['parts'] == [
            {'label': 'country', 'text': 'A Country'}
        ]


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
        assert view_model['abstracts'][1]['title'] == {
            'label': 'title', 'text': 'Summary'
        }


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
        assert section['title'] == {'label': 'title', 'text': 'Methods'}
        assert [block['type'] for block in section['blocks']] == [
            'paragraph', 'figure', 'paragraph'
        ]

    def test_should_keep_a_nested_section_inside_its_parent(self):
        view_model = get_article_view_model(get_article_xml(
            body='<sec><title>Outer</title><sec><title>Inner</title></sec></sec>'
        ))
        outer = view_model['sections'][0]
        assert [block['title']['text'] for block in outer['blocks']] == ['Inner']

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
        assert figure['label'] == 'fig'
        assert figure['heading'] == {'label': 'label', 'text': 'Figure 1'}
        assert get_paragraph_texts(figure['paragraphs']) == ['What it shows.']
        assert figure['graphics'] == [{'label': 'graphic', 'href': 'image1.png'}]

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
        assert [
            block['title']['text'] for block in view_model['acknowledgements']
        ] == ['Thanks']

    def test_should_read_an_appendix(self):
        view_model = get_article_view_model(get_article_xml(
            back='<app-group><app><title>Appendix 1</title><p>More.</p></app></app-group>'
        ))
        assert [
            appendix['title']['text'] for appendix in view_model['appendices']
        ] == ['Appendix 1']


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
            'label': 'ref',
            'id': 'b0',
            'authors': [
                [
                    {'label': 'given-names', 'text': 'J'},
                    {'label': 'surname', 'text': 'Smith'}
                ],
                [
                    {'label': 'given-names', 'text': 'A'},
                    {'label': 'surname', 'text': 'Jones'}
                ]
            ],
            'articleTitle': {'label': 'article-title', 'text': 'The cited title'},
            'source': {'label': 'source', 'text': 'A Journal'},
            'year': {'label': 'year', 'text': '2020'},
            'volume': {'label': 'volume', 'text': '12'},
            'issue': {'label': 'issue', 'text': '3'},
            'firstPage': {'label': 'fpage', 'text': '100'},
            'lastPage': {'label': 'lpage', 'text': '110'},
            'doi': {'label': 'pub-id', 'text': '10.1000/xyz'}
        }]

    def test_should_read_a_citation_that_states_almost_nothing(self):
        view_model = get_article_view_model(get_article_xml(
            back='<ref-list><ref id="b0"><element-citation/></ref></ref-list>'
        ))
        reference = view_model['references'][0]
        assert reference['id'] == 'b0'
        assert reference['authors'] == []
        assert reference['articleTitle'] is None


FULL_ARTICLE_XML = get_article_xml(
    front=(
        '<title-group><article-title>The title</article-title></title-group>'
        '<contrib-group><contrib><name><surname>Smith</surname>'
        '<given-names>J</given-names></name><xref ref-type="aff" rid="aff1"/>'
        '</contrib></contrib-group>'
        '<aff id="aff1"><institution content-type="orgname">A University'
        '</institution><country>A Country</country></aff>'
        '<abstract><p>An abstract.</p></abstract>'
    ),
    body=(
        '<sec><title>Methods</title><p>A paragraph.</p>'
        '<fig id="fig_0"><label>Figure 1</label>'
        '<caption><title>What it shows</title><p>A caption.</p></caption></fig>'
        '<table-wrap id="tab_0"><label>Table 1</label>'
        '<table><tbody><tr><td>a</td></tr></tbody></table></table-wrap>'
        '</sec>'
    ),
    back=(
        '<ack><sec><title>Thanks</title><p>To everyone.</p></sec></ack>'
        '<ref-list><ref id="b0"><element-citation>'
        '<source>A Journal</source></element-citation></ref></ref-list>'
    )
)


def iter_labelled_values(value) -> List[dict]:
    if isinstance(value, dict):
        if set(value) == {'label', 'text'}:
            return [value]
        return [
            found for item in value.values() for found in iter_labelled_values(item)
        ]
    if isinstance(value, list):
        return [found for item in value for found in iter_labelled_values(item)]
    return []


class TestElementLabels:
    def test_should_name_the_element_every_value_came_from(self):
        values = iter_labelled_values(get_article_view_model(FULL_ARTICLE_XML))
        assert values
        assert all(value['label'] for value in values)

    def test_should_name_the_element_of_each_kind_of_text(self):
        values = iter_labelled_values(get_article_view_model(FULL_ARTICLE_XML))
        label_by_text = {value['text']: value['label'] for value in values}
        assert label_by_text['The title'] == 'article-title'
        assert label_by_text['Smith'] == 'surname'
        assert label_by_text['A University'] == 'orgname'
        assert label_by_text['A Country'] == 'country'
        assert label_by_text['Methods'] == 'title'
        assert label_by_text['Figure 1'] == 'label'
        assert label_by_text['What it shows'] == 'title'
        assert label_by_text['A Journal'] == 'source'

    def test_should_name_the_element_a_paragraph_came_from(self):
        view_model = get_article_view_model(FULL_ARTICLE_XML)
        section = view_model['sections'][0]
        paragraph = next(
            block for block in section['blocks'] if block['type'] == 'paragraph'
        )
        assert paragraph['label'] == 'p'
        assert view_model['abstracts'][0]['paragraphs'][0]['label'] == 'p'

    def test_should_name_the_element_a_table_cell_came_from(self):
        view_model = get_article_view_model(FULL_ARTICLE_XML)
        table = next(
            block for block in view_model['sections'][0]['blocks']
            if block['type'] == 'table'
        )
        assert table['label'] == 'table-wrap'
        assert table['rows'][0][0]['label'] == 'td'


class TestRenderedArticle:
    def test_should_mark_each_value_with_the_element_it_came_from(self):
        html = get_rendered_article(FULL_ARTICLE_XML)
        for label in [
            'article-title', 'surname', 'given-names', 'orgname', 'country',
            'abstract', 'sec', 'title', 'p', 'fig', 'label', 'table-wrap',
            'td', 'ack', 'ref', 'source'
        ]:
            assert f'data-jats="{label}"' in html

    def test_should_name_the_element_beside_the_value_it_labels(self):
        html = get_rendered_article(FULL_ARTICLE_XML)
        assert (
            '<h1 data-jats="article-title">'
            '<span class="jats-tag">article-title</span>The title</h1>'
        ) in html

    def test_should_read_as_the_article_without_the_names_shown(self):
        html = get_rendered_article(FULL_ARTICLE_XML)
        text_only = re.sub(r'<span class="jats-tag">[^<]*</span>', '', html)
        text_only = re.sub(r'<[^>]+>', '', text_only)
        assert 'The title' in text_only
        assert 'J Smith' in text_only
        assert 'A University, A Country' in text_only
        assert 'A paragraph.' in text_only

    def test_should_keep_a_reference_reading_as_a_citation(self):
        html = get_rendered_article(get_article_xml(
            back=(
                '<ref-list><ref id="b0"><element-citation>'
                '<article-title>The cited title</article-title>'
                '<source>A Journal</source><year>2020</year>'
                '<volume>12</volume><fpage>100</fpage><lpage>110</lpage>'
                '<person-group><name><surname>Smith</surname>'
                '<given-names>J</given-names></name></person-group>'
                '</element-citation></ref></ref-list>'
            )
        ))
        text_only = re.sub(r'<span class="jats-tag">[^<]*</span>', '', html)
        text_only = re.sub(r'<[^>]+>', '', text_only)
        assert 'J Smith. The cited title. A Journal, 12, 100-110, 2020' in text_only

    def test_should_not_begin_a_citation_with_a_separator(self):
        html = get_rendered_article(get_article_xml(
            back=(
                '<ref-list><ref id="b0"><element-citation>'
                '<source>A Journal</source></element-citation></ref></ref-list>'
            )
        ))
        text_only = re.sub(r'<span class="jats-tag">[^<]*</span>', '', html)
        text_only = re.sub(r'<[^>]+>', '', text_only)
        assert 'References' in text_only
        assert text_only.endswith('A Journal')


FIGURE_ARTICLE_XML = get_article_xml(
    body=(
        '<sec><fig id="fig_0"><label>Figure 1</label>'
        '<caption><title>A figure</title></caption>'
        '<graphic xlink:href="image1.png"/></fig></sec>'
    )
)


class TestFigureGraphic:
    def test_should_show_the_image_where_it_travels_beside_the_xml(self):
        html = get_rendered_article(
            FIGURE_ARTICLE_XML, assets={'image1.png': 'blob:an-image'}
        )
        assert '<img' in html
        assert 'src="blob:an-image"' in html
        assert 'alt="image1.png"' in html
        assert 'class="jats-graphic-image"' in html

    def test_should_name_the_file_where_the_image_is_not_beside_the_xml(self):
        html = get_rendered_article(FIGURE_ARTICLE_XML)
        assert '<img' not in html
        assert 'image1.png (in the zip output)' in html

    def test_should_name_the_element_the_graphic_came_from(self):
        html = get_rendered_article(
            FIGURE_ARTICLE_XML, assets={'image1.png': 'blob:an-image'}
        )
        assert 'data-jats="graphic"' in html

    def test_should_name_the_file_where_the_archive_carries_a_different_one(self):
        html = get_rendered_article(
            FIGURE_ARTICLE_XML, assets={'other.png': 'blob:another-image'}
        )
        assert '<img' not in html
        assert 'image1.png (in the zip output)' in html

    def test_should_show_every_panel_a_figure_carries(self):
        html = get_rendered_article(
            get_article_xml(body=(
                '<sec><fig id="fig_0"><label>Figure 1</label>'
                '<graphic xlink:href="image1.png"/>'
                '<graphic xlink:href="image2.png"/>'
                '<graphic xlink:href="image3.png"/>'
                '</fig></sec>'
            )),
            assets={
                'image1.png': 'blob:one',
                'image2.png': 'blob:two',
                'image3.png': 'blob:three'
            }
        )
        assert html.count('<img') == 3
        for url in ['blob:one', 'blob:two', 'blob:three']:
            assert f'src="{url}"' in html

    def test_should_say_where_a_figure_names_no_file_for_its_graphic(self):
        html = get_rendered_article(get_article_xml(
            body='<sec><fig id="fig_0"><label>Figure 1</label><graphic/></fig></sec>'
        ))
        assert '<img' not in html
        assert 'no file named for this graphic' in html
        assert 'data-jats="graphic"' in html

    def test_should_name_only_the_graphics_a_reading_will_ask_for(self):
        view_model = get_article_view_model(get_article_xml(
            body=(
                '<sec><fig><graphic xlink:href="image1.png"/>'
                '<graphic xlink:href="image2.png"/><graphic/></fig>'
                '<sec><fig><graphic xlink:href="image3.png"/></fig></sec></sec>'
            ),
            back='<app-group><app><fig><graphic xlink:href="image4.png"/></fig></app></app-group>'
        ))
        assert view_model['graphicHrefs'] == [
            'image1.png', 'image2.png', 'image3.png', 'image4.png'
        ]

    def test_should_name_no_graphic_where_the_figures_name_no_file(self):
        view_model = get_article_view_model(get_article_xml(
            body='<sec><fig><graphic/></fig></sec>'
        ))
        assert view_model['graphicHrefs'] == []

    def test_should_say_which_figure_asks_for_each_graphic(self):
        view_model = get_article_view_model(get_article_xml(
            body=(
                '<sec><fig id="fig_0"><label>Figure 1</label>'
                '<graphic xlink:href="image1.png"/>'
                '<graphic xlink:href="image2.png"/></fig>'
                '<fig id="fig_1"><caption><title>An unlabelled figure</title></caption>'
                '<graphic xlink:href="image3.png"/></fig></sec>'
            )
        ))
        assert view_model['graphicReferences'] == [
            {'href': 'image1.png', 'figure': 'Figure 1'},
            {'href': 'image2.png', 'figure': 'Figure 1'},
            {'href': 'image3.png', 'figure': 'An unlabelled figure'}
        ]

    def test_should_fall_back_to_the_figure_id_where_it_has_no_name(self):
        view_model = get_article_view_model(get_article_xml(
            body='<sec><fig id="fig_7"><graphic xlink:href="image1.png"/></fig></sec>'
        ))
        assert view_model['graphicReferences'] == [
            {'href': 'image1.png', 'figure': 'fig_7'}
        ]


AFFILIATED_ARTICLE_XML = get_article_xml(
    front=(
        '<contrib-group>'
        '<contrib><name><surname>Smith</surname><given-names>J</given-names></name>'
        '<xref ref-type="aff" rid="aff1"/><xref ref-type="aff" rid="aff2"/></contrib>'
        '<contrib><name><surname>Jones</surname><given-names>A</given-names></name>'
        '<xref ref-type="aff" rid="aff2"/></contrib>'
        '</contrib-group>'
        '<aff id="aff1"><institution content-type="orgname">A University</institution></aff>'
        '<aff id="aff2"><institution content-type="orgname">An Institute</institution></aff>'
    )
)


class TestAffiliationReferences:
    def test_should_point_each_reference_at_the_affiliation_it_names(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        assert 'href="#aff1"' in html
        assert 'href="#aff2"' in html
        assert 'id="aff1"' in html
        assert 'id="aff2"' in html

    def test_should_read_as_the_number_the_list_gives_the_affiliation(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        text_only = re.sub(r'<span class="jats-tag">[^<]*</span>', '', html)
        text_only = re.sub(r'<[^>]+>', '', text_only)
        assert 'J Smith 1 2, A Jones 2' in text_only

    def test_should_carry_the_affiliation_text_to_rest_on(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        assert 'title="A University"' in html
        assert 'title="An Institute"' in html

    def test_should_name_the_element_a_reference_came_from(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        assert 'data-jats="xref"' in html

    def test_should_say_where_a_reference_points_at_nothing(self):
        html = get_rendered_article(get_article_xml(
            front=(
                '<contrib-group><contrib>'
                '<name><surname>Smith</surname></name>'
                '<xref ref-type="aff" rid="aff9"/></contrib></contrib-group>'
            )
        ))
        assert 'title="no affiliation with id aff9"' in html
        assert '>aff9</a>' in html

    def test_should_keep_an_affiliation_the_document_repeats(self):
        view_model = get_article_view_model(get_article_xml(
            front=(
                '<aff id="aff1"><institution content-type="orgname">A University'
                '</institution></aff>'
                '<aff id="aff2"><institution content-type="orgname">A University'
                '</institution></aff>'
            )
        ))
        assert [
            affiliation['id'] for affiliation in view_model['affiliations']
        ] == ['aff1', 'aff2']

    def test_should_keep_what_separates_two_authors_inside_the_first(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        authors = re.search(r'<p class="jats-authors">.*?</p>', html, re.DOTALL)
        assert authors
        # nothing between one author and the next, so a line each strands nothing
        assert '</span><span data-jats="contrib"' in authors.group(0)
        assert authors.group(0).count('class="jats-separator"') == 1

    def test_should_separate_no_author_where_there_is_only_one(self):
        html = get_rendered_article(get_article_xml(
            front=(
                '<contrib-group><contrib><name><surname>Smith</surname>'
                '</name></contrib></contrib-group>'
            )
        ))
        assert 'jats-separator' not in html


class TestElementIds:
    def test_should_say_which_one_an_element_with_an_id_is(self):
        html = get_rendered_article(get_article_xml(
            body=(
                '<sec id="sec1"><title>Methods</title>'
                '<fig id="fig_0"><label>Figure 1</label></fig>'
                '<table-wrap id="tab_0"><label>Table 1</label></table-wrap></sec>'
            ),
            back='<ref-list><ref id="b0"><element-citation/></ref></ref-list>'
        ))
        for element_id in ['sec1', 'fig_0', 'tab_0', 'b0']:
            assert f'<span class="jats-tag-id">#{element_id}</span>' in html

    def test_should_say_nothing_extra_where_an_element_has_no_id(self):
        html = get_rendered_article(get_article_xml(
            body='<sec><title>Methods</title></sec>'
        ))
        assert '<span class="jats-tag">sec</span>' in html
        assert 'jats-tag-id' not in html

    def test_should_say_which_one_an_affiliation_is(self):
        html = get_rendered_article(AFFILIATED_ARTICLE_XML)
        assert '<span class="jats-tag-id">#aff1</span>' in html
        assert '<span class="jats-tag-id">#aff2</span>' in html
