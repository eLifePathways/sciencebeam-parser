from typing import List, Optional

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutLink,
    LayoutPageCoordinates,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import (
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticReference,
    SemanticTitle
)
from sciencebeam_parser.models.citation.links import (
    get_doi_for_link_uri,
    set_reference_doi_from_links
)


DOI_1 = '10.12345/fern.2001'
DOI_2 = '10.12345/moss.2002'

TOKEN_COORDINATES = LayoutPageCoordinates(x=100, y=200, width=50, height=10, page_number=1)


def _link_over(coordinates: LayoutPageCoordinates, uri: str) -> LayoutLink:
    return LayoutLink(uri=uri, coordinates_list=[coordinates])


def _get_doi_values(reference: SemanticReference) -> List[Optional[str]]:
    return [
        identifier.value
        for identifier in reference.iter_by_type(SemanticExternalIdentifier)
        if identifier.external_identifier_type == SemanticExternalIdentifierTypes.DOI
    ]


class TestGetDoiForLinkUri:
    def test_should_take_the_doi_from_a_doi_org_link(self):
        assert get_doi_for_link_uri(f'https://doi.org/{DOI_1}') == DOI_1

    def test_should_take_the_doi_from_a_dx_doi_org_link(self):
        assert get_doi_for_link_uri(f'http://dx.doi.org/{DOI_1}') == DOI_1

    def test_should_take_the_doi_from_a_publisher_link(self):
        assert get_doi_for_link_uri(f'https://publisher.example/doi/{DOI_1}') == DOI_1

    def test_should_drop_a_malformed_doi_org_prefix(self):
        assert get_doi_for_link_uri(f'https://doi.org:{DOI_1}') == DOI_1

    def test_should_take_a_bare_doi(self):
        assert get_doi_for_link_uri(f'doi:{DOI_1}') == DOI_1

    def test_should_ignore_a_link_without_a_doi(self):
        assert get_doi_for_link_uri('https://example.org/ferns') is None


class TestSetReferenceDoiFromLinks:
    def test_should_add_the_doi_of_a_link_over_the_reference(self):
        token = LayoutToken('Ferns', coordinates=TOKEN_COORDINATES)
        reference = SemanticReference([SemanticTitle(LayoutBlock.for_tokens([token]))])
        set_reference_doi_from_links(
            reference, [token], [_link_over(TOKEN_COORDINATES, f'https://doi.org/{DOI_1}')]
        )
        assert _get_doi_values(reference) == [DOI_1]

    def test_should_replace_a_printed_doi(self):
        token = LayoutToken(DOI_2, coordinates=TOKEN_COORDINATES)
        reference = SemanticReference([SemanticExternalIdentifier(
            LayoutBlock.for_tokens([token]),
            value=DOI_2,
            external_identifier_type=SemanticExternalIdentifierTypes.DOI
        )])
        set_reference_doi_from_links(
            reference, [token], [_link_over(TOKEN_COORDINATES, f'https://doi.org/{DOI_1}')]
        )
        assert _get_doi_values(reference) == [DOI_1]

    def test_should_take_the_last_of_several_links(self):
        token = LayoutToken('Ferns', coordinates=TOKEN_COORDINATES)
        reference = SemanticReference()
        set_reference_doi_from_links(reference, [token], [
            _link_over(TOKEN_COORDINATES, f'https://doi.org/{DOI_1}'),
            _link_over(TOKEN_COORDINATES, f'https://doi.org/{DOI_2}')
        ])
        assert _get_doi_values(reference) == [DOI_2]

    def test_should_ignore_a_link_elsewhere_on_the_page(self):
        token = LayoutToken('Ferns', coordinates=TOKEN_COORDINATES)
        reference = SemanticReference()
        set_reference_doi_from_links(reference, [token], [
            _link_over(TOKEN_COORDINATES.move_by(dy=100), f'https://doi.org/{DOI_1}')
        ])
        assert not _get_doi_values(reference)

    def test_should_ignore_a_link_without_a_doi(self):
        token = LayoutToken('Ferns', coordinates=TOKEN_COORDINATES)
        reference = SemanticReference()
        set_reference_doi_from_links(
            reference, [token], [_link_over(TOKEN_COORDINATES, 'https://example.org/ferns')]
        )
        assert not _get_doi_values(reference)
