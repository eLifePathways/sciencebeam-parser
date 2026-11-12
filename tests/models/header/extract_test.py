from typing import List

from sciencebeam_parser.document.semantic_document import (
    SemanticAbstract,
    SemanticFront,
    SemanticRawAddress,
    SemanticRawAffiliation,
    SemanticRawAffiliationAddress,
    SemanticTitle
)
from sciencebeam_parser.document.layout_document import (
    LOGGER,
    LayoutBlock,
    LayoutFont,
    LayoutLine,
    LayoutLineMeta,
    LayoutPageCoordinates,
    LayoutToken,
    join_layout_tokens
)

from sciencebeam_parser.models.header.extract import (
    AbstractsMode,
    TitlesMode,
    HeaderSemanticExtractor,
    _split_trailing_title_punct,
    get_cleaned_abstract_text,
    get_cleaned_abstract_layout_block,
    iter_titles_split_by_language
)


TITLE_1 = 'the title 1'
ABSTRACT_1 = 'the abstract 1'
AUTHOR_1 = 'Author 1'
AUTHOR_2 = 'Author 2'
AFFILIATION_1 = 'Affiliation 1'
AFFILIATION_2 = 'Affiliation 2'
ADDRESS_1 = 'Address 1'
ADDRESS_2 = 'Address 2'

OTHER_1 = 'Other 1'

LONG_ABSTRACT_1 = ' '.join(['first variant text'] * 20)
LONG_ABSTRACT_2 = ' '.join(['second variant text'] * 20)

ENGLISH_ABSTRACT_1 = (
    'This study reports on the effects of the intervention that was carried out in a'
    ' sample of adolescents, and on the data that were collected from the participants.'
)

UNKNOWN_LANGUAGE_ABSTRACT_1 = ' '.join(['Leishmania mexicana'] * 15)

ENGLISH_ABSTRACT_2 = (
    'Background the incidence of the disease has risen steadily over the past decade and'
    ' the data were collected from the national registry by the research team.'
)

PORTUGUESE_ABSTRACT_1 = (
    'Este estudo relata os efeitos da intervenção que foi realizada em uma amostra de'
    ' adolescentes, e os dados que foram coletados dos participantes antes e depois.'
)


class TestGetCleanedAbstractText:
    def test_should_return_none_if_passed_in_text_was_none(self):
        assert get_cleaned_abstract_text(None) is None

    def test_should_return_empty_str_if_passed_in_text_was_empty(self):
        assert get_cleaned_abstract_text('') == ''

    def test_should_return_abstract_if_it_doesnt_contain_prefix(self):
        assert get_cleaned_abstract_text(ABSTRACT_1) == ABSTRACT_1

    def test_should_return_remove_abstract_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract ' + ABSTRACT_1
        ) == ABSTRACT_1

    def test_should_return_remove_abstract_dot_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract. ' + ABSTRACT_1
        ) == ABSTRACT_1

    def test_should_return_remove_abstract_colon_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract: ' + ABSTRACT_1
        ) == ABSTRACT_1


class TestGetCleanedAbstractLayoutBlock:
    def test_should_return_none_if_passed_in_text_was_none(self):
        assert get_cleaned_abstract_layout_block(None) is None

    def test_should_return_empty_str_if_passed_in_text_was_empty(self):
        layout_block = LayoutBlock(lines=[])
        assert get_cleaned_abstract_layout_block(layout_block) == layout_block

    def test_should_return_abstract_if_it_doesnt_contain_prefix(self):
        layout_block = LayoutBlock.for_text(ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert cleaned_layout_block is not None
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert cleaned_layout_block is not None
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_dot_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract. ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert cleaned_layout_block is not None
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_colon_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract: ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert cleaned_layout_block is not None
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1


class TestSplitTrailingTitlePunct:
    def test_no_trailing_punct_returns_unchanged(self):
        block = LayoutBlock.for_text(TITLE_1)
        clean, trailing = _split_trailing_title_punct(block)
        assert join_layout_tokens(clean.lines[0].tokens) == TITLE_1
        assert trailing == ''

    def test_trailing_dot_is_stripped_and_returned(self):
        block = LayoutBlock.for_text(TITLE_1 + ' .')
        clean, trailing = _split_trailing_title_punct(block)
        assert join_layout_tokens(clean.lines[0].tokens) == TITLE_1
        assert trailing == '.'

    def test_empty_block_returns_unchanged(self):
        block = LayoutBlock(lines=[])
        clean, trailing = _split_trailing_title_punct(block)
        assert clean == block
        assert trailing == ''


ENGLISH_TITLE_1 = 'Inpatient flow for the winter season in the northern health regions'

PORTUGUESE_TITLE_1 = (
    'Fluxo de internacao no inverno e as regioes de saude que foram vistas em uma analise'
)

ENGLISH_TITLE_FIRST_HALF_1 = 'Inpatient flow for the winter season in the northern'

ENGLISH_TITLE_SECOND_HALF_1 = 'health regions of a country that was studied for this report'

SPANISH_TITLE_TIED_1 = 'EVALUACION DE LA MIGRACION DE CONTENIDOS'

NAME_ONLY_CONTINUATION_1 = 'Lisboa Sampaio Oliveira Niskier Caceres Mendivil'


TITLE_FONT_1 = LayoutFont(font_id='title')
LABEL_FONT_1 = LayoutFont(font_id='label')


def _get_title_line(
    text: str, block_id: int, font: LayoutFont = TITLE_FONT_1
) -> LayoutLine:
    line_meta = LayoutLineMeta(line_id=block_id, block_id=block_id)
    return LayoutLine(tokens=[
        LayoutToken(token_text, font=font, line_meta=line_meta)
        for token_text in text.split(' ')
    ])


def _get_title_block(*text_and_block_id) -> LayoutBlock:
    return LayoutBlock(lines=[
        _get_title_line(*args) for args in text_and_block_id
    ])


class TestIterTitlesSplitByLanguage:
    def test_should_cut_between_the_blocks_each_language_is_set_in(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_1, 1), (PORTUGUESE_TITLE_1, 2)
        )))
        assert [language for _, language in titles] == ['en', 'pt']
        assert [block.text for block, _ in titles] == [ENGLISH_TITLE_1, PORTUGUESE_TITLE_1]

    def test_should_cut_between_lines_where_one_block_holds_both(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_1, 1), (PORTUGUESE_TITLE_1, 1)
        )))
        assert [language for _, language in titles] == ['en', 'pt']

    def test_should_keep_one_title_set_across_two_blocks_whole(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_FIRST_HALF_1, 1), (ENGLISH_TITLE_SECOND_HALF_1, 2)
        )))
        assert len(titles) == 1
        assert titles[0][1] == 'en'

    def test_should_keep_a_title_whose_language_cannot_be_told_whole(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (TITLE_1, 1), (TITLE_1, 2)
        )))
        assert len(titles) == 1
        assert titles[0][1] is None

    def test_should_keep_a_run_of_neither_language_with_the_one_before_it(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_1, 1), (TITLE_1, 2), (PORTUGUESE_TITLE_1, 3)
        )))
        assert [language for _, language in titles] == ['en', 'pt']
        assert titles[0][0].text == ENGLISH_TITLE_1 + ' ' + TITLE_1


SHORT_TITLE_1 = 'Foreword'

SHORT_TITLE_2 = 'Prefacio'

TITLE_LABEL_1 = 'Titulo'

RUNNING_HEAD_1 = 'Mental Health of Dentists and the Pandemic'

ENGLISH_TITLE_2 = 'Mental Health of Dentists during the Pandemic a Critical Review'


class TestTitleFurniture:
    def test_should_drop_a_label_naming_the_field_beside_the_title(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_1, 1, TITLE_FONT_1), (TITLE_LABEL_1, 2, LABEL_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [ENGLISH_TITLE_1]

    def test_should_keep_a_short_title_beside_another_short_title(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (SHORT_TITLE_1, 1, TITLE_FONT_1), (SHORT_TITLE_2, 2, LABEL_FONT_1)
        )))
        assert SHORT_TITLE_2 in ' '.join(block.text for block, _ in titles)

    def test_should_keep_a_short_continuation_set_in_the_title_font(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_FIRST_HALF_1, 1, TITLE_FONT_1), ('regions', 2, TITLE_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [
            ENGLISH_TITLE_FIRST_HALF_1 + ' regions'
        ]

    def test_should_drop_a_running_head_where_the_title_is_set_over_several_runs(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            ('Mental Health of Dentists during the Pandemic a', 1, TITLE_FONT_1),
            ('Critical Review', 2, TITLE_FONT_1),
            (RUNNING_HEAD_1, 3, TITLE_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [
            'Mental Health of Dentists during the Pandemic a Critical Review'
        ]

    def test_should_drop_a_label_where_the_title_is_set_over_several_runs(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            ('Worldwide differences in winter related', 1, TITLE_FONT_1),
            ('mortality rates', 2, TITLE_FONT_1),
            (TITLE_LABEL_1, 3, LABEL_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [
            'Worldwide differences in winter related mortality rates'
        ]

    def test_should_drop_a_running_head_repeating_the_titles_words(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_2, 1, TITLE_FONT_1), (RUNNING_HEAD_1, 2, TITLE_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [ENGLISH_TITLE_2]

    def test_should_keep_a_title_given_in_another_language(self):
        titles = list(iter_titles_split_by_language(_get_title_block(
            (ENGLISH_TITLE_1, 1, TITLE_FONT_1), (PORTUGUESE_TITLE_1, 2, TITLE_FONT_1)
        )))
        assert [block.text for block, _ in titles] == [
            ENGLISH_TITLE_1, PORTUGUESE_TITLE_1
        ]


class TestIterTitlesSplitByLanguageAmongTheDocumentsLanguages:
    def test_should_name_a_tied_run_among_the_languages_the_document_carries(self):
        titles = list(iter_titles_split_by_language(
            _get_title_block((SPANISH_TITLE_TIED_1, 1), (ENGLISH_TITLE_1, 2)),
            frozenset({'es', 'en'})
        ))
        assert [language for _, language in titles] == ['es', 'en']

    def test_should_keep_the_title_whole_without_the_documents_languages(self):
        titles = list(iter_titles_split_by_language(
            _get_title_block((SPANISH_TITLE_TIED_1, 1), (ENGLISH_TITLE_1, 2))
        ))
        assert len(titles) == 1

    def test_should_not_name_a_run_carrying_no_word_of_that_language(self):
        titles = list(iter_titles_split_by_language(
            _get_title_block((PORTUGUESE_TITLE_1, 1), (NAME_ONLY_CONTINUATION_1, 2)),
            frozenset({'en'})
        ))
        assert [language for _, language in titles] == ['pt']

    def test_should_not_name_a_run_too_short_to_be_a_title(self):
        titles = list(iter_titles_split_by_language(
            _get_title_block((PORTUGUESE_TITLE_1, 1), (TITLE_1, 2)),
            frozenset({'en'})
        ))
        assert [language for _, language in titles] == ['pt']

    def test_should_keep_one_title_set_across_two_blocks_whole(self):
        titles = list(iter_titles_split_by_language(
            _get_title_block(
                (ENGLISH_TITLE_FIRST_HALF_1, 1), (ENGLISH_TITLE_SECOND_HALF_1, 2)
            ),
            frozenset({'en'})
        ))
        assert len(titles) == 1


def _get_abstract_block(text: str, page_number: int) -> LayoutBlock:
    coordinates = LayoutPageCoordinates(
        x=0, y=0, width=1, height=1, page_number=page_number
    )
    return LayoutBlock(lines=[
        LayoutLine(tokens=[
            token._replace(coordinates=coordinates) for token in line.tokens
        ])
        for line in LayoutBlock.for_text(text).lines
    ])


def _get_abstracts(entity_tokens, **kwargs) -> List[str]:
    front = SemanticFront(list(
        HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
            entity_tokens, **kwargs
        )
    ))
    return [abstract.get_text() for abstract in front.iter_by_type(SemanticAbstract)]


class TestHeaderSemanticExtractorAbstractHeading:
    def _get_headings(self, entity_tokens, **kwargs):
        front = SemanticFront(list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
                entity_tokens, **kwargs
            )
        ))
        return [
            abstract.heading_block.text if abstract.heading_block is not None else None
            for abstract in front.iter_by_type(SemanticAbstract)
        ]

    def test_should_carry_the_heading_labelled_above_the_abstract(self):
        assert self._get_headings([
            ('O', LayoutBlock.for_text('Plain language summary')),
            ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
        ]) == ['Plain language summary']

    def test_should_take_the_heading_from_the_last_line_above_the_abstract(self):
        heading_block = LayoutBlock(lines=[
            LayoutBlock.for_text('Word Count: 1,215').lines[0],
            LayoutBlock.for_text('ABSTRACT').lines[0]
        ])
        assert self._get_headings([
            ('O', heading_block),
            ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
        ]) == ['ABSTRACT']

    def test_should_keep_the_heading_where_abstracts_are_merged_by_language(self):
        assert self._get_headings([
            ('O', LayoutBlock.for_text('Abstract')),
            ('<abstract>', LayoutBlock.for_text(
                'This study examines the way in which the words of a language'
                ' are used by the people who speak it.'
            )),
            ('<abstract>', LayoutBlock.for_text(
                'The results show that the words of a language are used in the'
                ' way that the people who speak it expect them to be.'
            ))
        ], abstracts_mode=AbstractsMode.MERGED_BY_LANGUAGE) == ['Abstract']

    def test_should_carry_no_heading_where_the_block_above_is_too_long(self):
        assert self._get_headings([
            ('O', LayoutBlock.for_text(
                'This manuscript was compiled on the fourth of July in the year'
            )),
            ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
        ]) == [None]

    def test_should_carry_no_heading_where_another_label_sits_above(self):
        assert self._get_headings([
            ('<title>', LayoutBlock.for_text(TITLE_1)),
            ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
        ]) == [None]


class TestHeaderSemanticExtractorContinuedAbstract:
    def test_should_join_a_block_continuing_the_abstract_on_the_next_page(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('<abstract>', _get_abstract_block('continues on the next page.', 2))
        ]) == ['The abstract begins here and continues on the next page.']

    def test_should_join_it_where_the_default_mode_would_have_dropped_it(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('<abstract>', _get_abstract_block('continues on the next page.', 2))
        ], abstracts_mode=AbstractsMode.FIRST) == [
            'The abstract begins here and continues on the next page.'
        ]

    def test_should_keep_a_block_with_a_heading_above_it_apart(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('O', _get_abstract_block('Plain language summary', 2)),
            ('<abstract>', _get_abstract_block(
                'this summary says the same thing in fewer words.', 2
            ))
        ], abstracts_mode=AbstractsMode.VARIANTS) == [
            'The abstract begins here and',
            'this summary says the same thing in fewer words.'
        ]

    def test_should_keep_a_block_apart_where_the_abstract_above_it_is_finished(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins and ends here.', 1)),
            ('<abstract>', _get_abstract_block(
                'bioetica, aspectos legales y politicos del asunto.', 2
            ))
        ], abstracts_mode=AbstractsMode.VARIANTS) == [
            'The abstract begins and ends here.',
            'bioetica, aspectos legales y politicos del asunto.'
        ]

    def test_should_keep_a_block_opening_a_new_sentence_apart(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('<abstract>', _get_abstract_block('O resumo comeca aqui tambem.', 2))
        ], abstracts_mode=AbstractsMode.VARIANTS) == [
            'The abstract begins here and', 'O resumo comeca aqui tambem.'
        ]

    def test_should_keep_a_block_pages_away_apart(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('<abstract>', _get_abstract_block('continues much later on.', 5))
        ], abstracts_mode=AbstractsMode.VARIANTS) == [
            'The abstract begins here and', 'continues much later on.'
        ]

    def test_should_keep_a_long_block_above_it_from_counting_as_a_heading(self):
        assert _get_abstracts([
            ('<abstract>', _get_abstract_block('The abstract begins here and', 1)),
            ('O', _get_abstract_block(
                'This manuscript was compiled on the fourth of July in the year', 2
            )),
            ('<abstract>', _get_abstract_block('continues on the next page.', 2))
        ]) == ['The abstract begins here and continues on the next page.']


class TestHeaderSemanticExtractorTitles:
    def test_should_keep_one_title_labelled_in_two_languages_whole_where_asked(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
                [('<title>', _get_title_block(
                    (ENGLISH_TITLE_1, 1), (PORTUGUESE_TITLE_1, 2)
                ))],
                titles_mode=TitlesMode.AS_LABELLED
            )
        )
        front = SemanticFront(semantic_content_list)
        assert len(list(front.iter_by_type(SemanticTitle))) == 1

    def test_should_split_a_title_labelled_in_two_languages(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
                [('<title>', _get_title_block(
                    (ENGLISH_TITLE_1, 1), (PORTUGUESE_TITLE_1, 2)
                ))],
                titles_mode=TitlesMode.SPLIT_BY_LANGUAGE
            )
        )
        front = SemanticFront(semantic_content_list)
        semantic_titles = list(front.iter_by_type(SemanticTitle))
        assert [title.get_text() for title in semantic_titles] == [
            ENGLISH_TITLE_1, PORTUGUESE_TITLE_1
        ]
        assert [title.language for title in semantic_titles] == ['en', 'pt']

    def test_should_keep_the_trailing_dot_on_the_last_of_the_split_titles(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
                [('<title>', _get_title_block(
                    (ENGLISH_TITLE_1, 1), (PORTUGUESE_TITLE_1 + ' .', 2)
                ))],
                titles_mode=TitlesMode.SPLIT_BY_LANGUAGE
            )
        )
        front = SemanticFront(semantic_content_list)
        semantic_titles = list(front.iter_by_type(SemanticTitle))
        assert [title.trailing_text for title in semantic_titles] == ['', '.']


class TestHeaderSemanticExtractor:
    def test_should_set_title_and_abstract(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert front.get_text_by_type(SemanticTitle) == TITLE_1
        assert front.get_text_by_type(SemanticAbstract) == ABSTRACT_1

    def test_should_strip_trailing_dot_from_title_into_trailing_text(self):
        title_with_dot = TITLE_1 + ' .'
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(title_with_dot))
            ])
        )
        front = SemanticFront(semantic_content_list)
        semantic_title = next(front.iter_by_type(SemanticTitle))
        assert front.get_text_by_type(SemanticTitle) == TITLE_1
        assert semantic_title.trailing_text == '.'

    def test_should_keep_every_title_the_page_prints(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<title>', LayoutBlock.for_text('other'))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert [
            title.get_text() for title in front.iter_by_type(SemanticTitle)
        ] == [TITLE_1, 'other']

    def test_should_keep_a_title_the_page_prints_again_only_once(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<title>', LayoutBlock.for_text(TITLE_1.upper() + ' .'))
            ])
        )
        front = SemanticFront(semantic_content_list)
        assert [
            title.get_text() for title in front.iter_by_type(SemanticTitle)
        ] == [TITLE_1]

    def test_should_keep_two_titles_that_say_different_things(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<title>', LayoutBlock.for_text('other'))
            ])
        )
        front = SemanticFront(semantic_content_list)
        assert [
            title.get_text() for title in front.iter_by_type(SemanticTitle)
        ] == [TITLE_1, 'other']

    def _abstracts(self, texts, abstracts_mode=None):
        kwargs = {} if abstracts_mode is None else {'abstracts_mode': abstracts_mode}
        front = SemanticFront(list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks(
                [('<abstract>', LayoutBlock.for_text(text)) for text in texts],
                **kwargs
            )
        ))
        return list(front.iter_by_type(SemanticAbstract))

    def test_should_keep_only_the_first_abstract_by_default(self):
        abstracts = self._abstracts([LONG_ABSTRACT_1, LONG_ABSTRACT_2])
        assert [
            join_layout_tokens(list(abstract.merged_block.iter_all_tokens()))
            for abstract in abstracts
        ] == [LONG_ABSTRACT_1]

    def test_should_not_declare_a_language_by_default(self):
        abstracts = self._abstracts([ENGLISH_ABSTRACT_1])
        assert [abstract.language for abstract in abstracts] == [None]

    def test_should_add_additional_abstract_as_further_abstract(self):
        abstracts = self._abstracts(
            [LONG_ABSTRACT_1, LONG_ABSTRACT_2], AbstractsMode.VARIANTS
        )
        assert [
            join_layout_tokens(list(abstract.merged_block.iter_all_tokens()))
            for abstract in abstracts
        ] == [LONG_ABSTRACT_1, LONG_ABSTRACT_2]

    def test_should_ignore_additional_abstract_much_shorter_than_the_first(self):
        abstracts = self._abstracts([LONG_ABSTRACT_1, 'other'], AbstractsMode.VARIANTS)
        assert len(abstracts) == 1

    def test_should_detect_the_language_of_each_abstract(self):
        abstracts = self._abstracts(
            [ENGLISH_ABSTRACT_1, PORTUGUESE_ABSTRACT_1], AbstractsMode.VARIANTS
        )
        assert [abstract.language for abstract in abstracts] == ['en', 'pt']

    def test_should_leave_the_language_unset_where_it_cannot_tell(self):
        abstracts = self._abstracts([ABSTRACT_1], AbstractsMode.VARIANTS)
        assert [abstract.language for abstract in abstracts] == [None]

    def test_should_merge_abstracts_sharing_a_language(self):
        abstracts = self._abstracts(
            [ENGLISH_ABSTRACT_1, PORTUGUESE_ABSTRACT_1, ENGLISH_ABSTRACT_2],
            AbstractsMode.MERGED_BY_LANGUAGE
        )
        assert [abstract.language for abstract in abstracts] == ['en', 'pt']
        merged_english = abstracts[0].get_text()
        assert ENGLISH_ABSTRACT_1 in merged_english
        assert ENGLISH_ABSTRACT_2 in merged_english

    def test_should_keep_an_abstract_of_unknown_language_on_its_own_when_merging(self):
        abstracts = self._abstracts(
            [ENGLISH_ABSTRACT_1, UNKNOWN_LANGUAGE_ABSTRACT_1, ENGLISH_ABSTRACT_2],
            AbstractsMode.MERGED_BY_LANGUAGE
        )
        assert [abstract.language for abstract in abstracts] == ['en', None]

    def test_should_add_raw_authors(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<author>', LayoutBlock.for_text(AUTHOR_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert front.get_raw_authors_text() == AUTHOR_1

    def test_should_add_raw_affiliation_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<address>', LayoutBlock.for_text(ADDRESS_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert len(aff_address_list) == 1
        aff_address = aff_address_list[0]
        assert aff_address.get_text_by_type(SemanticRawAffiliation) == AFFILIATION_1
        assert aff_address.get_text_by_type(SemanticRawAddress) == ADDRESS_1

    def test_should_split_raw_affiliation_on_new_aff_without_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]

    def test_should_split_raw_affiliation_on_new_aff_with_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<address>', LayoutBlock.for_text(ADDRESS_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2)),
                ('<address>', LayoutBlock.for_text(ADDRESS_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]
        assert [
            aff_address.get_text_by_type(SemanticRawAddress)
            for aff_address in aff_address_list
        ] == [ADDRESS_1, ADDRESS_2]

    def test_should_split_raw_affiliation_separated_by_other(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('O', LayoutBlock.for_text(OTHER_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]

    def test_should_split_raw_affiliation_separated_by_known_label(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<author>', LayoutBlock.for_text(AUTHOR_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]
