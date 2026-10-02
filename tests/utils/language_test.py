from sciencebeam_parser.utils.language import detect_language


ENGLISH_1 = (
    'This study reports on the effects of the intervention that was carried out in a'
    ' sample of adolescents, and on the data that were collected from the participants'
    ' before and after it was delivered by the research team at the centre.'
)

PORTUGUESE_1 = (
    'Este estudo relata os efeitos da intervenção que foi realizada em uma amostra de'
    ' adolescentes, e os dados que foram coletados dos participantes antes e depois de'
    ' ter sido aplicada pela equipe de pesquisa no centro.'
)

SPANISH_1 = (
    'Este estudio informa sobre los efectos de la intervención que se llevó a cabo en'
    ' una muestra de adolescentes, y sobre los datos que se recogieron de los'
    ' participantes antes y después de que fuera aplicada por el equipo en el centro.'
)


PORTUGUESE_TITLE_AND_ENGLISH_TITLE_1 = (
    'A pandemia de COVID-19, o isolamento social, consequências na saúde mental e'
    ' estratégias de enfrentamento: uma revisão integrativa'
    ' The COVID-19 pandemic, social isolation, consequences on mental health and coping'
    ' strategies: an integrative review'
)


class TestDetectLanguage:
    def test_should_detect_english(self):
        assert detect_language(ENGLISH_1) == 'en'

    def test_should_detect_portuguese(self):
        assert detect_language(PORTUGUESE_1) == 'pt'

    def test_should_distinguish_spanish_from_portuguese(self):
        assert detect_language(SPANISH_1) == 'es'

    def test_should_return_none_for_text_shorter_than_the_minimum(self):
        assert detect_language('the of and to in a is that for') is None

    def test_should_return_none_where_no_stopword_appears(self):
        assert detect_language(' '.join(['Leishmania'] * 40)) is None

    def test_should_return_none_for_text_written_in_two_languages_at_once(self):
        assert detect_language(PORTUGUESE_TITLE_AND_ENGLISH_TITLE_1) is None

    def test_should_return_none_for_empty_text(self):
        assert detect_language('') is None
