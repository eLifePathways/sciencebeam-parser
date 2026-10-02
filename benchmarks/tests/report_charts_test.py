from __future__ import annotations

import pytest

from benchmarks.report_charts import SERIES_COLOURS, ChartSpec, chart_markdown, render_chart


def _spec(
    field: str = "abstract",
    method: str = "levenshtein",
    scope: str = "all",
    series=("base", "head"),
    values=((0.4, 0.5), (0.45, 0.55)),
    corpora=("biorxiv", "ore"),
    n_docs: int = 40,
) -> ChartSpec:
    return ChartSpec(
        field=field, method=method, scope=scope, n_docs=n_docs,
        corpora=corpora, series=series, values=values,
    )


class TestChartSpec:
    def test_should_name_the_file_after_the_row_it_draws(self):
        assert _spec().filename == "abstract-levenshtein-all.png"

    def test_should_distinguish_the_gold_scope_in_the_filename(self):
        assert _spec(scope="gold").filename == "abstract-levenshtein-gold.png"

    def test_should_say_which_scope_it_covers(self):
        assert _spec(scope="gold", n_docs=12).caption == (
            "over the 12 documents whose gold records it"
        )

    def test_should_describe_itself_for_a_reader_who_cannot_see_it(self):
        assert _spec().alt_text == (
            "abstract (levenshtein) — f1 by corpus, over all 40 documents"
        )


class TestChartMarkdown:
    def test_should_be_empty_without_any_chart(self):
        assert not chart_markdown([], "charts")

    def test_should_link_by_relative_path_by_default(self):
        lines = chart_markdown([_spec()], "charts")
        assert "(charts/abstract-levenshtein-all.png)" in lines[2]

    def test_should_link_under_the_base_url_where_one_is_given(self):
        lines = chart_markdown([_spec()], "charts", base_url="https://example.org/a/")
        assert "(https://example.org/a/abstract-levenshtein-all.png)" in lines[2]

    def test_should_apply_the_prefix_to_the_filename(self):
        lines = chart_markdown([_spec()], "charts", prefix="abc123-")
        assert "(charts/abc123-abstract-levenshtein-all.png)" in lines[2]

    def test_should_carry_the_alt_text(self):
        lines = chart_markdown([_spec()], "charts")
        assert lines[2].startswith("![abstract (levenshtein) — f1 by corpus, over all 40")


class TestRenderChart:
    def test_should_write_a_png_named_for_the_row(self, tmp_path):
        path = render_chart(_spec(), tmp_path)
        assert path == tmp_path / "abstract-levenshtein-all.png"
        assert path.read_bytes().startswith(b"\x89PNG")

    def test_should_apply_the_prefix_so_runs_do_not_overwrite_each_other(self, tmp_path):
        path = render_chart(_spec(), tmp_path, prefix="sha1-")
        assert path.name == "sha1-abstract-levenshtein-all.png"

    def test_should_draw_the_same_bytes_twice(self, tmp_path):
        first = render_chart(_spec(), tmp_path, prefix="a-").read_bytes()
        second = render_chart(_spec(), tmp_path, prefix="b-").read_bytes()
        assert first == second

    def test_should_draw_a_row_a_variant_did_not_score(self, tmp_path):
        path = render_chart(_spec(values=((0.4, None), (None, 0.55))), tmp_path)
        assert path.exists()

    def test_should_refuse_more_variants_than_stay_distinguishable(self, tmp_path):
        too_many = len(SERIES_COLOURS) + 1
        spec = _spec(
            series=tuple(f"run{index}" for index in range(too_many)),
            values=tuple((0.5, 0.5) for _ in range(too_many)),
        )
        with pytest.raises(ValueError, match="distinguishable"):
            render_chart(spec, tmp_path)
