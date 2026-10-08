from __future__ import annotations

import pytest

from benchmarks.report_charts import (
    DPI,
    SERIES_COLOURS,
    ChartSpec,
    _draw_bars,
    ComputeChartSpec,
    _drop_colliding_labels,
    _legend_layout,
    chart_markdown,
    render_compute_chart,
    render_chart,
)


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
            "Abstract (levenshtein) — f1 by corpus, over all 40 documents"
        )

    def test_should_name_the_field_the_way_a_reader_would(self):
        assert _spec(field="author_full_names").title.startswith("Authors (")


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
        assert lines[2].startswith("![Abstract (levenshtein) — f1 by corpus, over all 40")


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


class TestCrowdedLabels:
    """Two numbers running into each other are worse than no numbers."""

    def _spec(self, n_series: int, n_corpora: int) -> ChartSpec:
        corpora = tuple(f"corpus_{index}" for index in range(n_corpora))
        return ChartSpec(
            field="reference_title", method="levenshtein", scope="all", n_docs=60,
            corpora=corpora,
            series=tuple(f"run {index}" for index in range(n_series)),
            values=tuple(
                tuple(0.5 + 0.01 * index for _ in corpora) for index in range(n_series)
            ),
        )

    def _surviving_labels(self, spec: ChartSpec) -> int:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        figure, axes = plt.subplots(figsize=(7.0, 4.0), dpi=DPI)
        labels = _draw_bars(axes, spec)
        kept = _drop_colliding_labels(figure, labels)
        plt.close(figure)
        return kept

    def test_should_keep_every_label_where_they_fit(self):
        assert self._surviving_labels(self._spec(2, 2)) == 4

    def test_should_drop_them_all_where_three_series_meet_six_corpora(self):
        assert self._surviving_labels(self._spec(3, 6)) == 0

    def test_should_still_draw_the_crowded_chart(self, tmp_path):
        assert render_chart(self._spec(3, 6), tmp_path).read_bytes().startswith(b"\x89PNG")


class TestComputeChart:
    def _spec(self, values, metric="cpu_seconds_per_doc") -> ComputeChartSpec:
        return ComputeChartSpec(
            metric=metric,
            series=tuple(f"run {index}" for index in range(len(values))),
            values=tuple(values),
        )

    def test_should_name_the_file_after_the_metric(self):
        assert self._spec([1.0, 2.0]).filename == "compute-cpu_seconds_per_doc.png"

    def test_should_take_its_title_from_the_metric(self):
        assert self._spec([1.0, 2.0]).title == "Compute per document"

    def test_should_say_what_the_axis_measures(self):
        assert self._spec([1.0, 2.0]).axis_label == "CPU-seconds per document"

    def test_should_use_the_title_it_was_given(self):
        spec = ComputeChartSpec(
            metric="latency_median", series=("a", "b"), values=(1.0, 2.0),
            title_override="What it waited",
        )
        assert spec.title == "What it waited"

    def test_should_draw_where_two_variants_recorded_it(self, tmp_path):
        path = render_compute_chart(self._spec([1.0, 2.0]), tmp_path)
        assert path is not None and path.read_bytes().startswith(b"\x89PNG")

    def test_should_draw_nothing_for_a_single_variant(self, tmp_path):
        assert render_compute_chart(self._spec([1.0, None]), tmp_path) is None
        assert not list(tmp_path.iterdir())

    def test_should_draw_nothing_where_none_recorded_it(self, tmp_path):
        assert render_compute_chart(self._spec([None, None]), tmp_path) is None

    def test_should_keep_a_variants_colour_where_another_is_absent(self, tmp_path):
        # The third variant has no figure, so the fourth keeps slot 4's colour rather
        # than sliding into slot 3: colour follows the variant, not the bar's rank.
        spec = self._spec([1.0, 2.0, None, 4.0])
        path = render_compute_chart(spec, tmp_path)
        assert path is not None

    def test_should_apply_the_prefix(self, tmp_path):
        path = render_compute_chart(self._spec([1.0, 2.0]), tmp_path, prefix="sha-")
        assert path is not None and path.name.startswith("sha-")


class TestLegendLayout:
    """Matplotlib fills a legend column by column; people read it row by row."""

    def _rows(self, count: int, columns: int = 4):
        width, order = _legend_layout(count, columns)
        n_rows = -(-count // width)
        rows: list = [[] for _ in range(n_rows)]
        for slot, item in enumerate(order):
            rows[slot % n_rows].append(item)
        return rows

    def test_should_read_left_to_right_on_one_row(self):
        assert self._rows(4) == [[0, 1, 2, 3]]

    def test_should_read_left_to_right_across_two_rows(self):
        assert self._rows(6) == [[0, 1, 2], [3, 4, 5]]

    def test_should_handle_a_row_that_does_not_fill(self):
        assert self._rows(5) == [[0, 1, 2], [3, 4]]

    def test_should_narrow_to_the_grid_it_will_draw(self):
        # Six entries asked to fill four columns become three columns of two; telling
        # matplotlib four would undo the reordering.
        assert _legend_layout(6, 4)[0] == 3

    def test_should_not_widen_beyond_what_was_asked(self):
        assert _legend_layout(8, 4)[0] == 4


class TestComputeChartCoverage:
    """A variant records this only if the run that made its predictions measured it."""

    def _spec(self, values):
        return ComputeChartSpec(
            metric="cpu_seconds_per_doc",
            series=tuple(f"run {index}" for index in range(len(values))),
            values=tuple(values),
        )

    def test_should_say_nothing_extra_where_every_variant_recorded_it(self):
        assert self._spec([1.0, 2.0]).caption == "CPU-seconds per document"

    def test_should_say_how_many_recorded_it_where_some_did_not(self):
        assert self._spec([1.0, 2.0, None]).caption == (
            "CPU-seconds per document — 2 of 3 variants recorded it"
        )

    def test_should_carry_that_into_the_alt_text(self):
        assert "2 of 3 variants recorded it" in self._spec([1.0, 2.0, None]).alt_text
