"""A comparison declared in a file: which variants, which rows, which charts.

A view over summaries that already exist. It decides nothing about scoring, so adding one
costs no run and moves no figure, and it names runs the way the predictions store does so
that a checked-in file carries no run id.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from benchmarks.report_cost import COMPUTE_METRICS
from benchmarks.report_grid import (
    SCOPE_ALL,
    SCOPE_GOLD,
    ChartConfig,
    ComputeChartConfig,
    Selection,
    SelectionError,
)

SCOPE_BOTH = "both"
SCOPES = (SCOPE_ALL, SCOPE_GOLD, SCOPE_BOTH)


@dataclass(frozen=True)
class VariantSpec:
    """One column. `current` is the run under test; the rest come from the store."""
    label: str
    tool: Optional[str] = None
    version: Optional[str] = None
    profile: str = "default"
    current: bool = False
    summary: Optional[str] = None
    primary: bool = False


@dataclass(frozen=True)
class RowSpec:
    field: str
    methods: Tuple[str, ...] = ()
    scope: str = SCOPE_BOTH
    # Asserted, not selected: a summary gives a field exactly one scoring type.
    expected_type: Optional[str] = None


@dataclass(frozen=True)
class ComparisonConfig:
    variants: Tuple[VariantSpec, ...]
    rows: Tuple[RowSpec, ...] = ()
    corpora: Optional[Tuple[str, ...]] = None
    charts: Tuple[ChartConfig, ...] = dataclass_field(default_factory=tuple)
    compute_charts: Tuple[ComputeChartConfig, ...] = dataclass_field(
        default_factory=tuple)
    name: str = "comparison"


def _require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise SelectionError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _unknown_keys(entry: dict, allowed: Sequence[str], what: str) -> None:
    unknown = [key for key in entry if key not in allowed]
    if unknown:
        raise SelectionError(
            f"{what} has unknown key(s) {', '.join(repr(k) for k in unknown)};"
            f" allowed: {', '.join(allowed)}"
        )


def _parse_variant(entry: Any, index: int) -> VariantSpec:
    entry = _require_mapping(entry, f"variants[{index}]")
    _unknown_keys(
        entry,
        ("label", "tool", "version", "profile", "current", "summary", "primary"),
        f"variants[{index}]",
    )
    current = bool(entry.get("current"))
    summary = entry.get("summary")
    tool, version = entry.get("tool"), entry.get("version")
    named = bool(tool and version)
    if sum([current, bool(summary), named]) != 1:
        raise SelectionError(
            f"variants[{index}] needs exactly one of: current: true, a summary path,"
            " or both tool and version"
        )
    profile = entry.get("profile", "default")
    label = entry.get("label") or _default_label(tool, version, profile, current, summary)
    return VariantSpec(
        label=label, tool=tool, version=version, profile=profile,
        current=current, summary=summary, primary=bool(entry.get("primary")),
    )


def _default_label(
    tool: Optional[str], version: Optional[str], profile: str,
    current: bool, summary: Optional[str],
) -> str:
    """What a column is called when the file does not say.

    Carries the profile, because two columns of one version differing only by profile is
    the comparison this exists for, and a label that drops it makes them indistinguishable.
    """
    if current:
        return f"this run ({profile})" if profile != "default" else "this run"
    if tool and version:
        return f"{tool} {version} ({profile})"
    return Path(str(summary)).parent.name or str(summary)


def _parse_row(entry: Any, index: int) -> RowSpec:
    entry = _require_mapping(entry, f"rows[{index}]")
    _unknown_keys(entry, ("field", "method", "methods", "scope", "type"), f"rows[{index}]")
    name = entry.get("field")
    if not name:
        raise SelectionError(f"rows[{index}] needs a field")
    if "method" in entry and "methods" in entry:
        raise SelectionError(f"row {name!r} sets both method and methods; use one")
    methods = entry.get("methods", [entry["method"]] if "method" in entry else [])
    if not isinstance(methods, list):
        raise SelectionError(f"row {name!r} methods must be a list")
    scope = entry.get("scope", SCOPE_BOTH)
    if scope not in SCOPES:
        raise SelectionError(
            f"row {name!r} has scope {scope!r}; expected one of {', '.join(SCOPES)}"
        )
    return RowSpec(
        field=name, methods=tuple(methods), scope=scope,
        expected_type=entry.get("type"),
    )


def _parse_chart(entry: Any, index: int):
    entry = _require_mapping(entry, f"charts[{index}]")
    if "compute" in entry:
        return _parse_compute_chart(entry, index)
    _unknown_keys(entry, ("row", "title", "corpora"), f"charts[{index}]")
    row = _require_mapping(entry.get("row", {}), f"charts[{index}].row")
    _unknown_keys(row, ("field", "method", "scope"), f"charts[{index}].row")
    if not row.get("field") or not row.get("method"):
        raise SelectionError(f"charts[{index}] needs a row with a field and a method")
    scope = row.get("scope", SCOPE_ALL)
    if scope not in (SCOPE_ALL, SCOPE_GOLD):
        raise SelectionError(
            f"charts[{index}] has scope {scope!r}; a chart draws one of"
            f" {SCOPE_ALL} or {SCOPE_GOLD}"
        )
    corpora = entry.get("corpora")
    return ChartConfig(
        field=row["field"], method=row["method"], scope=scope,
        title=entry.get("title"),
        corpora=tuple(corpora) if corpora else None,
    )


def _parse_compute_chart(entry: dict, index: int) -> ComputeChartConfig:
    _unknown_keys(entry, ("compute", "title"), f"charts[{index}]")
    metric = entry["compute"]
    if metric not in COMPUTE_METRICS:
        raise SelectionError(
            f"charts[{index}] asks for compute {metric!r};"
            + " known: " + ", ".join(repr(name) for name in sorted(COMPUTE_METRICS))
        )
    return ComputeChartConfig(metric=metric, title=entry.get("title"))


def parse_comparison(data: Any, name: str = "comparison") -> ComparisonConfig:
    data = _require_mapping(data, "comparison file")
    _unknown_keys(data, ("variants", "rows", "corpora", "charts"), "comparison file")
    variants = data.get("variants") or []
    if len(variants) < 2:
        raise SelectionError("a comparison needs at least two variants")
    if sum(bool(_require_mapping(v, "variant").get("primary")) for v in variants) > 1:
        raise SelectionError("only one variant can be the primary")
    corpora = data.get("corpora")
    parsed = [
        _parse_chart(entry, index)
        for index, entry in enumerate(data.get("charts") or [])
    ]
    return ComparisonConfig(
        name=name,
        variants=tuple(
            _parse_variant(entry, index) for index, entry in enumerate(variants)
        ),
        rows=tuple(
            _parse_row(entry, index) for index, entry in enumerate(data.get("rows") or [])
        ),
        corpora=tuple(corpora) if corpora else None,
        charts=tuple(
            chart for chart in parsed if isinstance(chart, ChartConfig)
        ),
        compute_charts=tuple(
            chart for chart in parsed if isinstance(chart, ComputeChartConfig)
        ),
    )


COMPARISON_DIR = Path("benchmarks/comparisons")


def load_comparison(name_or_path: str, base_dir: Path = COMPARISON_DIR) -> ComparisonConfig:
    """A name resolves under the comparisons directory; a path is taken as given."""
    path = Path(name_or_path)
    if not path.exists() and path.suffix not in (".yml", ".yaml"):
        path = base_dir / f"{name_or_path}.yml"
    if not path.exists():
        raise SelectionError(f"No comparison file at {path}")
    return parse_comparison(yaml.safe_load(path.read_text(encoding="utf-8")), name=path.stem)


def primary_index(config: ComparisonConfig) -> int:
    """Which column the deltas measure against. The last unless one says otherwise, so
    reordering for the sake of reading does not move the reference with it."""
    for index, variant in enumerate(config.variants):
        if variant.primary:
            return index
    return len(config.variants) - 1


def store_variants(config: ComparisonConfig) -> List[VariantSpec]:
    """The variants that have to come from the predictions store.

    A comparison may name runs that are not among `eval.yml`'s baselines -- that is most
    of why it exists -- so they have to be fetched and scored before it can be rendered.
    Fetching and scoring is not generating: nothing is predicted to satisfy a comparison.
    """
    return [
        variant for variant in config.variants
        if variant.tool and variant.version and not variant.current
    ]


def available_baselines(runs_dir: Path) -> List[str]:
    """`tool/version/profile/split` for every stored baseline that was scored."""
    root = runs_dir / "baselines"
    if not root.is_dir():
        return []
    return sorted(
        str(path.parent.relative_to(root))
        for path in root.glob("*/*/*/*/summary.json")
    )


def resolve_variants(
    config: ComparisonConfig,
    runs_dir: Path,
    split: str,
    current_dir: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    """Each variant's summary, in the order declared, with the primary last.

    A variant that cannot be resolved is an error naming it, and saying what is there
    instead: this reads what a run produced, and will not start one to fill a gap.
    """
    resolved: List[Tuple[str, Path]] = []
    missing: List[str] = []
    for variant in config.variants:
        path = _variant_summary(variant, runs_dir, split, current_dir)
        if path is None:
            missing.append(
                f"{variant.label} (declared `current: true`, but no run under test was"
                " given -- pass --current-run)"
            )
            continue
        if not path.exists():
            missing.append(f"{variant.label} ({path})")
            continue
        resolved.append((variant.label, path))
    if missing:
        raise SelectionError(
            "No summary for " + "; ".join(missing) + "."
            + _what_is_there(runs_dir, split)
        )
    return resolved


def _what_is_there(runs_dir: Path, split: str) -> str:
    """Point at what the runs directory does hold, so the error has a next step."""
    if not runs_dir.is_dir():
        return (
            f" There is no {runs_dir} at all -- a git worktree does not get one, since"
            " it is gitignored and lives in the checkout that produced it. Point --runs"
            " at that checkout, or run a benchmark here first."
        )
    found = available_baselines(runs_dir)
    if not found:
        return (
            f" Nothing under {runs_dir / 'baselines'} has been scored yet."
            " Nothing is generated to satisfy a comparison -- run a benchmark first."
        )
    for_split = [name for name in found if name.endswith(f"/{split}")]
    listed = "\n  ".join(for_split or found)
    note = "" if for_split else f" (none for split {split!r})"
    return (
        f" Nothing is generated to satisfy a comparison. Scored baselines{note}:"
        f"\n  {listed}"
    )


def _variant_summary(
    variant: VariantSpec,
    runs_dir: Path,
    split: str,
    current_dir: Optional[Path],
) -> Optional[Path]:
    if variant.summary:
        return Path(variant.summary)
    if variant.current:
        return (current_dir / "summary.json") if current_dir else None
    # The layout `_run_baseline` writes, so a name resolves against whichever run is at
    # hand rather than against the run the file was written for.
    return (
        runs_dir / "baselines" / str(variant.tool) / str(variant.version)
        / variant.profile / split / "summary.json"
    )


def selected_fields(config: ComparisonConfig) -> Optional[Tuple[str, ...]]:
    if not config.rows:
        return None
    seen: List[str] = []
    for row in config.rows:
        if row.field not in seen:
            seen.append(row.field)
    return tuple(seen)


def row_filter(
    config: ComparisonConfig,
) -> Optional[Dict[str, Tuple[Tuple[str, str], ...]]]:
    """Per field, the (method, scope) pairs the file asked for. None keeps everything.

    An empty method stands for every method the field carries, and an empty scope for
    whichever scopes the field earns -- what a row that names neither means.
    """
    if not config.rows:
        return None
    wanted: Dict[str, List[Tuple[str, str]]] = {}
    for row in config.rows:
        # `both` is the absence of a constraint rather than a demand for two rows:
        # whether a field earns its gold row depends on the gold, so requiring one would
        # make the common case fail.
        scopes = ("",) if row.scope == SCOPE_BOTH else (row.scope,)
        for method in row.methods or ("",):
            for scope in scopes:
                pair = (method, scope)
                if pair not in wanted.setdefault(row.field, []):
                    wanted[row.field].append(pair)
    return {field: tuple(pairs) for field, pairs in wanted.items()}


def expected_types(config: ComparisonConfig) -> Dict[str, str]:
    return {
        row.field: row.expected_type
        for row in config.rows if row.expected_type is not None
    }


def to_selection(config: ComparisonConfig) -> Selection:
    return Selection(
        fields=selected_fields(config),
        corpora=config.corpora,
        row_filter=row_filter(config),
        expected_types=expected_types(config) or None,
        chart_configs=config.charts,
        compute_charts=config.compute_charts,
    )
