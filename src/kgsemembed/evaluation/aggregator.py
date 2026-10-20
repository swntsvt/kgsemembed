"""Aggregate Phase 2 experiment results into a publication-ready Markdown report.

This module is a reporting layer built on top of the result files written by
:mod:`kgsemembed.pipeline.run_experiment` under the
``data/results/{condition_id}/{dataset_id}/{pair_name}_results.json`` convention.
It loads previously generated results, aggregates their metrics into summary and
per-dataset tables, and renders a deterministic Markdown report.

It never reruns experiments, performs statistical testing, produces plots, or
emits LaTeX; statistical comparison results computed elsewhere are rendered as
Markdown tables only when supplied.  Ablation-group membership is always derived
from :data:`kgsemembed.pipeline.conditions.EXPERIMENT_CONDITIONS` rather than
duplicated here.

Aggregates across datasets use the dataset column as their unit, as
:mod:`kgsemembed.evaluation.stats` does: D1, D2, D3, D4_schema, D4_instance and
D5, where D3 is the mean of its 21 pairs.  A mean over pairs would let D3
outweigh every other dataset 21 to 1.  Sanity-check conditions (C19) are kept
out of the summary, breakdown and group tables and reported on their own.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from kgsemembed.evaluation import paired_tests
from kgsemembed.evaluation.reference_counts import result_metric
from kgsemembed.evaluation.stats import (
    _TABLE_COLUMNS as _STATS_COLUMNS,
    _sanitise_warning,
    pair_outcome_cell,
    result_delta,
)
from kgsemembed.pipeline.conditions import (
    EXPERIMENT_CONDITIONS,
    SANITY_CHECK_GROUP,
    get_condition,
    get_conditions_for_group,
)
from kgsemembed.utils.errors import DataError

_LOGGER = logging.getLogger("kgsemembed.evaluation.aggregator")

_RESULT_GLOB = "*_results.json"
_DEFAULT_RESULTS_DIR = "data/results/"

_TOP_LEVEL_FIELDS = (
    "condition_id",
    "dataset_id",
    "pair_name",
    "strategy",
    "model_key",
    "model_id",
)
_METRIC_FIELDS = (
    "f1",
    "precision",
    "recall",
    "threshold",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
)
_COUNT_FIELDS = ("n_source_entities", "n_candidates_per_entity")
_REQUIRED_FIELDS = (*_TOP_LEVEL_FIELDS, *_METRIC_FIELDS, *_COUNT_FIELDS)
_HITS_AT_1 = "hits_at_1"
_RESULT_COLUMNS = (*_REQUIRED_FIELDS, _HITS_AT_1, "ablation_group")

_PER_ENTITY_TYPE_FIELD = "per_entity_type"
_ENTITY_TYPE_COLUMN = "entity_type"
_REF_COUNT_COLUMN = "n_refs"
_OVERALL_ENTITY_TYPE = "overall"
_BREAKDOWN_COLUMNS = (*_RESULT_COLUMNS, _ENTITY_TYPE_COLUMN, _REF_COUNT_COLUMN)

_ENTITY_TYPES = ("class", "predicate")
_MIXED_DATASET_IDS = ("D3", "D4_schema")
_MIN_ENTITY_TYPE_REFS = 3
_ENTITY_TYPE_GROUP_KEYS = ("condition_id", "strategy", "model_key")
_ENTITY_TYPE_KEY_LABELS = {
    "condition_id": "Condition",
    "strategy": "Strategy",
    "model_key": "Model",
}
_ENTITY_TYPE_TABLE_COLUMNS = (
    "Condition",
    "Strategy",
    "Model",
    "Class F1",
    "Class n_refs",
    "Predicate F1",
    "Predicate n_refs",
    "Class vs Predicate gap",
)
_ENTITY_TYPE_FLOAT_COLUMNS = ("Class F1", "Predicate F1", "Class vs Predicate gap")
_ENTITY_TYPE_COUNT_COLUMNS = ("Class n_refs", "Predicate n_refs")
_NO_ENTITY_TYPE_DATA = "_No per-entity-type data available._"
_FINDING_DATASET_ID = "D3"
_NO_FINDING_DATA = (
    f"No {_FINDING_DATASET_ID} pair reports a per-entity-type breakdown, so no "
    "aggregate finding is computed."
)
_ONE_BUCKET_ONLY = (
    f"No {_FINDING_DATASET_ID} condition reports both entity types, so no gap "
    "can be computed."
)

_SUMMARY_FLOAT_FIELDS = (
    "mean_mrr",
    "std_mrr",
    "mean_hits_at_1",
    "mean_f1",
    "std_f1",
    "mean_recall_at_1",
    "mean_recall_at_5",
    "mean_recall_at_10",
)
_SUMMARY_COLUMNS = (
    "condition_id",
    "strategy",
    "model_key",
    "ablation_group",
    *_SUMMARY_FLOAT_FIELDS,
    "n_columns",
)
_SUMMARY_KEYS = ["condition_id", "strategy", "model_key", "ablation_group"]
_COLUMN_METRICS = ["mrr", _HITS_AT_1, "f1", "recall_at_1", "recall_at_5", "recall_at_10"]
_LEAD_METRICS = ("mrr", _HITS_AT_1, "f1")
_LEAD_LABELS = {"mrr": "MRR", _HITS_AT_1: "Hits@1", "f1": "F1"}
_REGISTRY_ORDER = {c.condition_id: index for index, c in enumerate(EXPERIMENT_CONDITIONS)}
_SUMMARY_NOTE = (
    "_Means over the dataset columns each condition was run on: D1, D2, D3 (the "
    "mean of its 21 pairs), D4_schema, D4_instance and D5. Conditions cover "
    "different columns (`n_columns`), so rows are listed in registry order and "
    "not ranked; compare conditions within a column in the Per-Dataset "
    "Breakdown._"
)

_ABLATION_GROUPS = ("A", "B", "C", "D", "E")
_GROUP_REFERENCE_CONDITIONS: Dict[str, Tuple[str, ...]] = {"E": ("C10",)}

_PAIRED_TITLE = "Per-Dataset Paired Tests"
_PAIRED_HEADINGS = {"mrr": "MRR (primary)", _HITS_AT_1: "Hits@1 (primary)", "f1": "F1 (secondary)"}
_CONSISTENCY_TITLE = "Cross-Dataset Consistency (Wilcoxon)"
_CONSISTENCY_HEADINGS = {"mrr": "MRR", "f1": "F1 (secondary)"}
_CONSISTENCY_NOTE = (
    "_A consistency summary, not the primary test: a two-sided Wilcoxon "
    "signed-rank test over the dataset columns (D3 as one column), Bonferroni "
    "within each ablation group. With at most six columns the smallest "
    "attainable p is 0.03125, so only group E (alpha / 1) can reach "
    "significance. The per-dataset paired tests above are the primary "
    "analysis._"
)
_CONFOUNDED_STRATEGY_AND_MODEL = "Confounded (strategy V4 vs V4+V6, model M2 vs M3)"
_CONFOUNDED_MODEL = "Confounded (model M2 vs M3)"

_PPAS_NOT_ISOLATED = (
    "Not isolated — C5 can sample (V4, M2 budget); C14 cannot "
    "(ppas_effective false: M3 has no budget)"
)
_PPAS_UNVARIED = "No — V2+V8 does not invoke PPAS under either model"


@dataclass(frozen=True)
class _PpasComparison:
    """
    One row of the PPAS ablation table.

    Attributes
    ----------
    condition_a : str
        Condition rendered in the ``Mean F1 (A)`` column.
    condition_b : str
        Condition rendered in the ``Mean F1 (B)`` column.
    dataset_id : str
        Dataset the two conditions are compared on.
    ablation_type : str
        Whether the pair holds the embedding model fixed.
    ppas_variable : str
        Whether PPAS is genuinely the variable under test, which it is only
        when the compared strategies consult a token budget at all.
    """

    condition_a: str
    condition_b: str
    dataset_id: str
    ablation_type: str
    ppas_variable: str


_PPAS_COMPARISONS = (
    _PpasComparison("C5", "C14", "D5", _CONFOUNDED_STRATEGY_AND_MODEL, _PPAS_NOT_ISOLATED),
    _PpasComparison("C10", "C15", "D1", _CONFOUNDED_MODEL, _PPAS_UNVARIED),
)
_NO_WARNING_CELL = ""


@dataclass(frozen=True)
class _SanityCheck:
    """
    A condition that must reproduce a reference condition exactly.

    Attributes
    ----------
    condition_id : str
        The sanity-check condition, e.g. ``"C19"``.
    reference_id : str
        The condition it must reproduce, e.g. ``"C10"``.
    dataset_id : str
        Dataset the two are compared on.
    expectation : str
        Why the two must agree.
    """

    condition_id: str
    reference_id: str
    dataset_id: str
    expectation: str


_SANITY_CHECKS = (
    _SanityCheck(
        "C19", "C10", "D1",
        "Identical metrics: V2+V8 never consults a token budget",
    ),
)


def _result_columns(include_entity_type_breakdown: bool = False) -> List[str]:
    """Return the loaded schema, widened by two columns when rows are expanded."""
    if include_entity_type_breakdown:
        return list(_BREAKDOWN_COLUMNS)
    return list(_RESULT_COLUMNS)


def _empty_results_frame(include_entity_type_breakdown: bool = False) -> pd.DataFrame:
    """Return an empty results DataFrame carrying the canonical schema."""
    return pd.DataFrame(columns=_result_columns(include_entity_type_breakdown))


def _load_json(path: Path) -> Optional[dict]:
    """Parse a result file, logging a warning and returning ``None`` on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _LOGGER.warning("Skipping malformed result file %s: %s", path, exc)
        return None


def _hits_at_1(payload: dict, counts: Optional[dict]) -> Optional[float]:
    """Return a result's source-level Hits@1, derived for older results, or ``None``."""
    try:
        return result_metric(payload, _HITS_AT_1, counts)
    except (KeyError, TypeError, ValueError):
        return None


def _build_record(payload: dict, path: Path, counts: Optional[dict] = None) -> Optional[dict]:
    """
    Flatten a result payload into a single row, or skip it with a warning.

    Parameters
    ----------
    payload : dict
        Parsed result JSON payload.
    path : Path
        Source file, used only for diagnostic logging.
    counts : Optional[dict]
        Per-pair reference counts, used to derive Hits@1 of results written
        before it was recorded (#139); without them such a row has none.

    Returns
    -------
    Optional[dict]
        A flat mapping over :data:`_RESULT_COLUMNS`, or ``None`` when a required
        field is absent or the condition is unregistered.
    """
    record: dict = {}
    for field in _TOP_LEVEL_FIELDS:
        record[field] = payload.get(field)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for field in _METRIC_FIELDS:
            record[field] = metrics.get(field)
    for field in _COUNT_FIELDS:
        record[field] = payload.get(field)
    missing = [field for field in _REQUIRED_FIELDS if record.get(field) is None]
    if missing:
        _LOGGER.warning("Skipping %s; missing required fields: %s", path, missing)
        return None
    record[_HITS_AT_1] = _hits_at_1(payload, counts)
    try:
        record["ablation_group"] = get_condition(record["condition_id"]).ablation_group
    except KeyError:
        _LOGGER.warning("Skipping %s; unknown condition %r.", path, record["condition_id"])
        return None
    return record


def _is_complete_bucket(entity_type: str, metrics: object, record: dict) -> bool:
    """
    Report whether one entity-type bucket carries every metric a row needs.

    A hand-edited or truncated result file can hold a bucket that is not a
    mapping at all, or one missing a metric.  Either is skipped with a warning
    rather than aborting the load or emitting a row of silent gaps, matching how
    :func:`_build_record` treats an incomplete pair-level result.

    Parameters
    ----------
    entity_type : str
        Key of the bucket, used only for the warning message.
    metrics : object
        Value found under that key, which need not be a mapping.
    record : dict
        The pair-level row, used only to identify the result in the warning.

    Returns
    -------
    bool
        ``True`` when the bucket can be expanded into a row.
    """
    if isinstance(metrics, dict) and all(field in metrics for field in _METRIC_FIELDS):
        return True
    _LOGGER.warning(
        "Skipping %s breakdown of %s; incomplete entity-type metrics.",
        entity_type,
        (record["condition_id"], record["dataset_id"], record["pair_name"]),
    )
    return False


def _entity_type_records(payload: dict, record: dict) -> List[dict]:
    """
    Expand a result's per-entity-type metrics into one row per entity type.

    Result files written before the breakdown existed carry no
    ``per_entity_type`` field and yield no extra rows, so old and new files
    aggregate side by side.

    Parameters
    ----------
    payload : dict
        Parsed result JSON payload.
    record : dict
        The pair-level row already built from ``payload``.

    Returns
    -------
    List[dict]
        One row per reported entity type, each copying the pair-level row with
        its metrics replaced by that entity type's.
    """
    breakdown = payload.get(_PER_ENTITY_TYPE_FIELD)
    if not isinstance(breakdown, dict):
        return []
    rows: List[dict] = []
    for entity_type, metrics in breakdown.items():
        if not _is_complete_bucket(entity_type, metrics, record):
            continue
        row = dict(record)
        row.update({field: metrics[field] for field in _METRIC_FIELDS})
        row[_HITS_AT_1] = metrics.get(_HITS_AT_1)
        row[_ENTITY_TYPE_COLUMN] = entity_type
        row[_REF_COUNT_COLUMN] = metrics.get(_REF_COUNT_COLUMN)
        rows.append(row)
    return rows


def _collect_records(
    root: Path, include_entity_type_breakdown: bool, counts: Optional[dict] = None
) -> List[dict]:
    """
    Read every result file under ``root`` into flat rows.

    Parameters
    ----------
    root : Path
        Root directory walked recursively for ``*_results.json`` files.
    include_entity_type_breakdown : bool
        Append one row per entity type after each pair-level row.
    counts : Optional[dict]
        Per-pair reference counts for deriving older results' Hits@1.

    Returns
    -------
    List[dict]
        Rows in file order, with entity-type rows following their pair.
    """
    records: List[dict] = []
    seen: set = set()
    missing_hits = 0
    for path in sorted(root.rglob(_RESULT_GLOB)):
        payload = _load_json(path)
        if payload is None:
            continue
        record = _build_record(payload, path, counts)
        if record is None:
            continue
        key = (record["condition_id"], record["dataset_id"], record["pair_name"])
        if key in seen:
            _LOGGER.warning("Duplicate result for %s; skipping %s.", key, path)
            continue
        seen.add(key)
        missing_hits += record[_HITS_AT_1] is None
        record[_ENTITY_TYPE_COLUMN] = _OVERALL_ENTITY_TYPE
        records.append(record)
        if include_entity_type_breakdown:
            records.extend(_entity_type_records(payload, record))
    if missing_hits:
        _LOGGER.warning(
            "Hits@1 is neither recorded nor derivable for %d result(s); Hits@1 "
            "means leave them out, unlike MRR and F1.", missing_hits,
        )
    return records


def _known_pair_count(path: Path) -> Optional[Tuple[Tuple[str, str], int]]:
    """Return ``((dataset_id, pair_name), n_train_refs)``, or ``None`` if unrecorded."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    population = payload.get("population")
    known = population.get("n_train_refs") if isinstance(population, dict) else None
    if known is None:
        return None
    return (str(payload.get("dataset_id")), str(payload.get("pair_name"))), int(known)


def check_known_pair_counts(paths: Iterable[Path]) -> None:
    """
    Refuse results of one pair that were scored with different known pairs.

    The D1 population changed in #137 (Bio-ML validation mappings became
    known pairs) without a protocol version change, so ``n_train_refs`` is
    what tells results from before and after it apart.  Files without a
    population record are not checked.

    Parameters
    ----------
    paths : Iterable[Path]
        Result files to be aggregated together.

    Raises
    ------
    DataError
        If two files of the same pair record different ``n_train_refs``.
    """
    seen: Dict[Tuple[str, str], Dict[int, Path]] = {}
    for path in paths:
        entry = _known_pair_count(path)
        if entry is not None:
            seen.setdefault(entry[0], {}).setdefault(entry[1], path)
    for (dataset_id, pair_name), counts in sorted(seen.items()):
        if len(counts) > 1:
            examples = ", ".join(f"{count} in {path}" for count, path in sorted(counts.items()))
            raise DataError(f"{dataset_id}/{pair_name} results mix known-pair counts: {examples}.")


def load_all_results(
    results_dir: str = _DEFAULT_RESULTS_DIR,
    include_entity_type_breakdown: bool = False,
    counts: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load every experiment result file under ``results_dir`` into one DataFrame.

    The directory is walked recursively for files matching ``*_results.json``.
    Malformed files, records missing a required field, and records for
    unregistered conditions are skipped with a warning; a repeated
    ``(condition_id, dataset_id, pair_name)`` combination keeps its first
    occurrence.  ``ablation_group`` is derived from ``EXPERIMENT_CONDITIONS``.
    Results of one pair must agree on their known-pair count
    (:func:`check_known_pair_counts`).

    Parameters
    ----------
    results_dir : str
        Root directory of previously generated result files.
    include_entity_type_breakdown : bool
        Additionally emit one row per entity type for every mixed-type result
        that reports a ``per_entity_type`` block, tagging each row with
        ``entity_type`` and ``n_refs``.  Pair-level rows are retained unchanged
        and carry ``entity_type == "overall"``.
    counts : Optional[dict]
        Per-pair reference counts
        (:func:`kgsemembed.evaluation.reference_counts.load_reference_counts`).
        ``hits_at_1`` is read from each result, or derived from its Recall@1
        and these counts when it was not recorded; without them it is missing
        for such results.

    Returns
    -------
    pd.DataFrame
        One row per AlignmentPair per condition over :data:`_RESULT_COLUMNS`, or
        an empty DataFrame with that schema when no valid results are found.
        With ``include_entity_type_breakdown`` the schema widens to
        :data:`_BREAKDOWN_COLUMNS` and entity-type rows are interleaved.

    Raises
    ------
    DataError
        If results of one pair record different known-pair counts.
    """
    root = Path(results_dir)
    if not root.is_dir():
        _LOGGER.warning("Results directory not found: %s", root)
        return _empty_results_frame(include_entity_type_breakdown)
    check_known_pair_counts(sorted(root.rglob(_RESULT_GLOB)))
    records = _collect_records(root, include_entity_type_breakdown, counts)
    if not records:
        return _empty_results_frame(include_entity_type_breakdown)
    return pd.DataFrame.from_records(
        records, columns=_result_columns(include_entity_type_breakdown)
    )


def _dataset_column_means(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per condition and dataset column, averaging that column's pairs."""
    keys = [*_SUMMARY_KEYS, "dataset_id"]
    return df.groupby(keys, sort=False)[_COLUMN_METRICS].mean().reset_index()


def _in_registry_order(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort summary rows by the conditions' position in the registry."""
    order = summary["condition_id"].map(_REGISTRY_ORDER)
    return summary.assign(_order=order).sort_values("_order").drop(columns="_order")


def build_condition_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per condition across its dataset columns.

    Each dataset column (D3 included) contributes one value, the mean over its
    pairs, so D3's 21 pairs weigh as much as one D1 pair.  Conditions cover
    different columns, so the rows are listed in registry order, not ranked.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.

    Returns
    -------
    pd.DataFrame
        One row per condition over :data:`_SUMMARY_COLUMNS`, in registry
        order: means over the condition's dataset columns (MRR and Hits@1
        first, then F1 and Recall@k), the population standard deviation of MRR
        and F1 across them, rounded to four decimal places, and the number of
        columns.
    """
    if df.empty:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    columns = _dataset_column_means(df.astype({_HITS_AT_1: float}))
    grouped = columns.groupby(_SUMMARY_KEYS, sort=False)
    summary = grouped.agg(
        mean_mrr=("mrr", "mean"),
        std_mrr=("mrr", lambda scores: float(scores.std(ddof=0))),
        mean_hits_at_1=(_HITS_AT_1, "mean"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", lambda scores: float(scores.std(ddof=0))),
        mean_recall_at_1=("recall_at_1", "mean"),
        mean_recall_at_5=("recall_at_5", "mean"),
        mean_recall_at_10=("recall_at_10", "mean"),
        n_columns=("dataset_id", "count"),
    ).reset_index()
    summary = _in_registry_order(summary).reset_index(drop=True)
    summary[list(_SUMMARY_FLOAT_FIELDS)] = summary[list(_SUMMARY_FLOAT_FIELDS)].round(4)
    return summary[list(_SUMMARY_COLUMNS)]


def build_dataset_breakdown_table(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """
    Pivot the mean of ``metric`` with conditions as rows and datasets as columns.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.
    metric : str
        Metric column to aggregate, e.g. ``"f1"``.

    Returns
    -------
    pd.DataFrame
        A ``condition_id`` × ``dataset_id`` pivot of mean ``metric`` values,
        rows sorted by ``condition_id`` and columns sorted deterministically,
        with every value rounded to four decimal places.

    Raises
    ------
    ValueError
        If ``metric`` is not a column of ``df``.
    """
    if metric not in df.columns:
        raise ValueError(
            f"Unknown metric {metric!r}; available columns: {sorted(df.columns)}."
        )
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="condition_id"))
    pivot = df.pivot_table(
        index="condition_id", columns="dataset_id", values=metric, aggfunc="mean"
    )
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    pivot.columns.name = None
    return pivot.round(4)


def _format_value(value: object) -> str:
    """Render a single cell value with four-decimal floats and ``N/A`` gaps."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "N/A" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def _render_dataframe(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [_format_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section(title: str, body: str) -> str:
    """Compose a level-one Markdown section from a title and body."""
    return f"# {title}\n\n{body}"


def _lead_pivots(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return the condition × dataset pivot of each lead metric (MRR, Hits@1, F1)."""
    frame = df.astype({_HITS_AT_1: float})
    return {metric: build_dataset_breakdown_table(frame, metric) for metric in _LEAD_METRICS}


def _cell(pivot: pd.DataFrame, condition_id: str, dataset_id: str) -> Optional[float]:
    """Return one pivot value, or ``None`` when the pivot lacks it."""
    if dataset_id not in pivot.columns or condition_id not in pivot.index:
        return None
    value = pivot.loc[condition_id, dataset_id]
    return None if pd.isna(value) else float(value)


def _best_conditions(df: pd.DataFrame) -> List[tuple]:
    """Return ``(dataset_id, condition_id)`` of the best condition by MRR per dataset."""
    breakdown = build_dataset_breakdown_table(df, "mrr")
    best: List[tuple] = []
    for dataset_id in breakdown.columns:
        column = breakdown[dataset_id].dropna()
        if not column.empty:
            best.append((dataset_id, column.idxmax()))
    return best


def _executive_summary_section(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build the executive-summary section naming the best condition by MRR per dataset."""
    best = _best_conditions(df) if not df.empty else []
    if not best:
        return _section("Executive Summary", "_No results available._")
    meta, pivots = summary.set_index("condition_id"), _lead_pivots(df)
    records = [
        {
            "Dataset": dataset_id,
            "Best Condition (MRR)": condition_id,
            "Strategy": meta.loc[condition_id, "strategy"],
            "Model": meta.loc[condition_id, "model_key"],
            **{_LEAD_LABELS[m]: _cell(pivots[m], condition_id, dataset_id) for m in _LEAD_METRICS},
        }
        for dataset_id, condition_id in best
    ]
    return _section("Executive Summary", _render_dataframe(pd.DataFrame.from_records(records)))


def _condition_summary_section(summary: pd.DataFrame) -> str:
    """Build the section listing every condition's dataset-column means."""
    if summary.empty:
        return _section("Condition Summary Table", "_No results available._")
    return _section("Condition Summary Table", f"{_SUMMARY_NOTE}\n\n{_render_dataframe(summary)}")


def _per_dataset_block(pivots: Dict[str, pd.DataFrame], dataset_id: str) -> str:
    """Build one per-dataset sub-table ranked by MRR, with Hits@1 and F1 beside it."""
    heading = f"## {dataset_id}"
    mrr = pivots["mrr"]
    if dataset_id not in mrr.columns or mrr[dataset_id].dropna().empty:
        return f"{heading}\n\n_No results available for {dataset_id}._"
    conditions = mrr[dataset_id].dropna().index
    frame = pd.DataFrame({
        "condition_id": list(conditions),
        **{f"mean_{m}": [_cell(pivots[m], c, dataset_id) for c in conditions] for m in _LEAD_METRICS},
    })
    frame = frame.sort_values(
        ["mean_mrr", "condition_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return f"{heading}\n\n{_render_dataframe(frame)}"


def _present_dataset_ids(df: pd.DataFrame) -> List[str]:
    """
    List every dataset ID appearing in the results, in deterministic order.

    Sub-datasets such as ``D4_schema`` and ``D4_instance`` are reported under
    their own IDs rather than being folded into a parent dataset, so the
    breakdown never looks for an ID the results do not carry.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.

    Returns
    -------
    List[str]
        The distinct ``dataset_id`` values, sorted lexicographically.
    """
    return sorted(df["dataset_id"].dropna().unique(), key=str)


def _per_dataset_section(df: pd.DataFrame) -> str:
    """Build the per-dataset breakdown section, each column ranked by MRR."""
    dataset_ids = _present_dataset_ids(df) if not df.empty else []
    if not dataset_ids:
        return _section("Per-Dataset Breakdown", "_No results available._")
    pivots = _lead_pivots(df)
    blocks = [_per_dataset_block(pivots, dataset_id) for dataset_id in dataset_ids]
    return _section("Per-Dataset Breakdown", "\n\n".join(blocks))


def _mixed_entity_type_rows(df_expanded: pd.DataFrame) -> pd.DataFrame:
    """
    Select the class and predicate rows of the mixed-type datasets.

    A frame loaded without ``include_entity_type_breakdown`` carries no
    ``entity_type`` column at all and yields nothing, so a caller that never
    asked for the breakdown still renders a well-formed report.

    Parameters
    ----------
    df_expanded : pd.DataFrame
        Results as returned by :func:`load_all_results` with
        ``include_entity_type_breakdown=True``.

    Returns
    -------
    pd.DataFrame
        The D3 and D4_schema entity-type rows whose bucket reported at least
        :data:`_MIN_ENTITY_TYPE_REFS` test references.
    """
    if not set(_BREAKDOWN_COLUMNS).issubset(df_expanded.columns):
        return pd.DataFrame(columns=list(_BREAKDOWN_COLUMNS))
    rows = df_expanded.copy()
    rows[_REF_COUNT_COLUMN] = pd.to_numeric(rows[_REF_COUNT_COLUMN], errors="coerce")
    return rows[
        rows[_ENTITY_TYPE_COLUMN].isin(_ENTITY_TYPES)
        & rows["dataset_id"].isin(_MIXED_DATASET_IDS)
        & (rows[_REF_COUNT_COLUMN] >= _MIN_ENTITY_TYPE_REFS)
    ]


def _entity_type_means(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean F1 and total references per condition and entity type."""
    grouped = rows.groupby(
        [*_ENTITY_TYPE_GROUP_KEYS, _ENTITY_TYPE_COLUMN], sort=True
    ).agg(mean_f1=("f1", "mean"), total_refs=(_REF_COUNT_COLUMN, "sum"))
    return grouped.unstack(_ENTITY_TYPE_COLUMN)


def _entity_type_column(wide: pd.DataFrame, metric: str, entity_type: str) -> pd.Series:
    """Return one aggregated metric of one entity type, all-missing when absent."""
    if (metric, entity_type) in wide.columns:
        return wide[(metric, entity_type)]
    return pd.Series(float("nan"), index=wide.index)


def _entity_type_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Build the per-condition class-versus-predicate comparison table."""
    wide = _entity_type_means(rows)
    class_f1 = _entity_type_column(wide, "mean_f1", "class")
    predicate_f1 = _entity_type_column(wide, "mean_f1", "predicate")
    table = pd.DataFrame(
        {
            "Class F1": class_f1,
            "Class n_refs": _entity_type_column(wide, "total_refs", "class"),
            "Predicate F1": predicate_f1,
            "Predicate n_refs": _entity_type_column(wide, "total_refs", "predicate"),
            "Class vs Predicate gap": class_f1 - predicate_f1,
        }
    ).reset_index()
    return table.rename(columns=_ENTITY_TYPE_KEY_LABELS)


def _ref_count_cell(value: object) -> Optional[int]:
    """Render a summed reference count as a whole number, keeping gaps missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _order_entity_type_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sort by class F1 descending and round the table for rendering."""
    ordered = table.sort_values(
        ["Class F1", "Condition"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    float_columns = list(_ENTITY_TYPE_FLOAT_COLUMNS)
    ordered[float_columns] = ordered[float_columns].round(4)
    for column in _ENTITY_TYPE_COUNT_COLUMNS:
        ordered[column] = ordered[column].map(_ref_count_cell)
    return ordered[list(_ENTITY_TYPE_TABLE_COLUMNS)]


def _largest_divergence(rows: pd.DataFrame) -> Optional[tuple]:
    """
    Locate the single pair whose class F1 most exceeds its predicate F1.

    Parameters
    ----------
    rows : pd.DataFrame
        Entity-type rows as returned by :func:`_mixed_entity_type_rows`.

    Returns
    -------
    Optional[tuple]
        ``(condition_id, pair_name, gap)`` for the widest gap, or ``None`` when
        no pair reports both entity types.
    """
    pivot = rows.pivot_table(
        index=["condition_id", "pair_name"],
        columns=_ENTITY_TYPE_COLUMN,
        values="f1",
        aggfunc="mean",
    )
    if not set(_ENTITY_TYPES).issubset(pivot.columns):
        return None
    gaps = (pivot["class"] - pivot["predicate"]).dropna()
    if gaps.empty:
        return None
    widest = gaps.abs().idxmax()
    return widest[0], widest[1], float(gaps.loc[widest])


def _harder_entity_type_clause(gap: float) -> str:
    """Open the finding by naming the entity type that scores lower, if either does."""
    if gap == 0:
        return "Neither entity type is harder"
    harder = "Predicates" if gap > 0 else "Classes"
    return f"{harder} are the harder entity type"


def _divergence_sentence(rows: pd.DataFrame) -> str:
    """
    Cite the condition and pair whose two entity types diverge most.

    The widest gap is taken by magnitude rather than sign, so the pair cited is
    the strongest example whichever entity type leads it; quoting a signed
    maximum would print a negative "exceeds by" whenever predicates lead.

    Parameters
    ----------
    rows : pd.DataFrame
        Entity-type rows as returned by :func:`_mixed_entity_type_rows`.

    Returns
    -------
    str
        The closing sentence of the finding paragraph.
    """
    widest = _largest_divergence(rows)
    if widest is None:
        return "No pair reports both entity types, so no divergence is cited."
    condition_id, pair_name, gap = widest
    if gap == 0:
        return "No pair diverges; class and predicate F1 are equal throughout."
    leader, trailer = ("class", "predicate") if gap > 0 else ("predicate", "class")
    return (
        f"The widest divergence is {condition_id} on {pair_name}, where {leader} F1 "
        f"exceeds {trailer} F1 by {abs(gap):.4f}."
    )


def _count_phrase(count: int, noun: str) -> str:
    """Render a count with its noun in grammatical agreement."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _finding_averages(complete: pd.DataFrame) -> tuple:
    """Return the mean class F1, the mean predicate F1, and their difference."""
    mean_class = float(complete["Class F1"].mean())
    mean_predicate = float(complete["Predicate F1"].mean())
    return mean_class, mean_predicate, mean_class - mean_predicate


def _finding_sentences(rows: pd.DataFrame, table: pd.DataFrame) -> str:
    """
    State the harder entity type, the mean gap, and the widest divergence.

    Only conditions reporting both entity types are averaged.  Averaging class
    F1 over one set of conditions and predicate F1 over another would quote a
    gap that does not equal the difference of the two means printed beside it,
    and the gap is subtracted from those means so the three always agree.

    Parameters
    ----------
    rows : pd.DataFrame
        Entity-type rows already narrowed to :data:`_FINDING_DATASET_ID`.
    table : pd.DataFrame
        Per-condition summary built from ``rows``.

    Returns
    -------
    str
        The finding paragraph, or a note when no condition reports both types.
    """
    complete = table.dropna(subset=["Class F1", "Predicate F1"])
    if complete.empty:
        return _ONE_BUCKET_ONLY
    scoped = rows[rows["condition_id"].isin(set(complete["Condition"]))]
    mean_class, mean_predicate, gap = _finding_averages(complete)
    return (
        f"{_harder_entity_type_clause(gap)}: across the "
        f"{_count_phrase(len(complete), 'condition')} reporting both entity types "
        f"on {_FINDING_DATASET_ID}, mean class F1 is {mean_class:.4f} against mean "
        f"predicate F1 of {mean_predicate:.4f}. The mean class-versus-predicate gap "
        f"is {gap:.4f} F1 over "
        f"{_count_phrase(scoped['pair_name'].nunique(), f'{_FINDING_DATASET_ID} pair')}. "
        f"{_divergence_sentence(scoped)}"
    )


def _entity_type_finding(rows: pd.DataFrame) -> str:
    """
    Compose the data-driven finding paragraph beneath the summary table.

    Only :data:`_FINDING_DATASET_ID` rows are averaged.  D4_schema contributes a
    single pair, so weighting it equally with every Conference pair lets one
    alignment dictate which entity type the paragraph calls harder.  The summary
    table above still reports both datasets.

    Parameters
    ----------
    rows : pd.DataFrame
        Entity-type rows as returned by :func:`_mixed_entity_type_rows`.

    Returns
    -------
    str
        The finding paragraph, or a note when no qualifying result exists.
    """
    scoped = rows[rows["dataset_id"] == _FINDING_DATASET_ID]
    if scoped.empty:
        return _NO_FINDING_DATA
    return _finding_sentences(scoped, _entity_type_table(scoped))


def _entity_type_section(df_expanded: pd.DataFrame) -> str:
    """
    Build the class-versus-predicate analysis section for mixed-type datasets.

    The table covers every mixed-type dataset; the finding paragraph beneath it
    averages :data:`_FINDING_DATASET_ID` alone.  See :func:`_entity_type_finding`
    for why the two scopes differ.

    Parameters
    ----------
    df_expanded : pd.DataFrame
        Results as returned by :func:`load_all_results` with
        ``include_entity_type_breakdown=True``.

    Returns
    -------
    str
        The rendered Markdown section, carrying a placeholder note when no
        result reports a class or predicate breakdown.
    """
    rows = _mixed_entity_type_rows(df_expanded)
    if rows.empty:
        return _section("Entity-Type Analysis", _NO_ENTITY_TYPE_DATA)
    table = _order_entity_type_table(_entity_type_table(rows))
    body = f"{_render_dataframe(table)}\n\n{_entity_type_finding(rows)}"
    return _section("Entity-Type Analysis", body)


def _group_condition_ids(group: str) -> List[str]:
    """
    Return the conditions shown in an ablation group's sub-table.

    Group E holds only C20, whose sole comparison is against C10 in group B,
    so C10 is listed alongside it as a reference row to keep the contrast in
    one table.

    Parameters
    ----------
    group : str
        Ablation group identifier, e.g. ``"E"``.

    Returns
    -------
    List[str]
        Registry members of ``group`` followed by its reference conditions.
    """
    members = [condition.condition_id for condition in get_conditions_for_group(group)]
    return members + list(_GROUP_REFERENCE_CONDITIONS.get(group, ()))


def _ablation_group_block(summary: pd.DataFrame, group: str) -> str:
    """Build one ablation-group sub-table from its members and reference rows."""
    heading = f"## Group {group}"
    condition_ids = _group_condition_ids(group)
    block = summary[summary["condition_id"].isin(condition_ids)] if not summary.empty else summary
    if block.empty:
        return f"{heading}\n\n_No results available for Group {group}._"
    return f"{heading}\n\n{_render_dataframe(block)}{_reference_note(group)}"


def _reference_note(group: str) -> str:
    """
    Explain any reference rows a group's sub-table borrows from other groups.

    Parameters
    ----------
    group : str
        Ablation group identifier, e.g. ``"E"``.

    Returns
    -------
    str
        A Markdown note naming each reference condition and its own group,
        or an empty string when the group has no reference rows.
    """
    references = _GROUP_REFERENCE_CONDITIONS.get(group, ())
    if not references:
        return ""
    labels = ", ".join(f"{cid} (group {get_condition(cid).ablation_group})" for cid in references)
    return f"\n\n_{labels} shown for reference: the controlled comparison for Group {group}._"


def _ablation_group_section(summary: pd.DataFrame) -> str:
    """Build the ablation-group analysis section for groups A through E."""
    blocks = [_ablation_group_block(summary, group) for group in _ABLATION_GROUPS]
    return _section("Ablation Group Analysis", "\n\n".join(blocks))


def _effect_size_cell(result: dict) -> Optional[float]:
    """Return a comparison's rank-biserial effect size, or ``None`` when absent."""
    value = result.get("effect_size_r")
    return None if value is None else round(float(value), 4)


def _warning_cell(result: dict) -> str:
    """
    Render a Wilcoxon warning as one footnote cell.

    The cell is left blank when SciPy raised nothing, where the exported stats
    table prints a dash; escaping is shared so warning text renders identically
    in both outputs.

    Parameters
    ----------
    result : dict
        A Wilcoxon comparison result, which need not carry a warning.

    Returns
    -------
    str
        The escaped warning text, or an empty cell.
    """
    message = result.get("wilcoxon_warning")
    return _sanitise_warning(message) if message else _NO_WARNING_CELL


def _stats_row(result: dict) -> dict:
    """Project a Wilcoxon comparison result onto the report's stats columns."""
    p_value = result.get("p_value")
    return {
        "Condition A": result["condition_a"],
        "Condition B": result["condition_b"],
        "Metric": result.get("metric", "f1"),
        "n": result["n_units"],
        "pairs (W/T/L)": pair_outcome_cell(result),
        "p-value": None if p_value is None else round(float(p_value), 4),
        "min p": round(float(result["min_attainable_p"]), 4),
        "Corrected sig.": bool(result.get("corrected_significant", False)),
        "delta": round(result_delta(result), 4),
        "effect_size_r": _effect_size_cell(result),
        "Warning": _warning_cell(result),
    }


def _stats_table(results: List[dict]) -> str:
    """Render Wilcoxon results with the columns the exported stats table uses."""
    frame = pd.DataFrame.from_records(
        [_stats_row(result) for result in results], columns=list(_STATS_COLUMNS)
    )
    return _render_dataframe(frame)


def _statistical_section(stats_results: List[dict]) -> str:
    """
    Build the cross-dataset consistency section from Wilcoxon comparison results.

    The dataset-level Wilcoxon is a summary of whether a difference keeps its
    sign across dataset columns, not the primary test: with at most six
    columns no comparison outside group E can reach significance.  Results on
    MRR come first, then F1 (secondary); results without a ``metric`` are F1.
    The columns are the ones :func:`kgsemembed.evaluation.export_stats_table`
    writes, so both outputs expose the same statistical fields.

    Parameters
    ----------
    stats_results : List[dict]
        Wilcoxon comparison results; an empty list yields a placeholder note.

    Returns
    -------
    str
        The rendered Markdown section.
    """
    if not stats_results:
        return _section(_CONSISTENCY_TITLE, "_No statistical comparison results available._")
    blocks = [_CONSISTENCY_NOTE]
    for metric in ("mrr", "f1"):
        results = [r for r in stats_results if r.get("metric", "f1") == metric]
        if results:
            blocks.append(f"## {_CONSISTENCY_HEADINGS[metric]}\n\n{_stats_table(results)}")
    return _section(_CONSISTENCY_TITLE, "\n\n".join(blocks))


def _paired_method_note() -> str:
    """Describe the per-dataset paired tests with their settings."""
    defaults = paired_tests.TestSettings()
    return (
        f"_Per registered comparison and dataset column: B − A (positive when B "
        f"is better), {int(paired_tests.CONFIDENCE * 100)}% percentile bootstrap "
        f"interval over test sources resampled within each pair (D3's 21 pairs "
        f"are strata; its value is their mean), and a paired randomisation test "
        f"swapping each source's two outcomes; {defaults.n_replicates:,} "
        f"replicates each, seed {defaults.seed}. MRR and Hits@1 count test "
        f"reference sources; F1 adds unmatched sources' predictions and is "
        f"recomputed from TP, FP and FN. Holm adjustment within each ablation "
        f"group × column × metric at alpha {paired_tests.ALPHA}; intervals are "
        f"not adjusted. Ceiling: the most the metric can reach on the stored "
        f"top-20 lists (F1: its recall ceiling)._"
    )


def _interval_cell(row: dict) -> str:
    """Render a bootstrap interval as ``[low, high]``."""
    return f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"


def _paired_row(row: dict) -> dict:
    """Project one tested paired-test row onto the report columns."""
    return {
        "Group": row["group"],
        "Comparison": f"{row['condition_a']} vs {row['condition_b']}",
        "Column": row["dataset_id"],
        "A": row["value_a"],
        "B": row["value_b"],
        "Δ (B − A)": row["delta"],
        "95% CI": _interval_cell(row),
        "p": row["p_value"],
        "p (Holm)": row["p_holm"],
        "m": row["family_size"],
        "Sig.": "yes" if row["significant"] else "no",
        "Ceiling": row["ceiling"],
        "n": row["n_units"],
    }


def _paired_block(rows: List[dict], metric: str) -> str:
    """Render the tested rows of one metric as a sub-table."""
    tested = [_paired_row(row) for row in rows if row["tested"] and row["metric"] == metric]
    body = _render_dataframe(pd.DataFrame.from_records(tested)) if tested else "_Not tested._"
    return f"## {_PAIRED_HEADINGS[metric]}\n\n{body}"


def _untested_note(rows: List[dict]) -> str:
    """List each comparison and column that could not be tested, once."""
    reasons = {
        (row["condition_a"], row["condition_b"], row["dataset_id"]): row["reason"]
        for row in rows if not row["tested"]
    }
    lines = [f"- {a} vs {b} on {ds}: {reason}" for (a, b, ds), reason in reasons.items()]
    return "Not tested:\n\n" + "\n".join(lines) if lines else ""


def _paired_section(rows: List[dict]) -> str:
    """
    Build the per-dataset paired-test section, the report's primary analysis.

    Parameters
    ----------
    rows : List[dict]
        Output of :func:`kgsemembed.evaluation.paired_tests.run_paired_tests`.

    Returns
    -------
    str
        Method note, one sub-table per metric (MRR, Hits@1, then F1 as
        secondary) and a list of comparisons that could not be tested.
    """
    blocks = [_paired_method_note()]
    blocks += [_paired_block(rows, metric) for metric in paired_tests.METRICS]
    blocks.append(_untested_note(rows))
    return _section(_PAIRED_TITLE, "\n\n".join(block for block in blocks if block))


def _mean_metric(
    df: pd.DataFrame, condition_id: str, dataset_id: str, metric: str = "f1"
) -> Optional[float]:
    """Return a condition's mean ``metric`` on a dataset, or ``None`` when absent."""
    subset = df[(df["condition_id"] == condition_id) & (df["dataset_id"] == dataset_id)]
    if subset.empty:
        return None
    return round(float(subset[metric].mean()), 4)


def _ppas_row(df: pd.DataFrame, comparison: _PpasComparison) -> dict:
    """
    Build one PPAS comparison row from the loaded results.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.
    comparison : _PpasComparison
        The comparison definition to render.

    Returns
    -------
    dict
        A row over the PPAS ablation table's columns.
    """
    mean_a = _mean_metric(df, comparison.condition_a, comparison.dataset_id)
    mean_b = _mean_metric(df, comparison.condition_b, comparison.dataset_id)
    delta = round(mean_b - mean_a, 4) if mean_a is not None and mean_b is not None else None
    return {
        "Comparison": f"{comparison.condition_a} vs {comparison.condition_b}",
        "Dataset": comparison.dataset_id,
        "Ablation Type": comparison.ablation_type,
        "PPAS Variable?": comparison.ppas_variable,
        "Mean F1 (A)": mean_a,
        "Mean F1 (B)": mean_b,
        "Delta F1": delta,
    }


def _ppas_section(df: pd.DataFrame) -> str:
    """Build the PPAS ablation section from the loaded results."""
    rows = [_ppas_row(df, comparison) for comparison in _PPAS_COMPARISONS]
    return _section("PPAS Ablation", _render_dataframe(pd.DataFrame.from_records(rows)))


def _pair_metrics(df: pd.DataFrame, condition_id: str, dataset_id: str) -> pd.DataFrame:
    """Return a condition's metrics on a dataset, indexed and sorted by pair name."""
    subset = df[(df["condition_id"] == condition_id) & (df["dataset_id"] == dataset_id)]
    return subset.set_index("pair_name")[list(_METRIC_FIELDS)].sort_index()


def _reproduction_cell(df: pd.DataFrame, check: _SanityCheck) -> str:
    """
    State whether a sanity-check condition reproduces its reference exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Pair-level results as returned by :func:`load_all_results`.
    check : _SanityCheck
        The check to evaluate.

    Returns
    -------
    str
        ``"Yes"`` when both cover the same pairs with equal metrics, ``"No"``
        otherwise, and ``"N/A"`` when either has no result.
    """
    ours = _pair_metrics(df, check.condition_id, check.dataset_id)
    reference = _pair_metrics(df, check.reference_id, check.dataset_id)
    if ours.empty or reference.empty:
        return "N/A"
    same = ours.index.equals(reference.index) and ours.astype(float).equals(reference.astype(float))
    return "Yes" if same else "No"


def _sanity_row(df: pd.DataFrame, check: _SanityCheck) -> dict:
    """Build one sanity-check row from the loaded results."""
    a, b, ds = check.condition_id, check.reference_id, check.dataset_id
    return {
        "Check": f"{a} vs {b}",
        "Dataset": ds,
        "Expected": check.expectation,
        "Mean MRR (A)": _mean_metric(df, a, ds, "mrr"),
        "Mean MRR (B)": _mean_metric(df, b, ds, "mrr"),
        "Mean F1 (A)": _mean_metric(df, a, ds),
        "Mean F1 (B)": _mean_metric(df, b, ds),
        "Reproduces?": _reproduction_cell(df, check),
    }


def _sanity_section(df: pd.DataFrame) -> str:
    """Build the sanity-check section, kept apart from the result tables."""
    rows = [_sanity_row(df, check) for check in _SANITY_CHECKS]
    return _section("Sanity Checks", _render_dataframe(pd.DataFrame.from_records(rows)))


def _without_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sanity-check conditions, which are reported only in their own section."""
    if df.empty:
        return df
    return df[df["ablation_group"] != SANITY_CHECK_GROUP]


def _pair_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per AlignmentPair, dropping any interleaved entity-type rows.

    Every section other than the entity-type analysis aggregates one row per
    pair, so a frame loaded with the breakdown must be narrowed first or each
    mixed-type pair would be counted once more per entity type it reports.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`, loaded with or without
        ``include_entity_type_breakdown``.

    Returns
    -------
    pd.DataFrame
        The pair-level rows, unchanged when no breakdown was loaded.
    """
    if _ENTITY_TYPE_COLUMN not in df.columns:
        return df
    return df[df[_ENTITY_TYPE_COLUMN] == _OVERALL_ENTITY_TYPE]


def generate_markdown_report(
    df: pd.DataFrame,
    stats_results: Optional[List[dict]] = None,
    output_path: str = "data/results/report.md",
    paired_results: Optional[List[dict]] = None,
) -> None:
    """
    Render the full Phase 2 experiment report to ``output_path``.

    Sections are emitted in a fixed order: executive summary, condition summary,
    per-dataset breakdown, entity-type analysis, ablation-group analysis, the
    optional per-dataset paired tests (primary analysis) and cross-dataset
    Wilcoxon consistency summary, PPAS ablation and sanity checks.  The two
    statistical sections are included only when ``paired_results`` and
    ``stats_results``, respectively, are not ``None``.  Ranking tables lead
    with MRR, then Hits@1, then F1.  Sanity-check conditions appear only in
    the last section.
    Parent directories are created as needed and identical inputs always produce
    byte-for-byte identical output.

    The entity-type analysis is populated only when ``df`` was loaded with
    ``include_entity_type_breakdown=True``; otherwise that one section carries a
    placeholder note and every other section is unaffected.  The breakdown rows
    are consumed here alone, so no other section double-counts a mixed pair.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.
    stats_results : Optional[List[dict]]
        Wilcoxon comparison results to render; ``None`` omits the section.
    output_path : str
        Destination Markdown file.
    paired_results : Optional[List[dict]]
        Per-dataset paired-test rows to render; ``None`` omits the section.
    """
    pairs = _pair_level_frame(df)
    main = _without_sanity_checks(pairs)
    summary = build_condition_summary_table(main)
    sections = [
        _executive_summary_section(main, summary),
        _condition_summary_section(summary),
        _per_dataset_section(main),
        _entity_type_section(_without_sanity_checks(df)),
        _ablation_group_section(summary),
    ]
    if paired_results is not None:
        sections.append(_paired_section(paired_results))
    if stats_results is not None:
        sections.append(_statistical_section(stats_results))
    sections.append(_ppas_section(pairs))
    sections.append(_sanity_section(pairs))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
