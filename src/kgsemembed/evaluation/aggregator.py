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
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd

from kgsemembed.pipeline.conditions import get_condition, get_conditions_for_group

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
_RESULT_COLUMNS = (*_REQUIRED_FIELDS, "ablation_group")

_SUMMARY_FLOAT_FIELDS = (
    "mean_f1",
    "std_f1",
    "mean_mrr",
    "mean_recall_at_1",
    "mean_recall_at_5",
    "mean_recall_at_10",
)
_SUMMARY_COLUMNS = ("condition_id", "strategy", "model_key", *_SUMMARY_FLOAT_FIELDS, "n_pairs")

_DATASET_IDS = ("D1", "D2", "D3", "D4", "D5")
_ENTITY_TYPE_DATASETS = ("D3", "D4")
_ENTITY_TYPE_KEYWORDS = (
    ("Classes", ("class",)),
    ("Predicates", ("predicate", "property", "relation")),
    ("Instances", ("instance",)),
)
_ABLATION_GROUPS = ("A", "B", "C", "D")
_PPAS_COMPARISONS = (("C5", "C14", "D5"), ("C10", "C15", "D1"))
_STATS_COLUMNS = (
    "Condition A",
    "Condition B",
    "n",
    "p-value",
    "Corrected sig.",
    "delta-F1",
)


def _empty_results_frame() -> pd.DataFrame:
    """Return an empty results DataFrame carrying the canonical schema."""
    return pd.DataFrame(columns=list(_RESULT_COLUMNS))


def _load_json(path: Path) -> Optional[dict]:
    """Parse a result file, logging a warning and returning ``None`` on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _LOGGER.warning("Skipping malformed result file %s: %s", path, exc)
        return None


def _build_record(payload: dict, path: Path) -> Optional[dict]:
    """
    Flatten a result payload into a single row, or skip it with a warning.

    Parameters
    ----------
    payload : dict
        Parsed result JSON payload.
    path : Path
        Source file, used only for diagnostic logging.

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
    try:
        record["ablation_group"] = get_condition(record["condition_id"]).ablation_group
    except KeyError:
        _LOGGER.warning("Skipping %s; unknown condition %r.", path, record["condition_id"])
        return None
    return record


def load_all_results(results_dir: str = _DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """
    Load every experiment result file under ``results_dir`` into one DataFrame.

    The directory is walked recursively for files matching ``*_results.json``.
    Malformed files, records missing a required field, and records for
    unregistered conditions are skipped with a warning; a repeated
    ``(condition_id, dataset_id, pair_name)`` combination keeps its first
    occurrence.  ``ablation_group`` is derived from ``EXPERIMENT_CONDITIONS``.

    Parameters
    ----------
    results_dir : str
        Root directory of previously generated result files.

    Returns
    -------
    pd.DataFrame
        One row per AlignmentPair per condition over :data:`_RESULT_COLUMNS`, or
        an empty DataFrame with that schema when no valid results are found.
    """
    root = Path(results_dir)
    if not root.is_dir():
        _LOGGER.warning("Results directory not found: %s", root)
        return _empty_results_frame()
    records: List[dict] = []
    seen: set = set()
    for path in sorted(root.rglob(_RESULT_GLOB)):
        payload = _load_json(path)
        if payload is None:
            continue
        record = _build_record(payload, path)
        if record is None:
            continue
        key = (record["condition_id"], record["dataset_id"], record["pair_name"])
        if key in seen:
            _LOGGER.warning("Duplicate result for %s; skipping %s.", key, path)
            continue
        seen.add(key)
        records.append(record)
    if not records:
        return _empty_results_frame()
    return pd.DataFrame.from_records(records, columns=list(_RESULT_COLUMNS))


def build_condition_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per condition across every AlignmentPair.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.

    Returns
    -------
    pd.DataFrame
        One row per condition over :data:`_SUMMARY_COLUMNS`, sorted by
        ``mean_f1`` descending with ``condition_id`` breaking ties, and every
        floating-point value rounded to four decimal places.
    """
    if df.empty:
        return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
    grouped = df.groupby(["condition_id", "strategy", "model_key"], sort=False)
    summary = grouped.agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", lambda scores: float(scores.std(ddof=0))),
        mean_mrr=("mrr", "mean"),
        mean_recall_at_1=("recall_at_1", "mean"),
        mean_recall_at_5=("recall_at_5", "mean"),
        mean_recall_at_10=("recall_at_10", "mean"),
        n_pairs=("pair_name", "count"),
    ).reset_index()
    summary = summary.sort_values(
        ["mean_f1", "condition_id"], ascending=[False, True]
    ).reset_index(drop=True)
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


def _best_conditions(df: pd.DataFrame) -> List[tuple]:
    """Return ``(dataset_id, condition_id, mean_f1)`` of the best condition per dataset."""
    breakdown = build_dataset_breakdown_table(df, "f1")
    best: List[tuple] = []
    for dataset_id in breakdown.columns:
        column = breakdown[dataset_id].dropna()
        if column.empty:
            continue
        winner = column.idxmax()
        best.append((dataset_id, winner, float(column.loc[winner])))
    return best


def _executive_summary_section(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Build the executive-summary section naming the best condition per dataset."""
    best = _best_conditions(df) if not df.empty else []
    if not best:
        return _section("Executive Summary", "_No results available._")
    meta = summary.set_index("condition_id")
    records = [
        {
            "Dataset": dataset_id,
            "Best Condition": condition_id,
            "Strategy": meta.loc[condition_id, "strategy"],
            "Model": meta.loc[condition_id, "model_key"],
            "Mean F1": value,
        }
        for dataset_id, condition_id, value in best
    ]
    return _section("Executive Summary", _render_dataframe(pd.DataFrame.from_records(records)))


def _condition_summary_section(summary: pd.DataFrame) -> str:
    """Build the section listing every condition sorted by mean F1."""
    if summary.empty:
        return _section("Condition Summary Table", "_No results available._")
    return _section("Condition Summary Table", _render_dataframe(summary))


def _per_dataset_block(breakdown: Optional[pd.DataFrame], dataset_id: str) -> str:
    """Build one per-dataset breakdown sub-table from the shared F1 pivot."""
    heading = f"## {dataset_id}"
    if breakdown is None or dataset_id not in breakdown.columns:
        return f"{heading}\n\n_No results available for {dataset_id}._"
    frame = breakdown[dataset_id].dropna().reset_index()
    if frame.empty:
        return f"{heading}\n\n_No results available for {dataset_id}._"
    frame.columns = ["condition_id", "mean_f1"]
    frame = frame.sort_values(
        ["mean_f1", "condition_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return f"{heading}\n\n{_render_dataframe(frame)}"


def _per_dataset_section(df: pd.DataFrame) -> str:
    """Build the per-dataset breakdown section reusing the shared F1 pivot."""
    breakdown = build_dataset_breakdown_table(df, "f1") if not df.empty else None
    blocks = [_per_dataset_block(breakdown, dataset_id) for dataset_id in _DATASET_IDS]
    return _section("Per-Dataset Breakdown", "\n\n".join(blocks))


def _classify_entity_type(pair_name: str) -> Optional[str]:
    """Infer the entity-type sub-task of a pair from its name, or ``None``."""
    lowered = pair_name.lower()
    for label, keywords in _ENTITY_TYPE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _entity_type_frame(subset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean F1 and pair counts per entity-type label."""
    order = [label for label, _ in _ENTITY_TYPE_KEYWORDS]
    grouped = subset.groupby("entity_type").agg(
        mean_f1=("f1", "mean"), n_pairs=("pair_name", "count")
    )
    grouped = grouped.reindex([label for label in order if label in grouped.index])
    grouped["mean_f1"] = grouped["mean_f1"].round(4)
    frame = grouped.reset_index()
    frame.columns = ["entity_type", "mean_f1", "n_pairs"]
    return frame


def _entity_type_block(df: pd.DataFrame, dataset_id: str) -> str:
    """Build one entity-type sub-table for a dataset, noting absent sub-tasks."""
    heading = f"## {dataset_id}"
    subset = df[df["dataset_id"] == dataset_id].copy()
    if not subset.empty:
        subset["entity_type"] = subset["pair_name"].map(_classify_entity_type)
        subset = subset[subset["entity_type"].notna()]
    if subset.empty:
        return f"{heading}\n\n_No entity-type sub-task data available for {dataset_id}._"
    return f"{heading}\n\n{_render_dataframe(_entity_type_frame(subset))}"


def _entity_type_section(df: pd.DataFrame) -> str:
    """Build the entity-type analysis section for D3 and D4."""
    blocks = [_entity_type_block(df, dataset_id) for dataset_id in _ENTITY_TYPE_DATASETS]
    return _section("Entity-Type Analysis", "\n\n".join(blocks))


def _ablation_group_block(summary: pd.DataFrame, group: str) -> str:
    """Build one ablation-group sub-table using registry-defined membership."""
    heading = f"## Group {group}"
    condition_ids = [condition.condition_id for condition in get_conditions_for_group(group)]
    block = summary[summary["condition_id"].isin(condition_ids)] if not summary.empty else summary
    if block.empty:
        return f"{heading}\n\n_No results available for Group {group}._"
    return f"{heading}\n\n{_render_dataframe(block)}"


def _ablation_group_section(summary: pd.DataFrame) -> str:
    """Build the ablation-group analysis section for groups A through D."""
    blocks = [_ablation_group_block(summary, group) for group in _ABLATION_GROUPS]
    return _section("Ablation Group Analysis", "\n\n".join(blocks))


def _stats_row(result: dict) -> dict:
    """Project a Wilcoxon comparison result onto the report's stats columns."""
    return {
        "Condition A": result["condition_a"],
        "Condition B": result["condition_b"],
        "n": result["n_pairs"],
        "p-value": round(float(result["p_value"]), 4),
        "Corrected sig.": bool(result.get("corrected_significant", False)),
        "delta-F1": round(float(result["delta_f1"]), 4),
    }


def _statistical_section(stats_results: List[dict]) -> str:
    """Build the statistical-significance section from Wilcoxon comparison results."""
    if not stats_results:
        return _section("Statistical Significance", "_No statistical comparison results available._")
    frame = pd.DataFrame.from_records(
        [_stats_row(result) for result in stats_results], columns=list(_STATS_COLUMNS)
    )
    return _section("Statistical Significance", _render_dataframe(frame))


def _mean_metric(df: pd.DataFrame, condition_id: str, dataset_id: str) -> Optional[float]:
    """Return a condition's mean F1 on a dataset, or ``None`` when absent."""
    subset = df[(df["condition_id"] == condition_id) & (df["dataset_id"] == dataset_id)]
    if subset.empty:
        return None
    return round(float(subset["f1"].mean()), 4)


def _ppas_row(df: pd.DataFrame, condition_a: str, condition_b: str, dataset_id: str) -> dict:
    """Build one PPAS comparison row from the loaded results."""
    mean_a = _mean_metric(df, condition_a, dataset_id)
    mean_b = _mean_metric(df, condition_b, dataset_id)
    delta = round(mean_b - mean_a, 4) if mean_a is not None and mean_b is not None else None
    return {
        "Comparison": f"{condition_a} vs {condition_b}",
        "Dataset": dataset_id,
        "Mean F1 (A)": mean_a,
        "Mean F1 (B)": mean_b,
        "Delta F1": delta,
    }


def _ppas_section(df: pd.DataFrame) -> str:
    """Build the PPAS ablation section from the loaded results."""
    rows = [_ppas_row(df, a, b, dataset_id) for a, b, dataset_id in _PPAS_COMPARISONS]
    return _section("PPAS Ablation", _render_dataframe(pd.DataFrame.from_records(rows)))


def generate_markdown_report(
    df: pd.DataFrame,
    stats_results: Optional[List[dict]] = None,
    output_path: str = "data/results/report.md",
) -> None:
    """
    Render the full Phase 2 experiment report to ``output_path``.

    Sections are emitted in a fixed order: executive summary, condition summary,
    per-dataset breakdown, entity-type analysis, ablation-group analysis, an
    optional statistical-significance section, and PPAS ablation.  The
    statistical section is included only when ``stats_results`` is not ``None``.
    Parent directories are created as needed and identical inputs always produce
    byte-for-byte identical output.

    Parameters
    ----------
    df : pd.DataFrame
        Results as returned by :func:`load_all_results`.
    stats_results : Optional[List[dict]]
        Wilcoxon comparison results to render; ``None`` omits the section.
    output_path : str
        Destination Markdown file.
    """
    summary = build_condition_summary_table(df)
    sections = [
        _executive_summary_section(df, summary),
        _condition_summary_section(summary),
        _per_dataset_section(df),
        _entity_type_section(df),
        _ablation_group_section(summary),
    ]
    if stats_results is not None:
        sections.append(_statistical_section(stats_results))
    sections.append(_ppas_section(df))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
