"""Command-line entry point for the Section 4.1 results tables.

Reads every result file and prints a complete LaTeX ``table*`` of one mean
metric per condition per dataset.  The paper's primary metrics are re-ranking
MRR (the default, ``Table~\\ref{tab:main_results}``) and source-level Hits@1
(``--metric hits_at_1``); F1 (``--metric f1``) and reference-level Recall@1
(``--metric recall_at_1``) are secondary.  Units and ceilings are defined once
in ``kgsemembed.evaluation.metrics``.

Each cell is the mean metric over the pairs a condition was run on for that
dataset, so D3 cells average the 21 Conference pairs.  Conditions follow the
paper order and a rule separates consecutive ablation groups.  The best value
in each dataset column is set in bold; values that tie at four decimal places
are all bolded.  The final column averages D1-D4 (both D4 columns) and is shown
only for conditions run on all of them, so every mean it reports covers the
same datasets and the column can be compared row against row.  D5 is left out.

The table is wrapped in ``threeparttable`` so the Mean note sits below the
table at the table's width instead of stretching the last column; the paper's
preamble must load ``\\usepackage{threeparttable}``.

Below the conditions, the ranking tables add the top-20 candidate ceiling of
their metric per column (``scripts/metric_ceilings.py``, read from
``--ceilings``).  The F1 table instead adds a micro-averaged D3 column (TP, FP
and FN pooled over the 21 pairs) beside the macro mean, the F1 recall ceiling,
and the validation and test reference counts per column, so the instability
of F1 tuned on few validation references stays visible.  Results stored
before Hits@1 was recorded get it from their Recall@1 and the counts file
(``kgsemembed.evaluation.reference_counts``).  ``--baseline_dir``
adds the stage-1 n-gram baseline B0 (``kgsemembed.pipeline.ngram_baseline``)
as the first row, above a rule; it ranks the same stored candidates for the
same populations, so every row is directly comparable with it.  B0 takes part
in the bold best-per-column and the Mean column.  Without ``--baseline_dir``
the F1 table is unchanged.

The sanity check C19 (identical to C10 by design) is not a result and is left
out; ``report.md`` checks it.  C12 is retired (#142).

Strategy, model and ablation group come from the condition registry, so the
table cannot drift from the conditions that were actually run.  Results are
read through ``kgsemembed.evaluation.load_all_results``, so malformed files,
unregistered conditions and duplicate results are skipped with a warning
exactly as in the generated report.

Usage
-----
python scripts/generate_results_table.py \\
    --results_dir data/results/ --baseline_dir data/results_ngram_baseline/ \\
    --metric mrr
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kgsemembed.evaluation import load_all_results
from kgsemembed.evaluation.aggregator import check_known_pair_counts
from kgsemembed.evaluation.reference_counts import (
    DEFAULT_COUNTS_PATH,
    PairCounts,
    PairKey,
    load_reference_counts,
    result_metric,
)
from kgsemembed.pipeline.conditions import get_condition

CellMeans = Dict[Tuple[str, str], float]
Counts = Dict[PairKey, PairCounts]

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_results_table")

_CONDITION_ORDER = (
    "C1", "C2", "C17",
    "C9", "C3", "C10", "C18",
    "C20",
    "C4", "C5", "C6", "C7", "C8", "C11", "C13", "C16",
    "C14", "C15",
)
_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
_COLUMN_HEADS = (
    "D1", "D2", "D3", r"D4$_{\mathrm{sc}}$", r"D4$_{\mathrm{in}}$", "D5"
)
_MEAN_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance")
_D3_MICRO = "D3_micro"
_F1_DATASETS = ("D1", "D2", "D3", _D3_MICRO, "D4_schema", "D4_instance", "D5")
_F1_COLUMN_HEADS = (
    "D1", "D2", "D3", r"D3$_{\mu}$", r"D4$_{\mathrm{sc}}$", r"D4$_{\mathrm{in}}$", "D5"
)
_D3_PAIR_COUNT = 21
_MODEL_MARKERS = {"M4": r"M4$^\ddagger$"}
_MISSING = "---"
_CONDITION_NOTES = (
    r"C20 (V2, M2) is the controlled baseline for C10 (V2+V8, M2): "
    r"the two differ only by the V8 component, and C20 forms its own "
    r"group (E), placed after group B so it sits beside C10. "
    r"$\ddagger$: M4 (BioLORD) evaluated on D1 and D2 only."
)
_CAPTION = (
    r"\caption{Mean test F1 per condition per dataset (secondary matching "
    r"metric: thresholded top-1 predictions, threshold tuned on validation). "
    r"D3 values are means over 21 pairs; D3$_{\mu}$ pools TP, FP and FN over "
    r"them. Below the conditions: the F1 recall ceiling of the top-20 lists "
    r"with one prediction per source, and the validation and test reference "
    r"counts (D3: range over pairs; D3$_{\mu}$: total), on which F1's "
    r"threshold and value rest. "
    r"Dashes in dataset columns indicate conditions not run on that dataset. "
    r"Bold: best result per dataset column. "
    + _CONDITION_NOTES + "}"
)
_BASELINE_ID = "B0"
_BASELINE_LEADING = [_BASELINE_ID, "n-gram", _MISSING, _MISSING]
_BASELINE_CAPTION = (
    r" B0: stage-1 character 3-gram cosine order of the same top-20 candidate "
    r"lists every condition re-ranks, evaluated under the same populations, "
    r"validation-tuned threshold and metric definitions; it is not an "
    r"external matching system.}"
)
_METRICS = ("mrr", "hits_at_1", "f1", "recall_at_1")
_RANKING_NAMES = {
    "mrr": "re-ranking MRR", "hits_at_1": "Hits@1", "recall_at_1": "Recall@1"
}
_RANKING_UNITS = {
    "mrr": (
        r"primary metric; per test reference source, reciprocal rank of its "
        r"best-ranked gold target among the same stored top-20 candidates, 0 "
        r"when none is listed"
    ),
    "hits_at_1": (
        r"primary metric; fraction of test reference sources whose top-ranked "
        r"candidate is a gold target, a source with several gold targets "
        r"counted once"
    ),
    "recall_at_1": (
        r"secondary; fraction of test reference correspondences whose target "
        r"is ranked first, so a source with $n$ gold targets counts $n$ times "
        r"and at most one of them can be recovered"
    ),
}
_CEILING_SOURCES = {
    "mrr": "ranking_ceiling", "hits_at_1": "ranking_ceiling",
    "recall_at_1": "candidate_recall_at_20", "f1": "f1_recall_ceiling",
}
_CEILING_LABELS = {
    "ranking_ceiling": "Top-20 ceiling",
    "candidate_recall_at_20": "Recall@20 ceiling",
    "f1_recall_ceiling": "Recall ceiling",
}
_RANKING_CAPTION = (
    r"\caption{{Mean test {name} per condition per dataset ({unit}). D3 "
    r"values are means over 21 pairs. Ceiling row: the most the metric can "
    r"reach on the stored top-20 candidate lists. Dashes in dataset "
    r"columns indicate conditions not run on that dataset. Bold: best result "
    r"per dataset column. " + _CONDITION_NOTES + "}}"
)
_MEAN_NOTE = (
    r"\item[*] Mean over D1--D4 (D4 schema and instance), shown only for "
    r"conditions run on all of them; excludes D5 (top-20 ceiling 26.67\% on every metric)."
)


def _read_payload(path: Path) -> dict:
    """Parse one result file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_payloads(results_dir: str) -> List[dict]:
    """Read the result files ``load_all_results`` accepts, at their canonical paths."""
    frame = load_all_results(results_dir)
    if frame.empty:
        return []
    keys = zip(frame["condition_id"], frame["dataset_id"], frame["pair_name"])
    root = Path(results_dir)
    return [_read_payload(root / c / d / f"{p}_results.json") for c, d, p in keys]


def _baseline_payloads(baseline_dir: str) -> List[dict]:
    """Read every B0 result file, warning when there is none."""
    paths = sorted(Path(baseline_dir).glob(f"{_BASELINE_ID}/*/*_results.json"))
    if not paths:
        _LOGGER.warning("No B0 results under %s; the B0 row is omitted.", baseline_dir)
    return [_read_payload(path) for path in paths]


def _means_from_payloads(payloads: List[dict], metric: str, counts: Optional[Counts]) -> CellMeans:
    """Average a metric per condition and dataset over the given results."""
    rows = [
        {"condition_id": p["condition_id"], "dataset_id": p["dataset_id"],
         metric: result_metric(p, metric, counts)}
        for p in payloads
    ]
    means = _cell_means(pd.DataFrame(rows), metric)
    if metric == "f1":
        means.update(d3_micro_f1(payloads))
    return means


def _cell_means(frame: pd.DataFrame, metric: str) -> CellMeans:
    """Average one metric per condition and dataset, warning on partial D3."""
    if frame.empty:
        return {}
    cells = frame.groupby(["condition_id", "dataset_id"], as_index=False).agg(
        mean=(metric, "mean"), n_pairs=(metric, "size")
    )
    _warn_on_partial_d3(cells)
    keys = zip(cells["condition_id"], cells["dataset_id"])
    return {(str(c), str(d)): float(value) for (c, d), value in zip(keys, cells["mean"])}


def _pooled_f1(tp: int, fp: int, fn: int) -> float:
    """F1 of pooled counts, ``0.0`` when undefined."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def d3_micro_f1(payloads: List[dict]) -> CellMeans:
    """
    Micro-average D3 F1 per condition by pooling TP, FP and FN over its pairs.

    Parameters
    ----------
    payloads : List[dict]
        Parsed result files; non-D3 results are ignored.

    Returns
    -------
    CellMeans
        Pooled F1 keyed by ``(condition_id, "D3_micro")``.
    """
    pooled: Dict[str, List[int]] = {}
    for payload in payloads:
        if payload["dataset_id"] == "D3":
            totals = pooled.setdefault(payload["condition_id"], [0, 0, 0])
            for index, key in enumerate(("tp", "fp", "fn")):
                totals[index] += int(payload["counts"][key])
    return {(condition_id, _D3_MICRO): _pooled_f1(*totals) for condition_id, totals in pooled.items()}


def load_cell_means(
    results_dir: str, metric: str = "mrr", counts: Optional[Counts] = None
) -> CellMeans:
    """
    Average a pair-level metric for every condition and dataset with results.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.
    metric : str
        Metric, one of ``mrr``, ``hits_at_1``, ``f1``, ``recall_at_1``.
        ``f1`` also yields the pooled D3 cell under ``"D3_micro"``.
    counts : Optional[Counts]
        Per-pair reference counts, used to derive Hits@1 of results stored
        before it was recorded.

    Returns
    -------
    CellMeans
        Mean metric keyed by ``(condition_id, dataset_id)``.
    """
    return _means_from_payloads(_condition_payloads(results_dir), metric, counts)


def load_baseline_means(
    baseline_dir: str, metric: str = "mrr", counts: Optional[Counts] = None
) -> CellMeans:
    """
    Average a pair-level metric of the B0 n-gram baseline per dataset.

    B0 is not a registered condition, so its files are read directly rather
    than through ``load_all_results``.

    Parameters
    ----------
    baseline_dir : str
        Root holding ``B0/{dataset_id}/{pair}_results.json``.
    metric : str
        Metric, as for :func:`load_cell_means`.
    counts : Optional[Counts]
        Per-pair reference counts, as for :func:`load_cell_means`.

    Returns
    -------
    CellMeans
        Mean metric keyed by ``("B0", dataset_id)``; empty when none exist.
    """
    return _means_from_payloads(_baseline_payloads(baseline_dir), metric, counts)


def _warn_on_partial_d3(cells: pd.DataFrame) -> None:
    """Warn for any D3 cell not averaged over all Conference pairs."""
    partial = cells[(cells["dataset_id"] == "D3") & (cells["n_pairs"] != _D3_PAIR_COUNT)]
    for condition_id, count in zip(partial["condition_id"], partial["n_pairs"]):
        _LOGGER.warning(
            "%s/D3 has %d pairs, not %d; the caption will be inaccurate.",
            condition_id, count, _D3_PAIR_COUNT,
        )


def _format_f1(value: Optional[float]) -> str:
    """Render an F1 value to four decimal places, or a dash when absent."""
    return _MISSING if value is None else f"{value:.4f}"


def _columns(metric: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return the dataset columns and their headings for a metric's table."""
    if metric == "f1":
        return _F1_DATASETS, _F1_COLUMN_HEADS
    return _DATASETS, _COLUMN_HEADS


def best_per_dataset(means: CellMeans) -> Dict[str, str]:
    """
    Find the best formatted value in each dataset column.

    Parameters
    ----------
    means : CellMeans
        Mean metric keyed by ``(condition_id, dataset_id)``.

    Returns
    -------
    Dict[str, str]
        The best value per dataset as printed, so ties at four decimal places
        compare equal.  Datasets with no results are absent.
    """
    best: Dict[str, float] = {}
    for (condition_id, dataset_id), value in means.items():
        listed = condition_id in _CONDITION_ORDER or condition_id == _BASELINE_ID
        if listed and dataset_id in _F1_DATASETS:
            best[dataset_id] = max(value, best.get(dataset_id, value))
    return {dataset_id: _format_f1(value) for dataset_id, value in best.items()}


def _mean_over_d1_to_d4(values: Dict[str, Optional[float]]) -> Optional[float]:
    """Average D1-D4, or ``None`` unless the condition has every one of them."""
    present = [
        value for dataset_id in _MEAN_DATASETS
        if (value := values.get(dataset_id)) is not None
    ]
    if len(present) < len(_MEAN_DATASETS):
        return None
    return sum(present) / len(present)


def _leading_cells(condition_id: str) -> List[str]:
    """Return the condition, strategy, model and group cells of a row."""
    if condition_id == _BASELINE_ID:
        return list(_BASELINE_LEADING)
    condition = get_condition(condition_id)
    model = _MODEL_MARKERS.get(condition.model_key, condition.model_key)
    return [condition_id, condition.strategy_name, model, condition.ablation_group]


def _latex_row(
    condition_id: str, means: CellMeans, best: Dict[str, str], columns: Tuple[str, ...]
) -> str:
    """
    Render one condition as a LaTeX table row.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    means : CellMeans
        Mean metric keyed by ``(condition_id, dataset_id)``.
    best : Dict[str, str]
        Best formatted value per dataset, from :func:`best_per_dataset`.
    columns : Tuple[str, ...]
        Dataset columns of the table, from :func:`_columns`.

    Returns
    -------
    str
        ``condition & strategy & model & group & value ... & mean \\\\``.
    """
    values = {ds: means.get((condition_id, ds)) for ds in columns}
    cells = [_format_f1(values[ds]) for ds in columns]
    cells = [
        rf"\textbf{{{cell}}}" if cell == best.get(ds) else cell
        for ds, cell in zip(columns, cells)
    ]
    leading = _leading_cells(condition_id)
    trailing = [_format_f1(_mean_over_d1_to_d4(values))]
    return " & ".join(leading + cells + trailing) + r" \\"


def _body_rows(means: CellMeans, columns: Tuple[str, ...]) -> List[str]:
    """Render every condition in paper order with rules between groups."""
    best = best_per_dataset(means)
    lines: List[str] = []
    if any(condition_id == _BASELINE_ID for condition_id, _ in means):
        lines += [_latex_row(_BASELINE_ID, means, best, columns), r"\hline"]
    previous_group: Optional[str] = None
    for condition_id in _CONDITION_ORDER:
        group = get_condition(condition_id).ablation_group
        if previous_group is not None and group != previous_group:
            lines.append(r"\hline")
        previous_group = group
        lines.append(_latex_row(condition_id, means, best, columns))
    return lines


def _pairs_by_dataset(counts: Counts) -> Dict[str, List[PairCounts]]:
    """Group the per-pair counts by dataset column."""
    grouped: Dict[str, List[PairCounts]] = {}
    for (dataset_id, _), row in sorted(counts.items()):
        grouped.setdefault(dataset_id, []).append(row)
    return grouped


def ceiling_values(counts: Counts, key: str) -> Dict[str, float]:
    """
    Average a per-pair candidate ceiling per dataset column.

    Parameters
    ----------
    counts : Counts
        Per-pair reference counts and ceilings.
    key : str
        ``ranking_ceiling``, ``candidate_recall_at_20`` or ``f1_recall_ceiling``.

    Returns
    -------
    Dict[str, float]
        Ceiling per dataset, D3 averaged over its pairs like the metric.  For
        ``f1_recall_ceiling`` the pooled D3 ceiling is added under
        ``"D3_micro"``.
    """
    grouped = _pairs_by_dataset(counts)
    values = {ds: sum(row[key] for row in rows) / len(rows) for ds, rows in grouped.items()}
    if key == "f1_recall_ceiling" and "D3" in grouped:
        rows = grouped["D3"]
        listed = sum(row["n_test_sources_gold_listed"] for row in rows)
        values[_D3_MICRO] = listed / sum(row["n_test_refs"] for row in rows)
    return values


def _count_cells(counts: Counts, key: str, columns: Tuple[str, ...]) -> List[str]:
    """Format one reference count per column: D3 as its pair range, D3 micro as the total."""
    grouped = _pairs_by_dataset(counts)
    cells = []
    for column in columns:
        rows = grouped.get("D3" if column == _D3_MICRO else column, [])
        values = [int(row[key]) for row in rows]
        if not values:
            cells.append(_MISSING)
        elif column == "D3":
            cells.append(f"{min(values):,}--{max(values):,}")
        else:
            cells.append(f"{sum(values):,}")
    return cells


def _annotation_row(label: str, cells: List[str], mean: str = _MISSING) -> str:
    """Render a non-condition row spanning the four leading columns."""
    return " & ".join([rf"\multicolumn{{4}}{{l}}{{{label}}}", *cells, mean]) + r" \\"


def reference_rows(metric: str, counts: Optional[Counts]) -> List[str]:
    """
    Render the ceiling row, and for F1 the reference-count rows.

    Parameters
    ----------
    metric : str
        Metric of the table.
    counts : Optional[Counts]
        Per-pair reference counts and ceilings; ``None`` renders nothing.

    Returns
    -------
    List[str]
        Rows to place after the conditions, led by a rule.
    """
    if counts is None:
        return []
    columns, _ = _columns(metric)
    key = _CEILING_SOURCES[metric]
    ceilings = ceiling_values(counts, key)
    cells = [_format_f1(ceilings.get(ds)) for ds in columns]
    lines = [r"\hline", _annotation_row(_CEILING_LABELS[key], cells, _format_f1(_mean_over_d1_to_d4(ceilings)))]
    if metric == "f1":
        lines.append(_annotation_row("Validation refs", _count_cells(counts, "n_val_refs", columns)))
        lines.append(_annotation_row("Test refs", _count_cells(counts, "n_test_refs", columns)))
    return lines


def _caption(metric: str, has_baseline: bool) -> Tuple[str, str]:
    """Return the caption and label for a metric, noting B0 when present."""
    if metric == "f1":
        caption, label = _CAPTION, r"\label{tab:matching_f1}"
    else:
        caption = _RANKING_CAPTION.format(name=_RANKING_NAMES[metric], unit=_RANKING_UNITS[metric])
        label = r"\label{tab:main_results}" if metric == "mrr" else rf"\label{{tab:ranking_{metric}}}"
    if has_baseline:
        caption = caption.removesuffix("}") + _BASELINE_CAPTION
    return caption, label


def _header_rows(metric: str = "mrr", has_baseline: bool = False) -> List[str]:
    """Render the table preamble, caption and column headings."""
    columns, column_heads = _columns(metric)
    heads = ["Cond", "Strategy", "Model", "Grp", *column_heads, r"Mean\tnote{*}"]
    return [
        r"\begin{table*}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{threeparttable}",
        *_caption(metric, has_baseline),
        r"\begin{tabular}{llll" + "r" * (len(columns) + 1) + "}",
        r"\hline",
        " & ".join(rf"\textbf{{{head}}}" for head in heads) + r" \\",
        r"\hline",
    ]


def _footer_rows() -> List[str]:
    """Render the closing rule, the Mean note below the table and the ends."""
    return [
        r"\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        _MEAN_NOTE,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]


def build_table(
    results_dir: str,
    metric: str = "mrr",
    baseline_dir: Optional[str] = None,
    counts_path: Optional[str] = None,
) -> List[str]:
    """
    Build the complete LaTeX results table for one metric.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.
    metric : str
        ``mrr`` (the main table), ``hits_at_1``, ``f1`` or ``recall_at_1``.
    baseline_dir : Optional[str]
        Root of the B0 n-gram baseline results; ``None`` omits the B0 row.
    counts_path : Optional[str]
        ``metric_ceilings.json``; ``None`` omits the ceiling and count rows
        and requires every result to record any Hits@1 asked for.

    Returns
    -------
    List[str]
        The table's lines, from ``\\begin{table*}`` to ``\\end{table*}``.
    """
    counts = load_reference_counts(counts_path) if counts_path is not None else None
    means = load_cell_means(results_dir, metric, counts)
    if baseline_dir is not None:
        check_known_pair_counts([*Path(results_dir).rglob("*_results.json"),
                                 *Path(baseline_dir).glob("B0/*/*_results.json")])
        means.update(load_baseline_means(baseline_dir, metric, counts))
    has_baseline = any(condition_id == _BASELINE_ID for condition_id, _ in means)
    columns, _ = _columns(metric)
    body = _body_rows(means, columns) + reference_rows(metric, counts)
    return _header_rows(metric, has_baseline) + body + _footer_rows()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the main results table."""
    parser = argparse.ArgumentParser(
        description="Print the main results table of a mean metric as LaTeX."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result files."
    )
    parser.add_argument(
        "--baseline_dir", default=None, help="Root of the B0 n-gram baseline results."
    )
    parser.add_argument("--metric", default="mrr", choices=_METRICS, help="Metric to tabulate.")
    parser.add_argument(
        "--ceilings", default=str(DEFAULT_COUNTS_PATH),
        help="Per-pair reference counts and ceilings from scripts/metric_ceilings.py.",
    )
    return parser


def main() -> None:
    """Read every result file and print the main results table."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    counts_path = args.ceilings if Path(args.ceilings).exists() else None
    if counts_path is None:
        _LOGGER.warning("No counts file at %s; ceiling and count rows are omitted.", args.ceilings)
    print("\n".join(build_table(args.results_dir, args.metric, args.baseline_dir, counts_path)))


if __name__ == "__main__":
    main()
