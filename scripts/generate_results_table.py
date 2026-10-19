"""Command-line entry point for the Section 4.1 main results table.

Reads every result file and prints a complete LaTeX ``table*`` of mean F1 per
condition per dataset, ready to paste as ``Table~\\ref{tab:main_results}``.

Each cell is the mean F1 over the pairs a condition was run on for that
dataset, so D3 cells average the 21 Conference pairs.  Conditions follow the
paper order and a rule separates consecutive ablation groups.  The best value
in each dataset column is set in bold; values that tie at four decimal places
are all bolded.  The final column averages D1-D4 (both D4 columns) and is shown
only for conditions run on all of them, so every mean it reports covers the
same datasets and the column can be compared row against row.  D5 is left out.

The table is wrapped in ``threeparttable`` so the Mean note sits below the
table at the table's width instead of stretching the last column; the paper's
preamble must load ``\\usepackage{threeparttable}``.

``--metric mrr`` or ``--metric recall_at_1`` prints the same layout for a
ranking metric instead (Table~\\ref{tab:ranking_mrr}, ...).  ``--baseline_dir``
adds the stage-1 n-gram baseline B0 (``kgsemembed.pipeline.ngram_baseline``)
as the first row, above a rule; it ranks the same stored candidates for the
same populations, so every row is directly comparable with it.  B0 takes part
in the bold best-per-column and the Mean column.  Without ``--baseline_dir``
the F1 table is unchanged.

Strategy, model and ablation group come from the condition registry, so the
table cannot drift from the conditions that were actually run.  Results are
read through ``kgsemembed.evaluation.load_all_results``, so malformed files,
unregistered conditions and duplicate results are skipped with a warning
exactly as in the generated report.

Usage
-----
python scripts/generate_results_table.py \\
    --results_dir data/results/ --baseline_dir data/results_ngram_baseline/ \\
    --metric f1
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
from kgsemembed.pipeline.conditions import get_condition

CellMeans = Dict[Tuple[str, str], float]

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_results_table")

_CONDITION_ORDER = (
    "C1", "C2", "C17",
    "C9", "C12", "C3", "C10", "C18",
    "C20",
    "C4", "C5", "C6", "C7", "C8", "C11", "C13", "C16",
    "C14", "C15", "C19",
)
_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
_COLUMN_HEADS = (
    "D1", "D2", "D3", r"D4$_{\mathrm{sc}}$", r"D4$_{\mathrm{in}}$", "D5"
)
_MEAN_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance")
_D3_PAIR_COUNT = 21
_MODEL_MARKERS = {"M2_uncapped": r"M2$^\dagger$", "M4": r"M4$^\ddagger$"}
_MISSING = "---"
_CAPTION = (
    r"\caption{Mean F1 per condition per dataset. "
    r"D3 values are means over 21 pairs. "
    r"Dashes in dataset columns indicate conditions not run on that dataset. "
    r"Bold: best result per dataset column. "
    r"$\dagger$: M2\_uncapped (same weights as M2, "
    r"\texttt{ppas\_budget=None}); C19 is identical to C10 by design. "
    r"C20 (V2, M2) is the controlled baseline for C10 (V2+V8, M2): "
    r"the two differ only by the V8 component, and C20 forms its own "
    r"group (E), placed after group B so it sits beside C10. "
    r"$\ddagger$: M4 (BioLORD) evaluated on D1 and D2 only.}"
)
_BASELINE_ID = "B0"
_BASELINE_LEADING = [_BASELINE_ID, "n-gram", _MISSING, _MISSING]
_BASELINE_CAPTION = (
    r" B0: stage-1 character 3-gram cosine order of the same top-20 candidate "
    r"lists every condition re-ranks, evaluated under the same populations, "
    r"validation-tuned threshold and metric definitions; it is not an "
    r"external matching system.}"
)
_METRICS = ("f1", "mrr", "recall_at_1")
_RANKING_NAMES = {"mrr": "MRR", "recall_at_1": "Recall@1"}
_RANKING_CAPTION = (
    r"\caption{{Mean test {name} per condition per dataset (source-level "
    r"ranking of the same stored top-20 candidates; a test reference source "
    r"with no gold target among them scores 0). D3 values are means over 21 pairs. Dashes in dataset "
    r"columns indicate conditions not run on that dataset. Bold: best result "
    r"per dataset column.}}"
)
_MEAN_NOTE = (
    r"\item[*] Mean over D1--D4 (D4 schema and instance), shown only for "
    r"conditions run on all of them; excludes D5 (recall@20 ceiling 26.67\%)."
)


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


def load_cell_means(results_dir: str, metric: str = "f1") -> CellMeans:
    """
    Average a pair-level metric for every condition and dataset with results.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.
    metric : str
        Metric column, one of ``f1``, ``mrr``, ``recall_at_1``.

    Returns
    -------
    CellMeans
        Mean metric keyed by ``(condition_id, dataset_id)``.
    """
    return _cell_means(load_all_results(results_dir), metric)


def load_baseline_means(baseline_dir: str, metric: str = "f1") -> CellMeans:
    """
    Average a pair-level metric of the B0 n-gram baseline per dataset.

    B0 is not a registered condition, so its files are read directly rather
    than through ``load_all_results``.

    Parameters
    ----------
    baseline_dir : str
        Root holding ``B0/{dataset_id}/{pair}_results.json``.
    metric : str
        Metric key inside each result's ``metrics`` block.

    Returns
    -------
    CellMeans
        Mean metric keyed by ``("B0", dataset_id)``; empty when none exist.
    """
    paths = sorted(Path(baseline_dir).glob(f"{_BASELINE_ID}/*/*_results.json"))
    if not paths:
        _LOGGER.warning("No B0 results under %s; the B0 row is omitted.", baseline_dir)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = [
        {"condition_id": p["condition_id"], "dataset_id": p["dataset_id"], metric: p["metrics"][metric]}
        for p in payloads
    ]
    return _cell_means(pd.DataFrame(rows), metric)


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


def best_per_dataset(means: CellMeans) -> Dict[str, str]:
    """
    Find the best formatted F1 in each dataset column.

    Parameters
    ----------
    means : CellMeans
        Mean F1 keyed by ``(condition_id, dataset_id)``.

    Returns
    -------
    Dict[str, str]
        The best value per dataset as printed, so ties at four decimal places
        compare equal.  Datasets with no results are absent.
    """
    best: Dict[str, float] = {}
    for (condition_id, dataset_id), value in means.items():
        listed = condition_id in _CONDITION_ORDER or condition_id == _BASELINE_ID
        if listed and dataset_id in _DATASETS:
            best[dataset_id] = max(value, best.get(dataset_id, value))
    return {dataset_id: _format_f1(value) for dataset_id, value in best.items()}


def _mean_over_d1_to_d4(values: Dict[str, Optional[float]]) -> Optional[float]:
    """Average D1-D4, or ``None`` unless the condition has every one of them."""
    present = [
        value for dataset_id in _MEAN_DATASETS
        if (value := values[dataset_id]) is not None
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


def _latex_row(condition_id: str, means: CellMeans, best: Dict[str, str]) -> str:
    """
    Render one condition as a LaTeX table row.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    means : CellMeans
        Mean F1 keyed by ``(condition_id, dataset_id)``.
    best : Dict[str, str]
        Best formatted F1 per dataset, from :func:`best_per_dataset`.

    Returns
    -------
    str
        ``condition & strategy & model & group & F1 ... & mean \\\\``.
    """
    values = {ds: means.get((condition_id, ds)) for ds in _DATASETS}
    cells = [_format_f1(values[ds]) for ds in _DATASETS]
    cells = [
        rf"\textbf{{{cell}}}" if cell == best.get(ds) else cell
        for ds, cell in zip(_DATASETS, cells)
    ]
    leading = _leading_cells(condition_id)
    trailing = [_format_f1(_mean_over_d1_to_d4(values))]
    return " & ".join(leading + cells + trailing) + r" \\"


def _body_rows(means: CellMeans) -> List[str]:
    """Render every condition in paper order with rules between groups."""
    best = best_per_dataset(means)
    lines: List[str] = []
    if any(condition_id == _BASELINE_ID for condition_id, _ in means):
        lines += [_latex_row(_BASELINE_ID, means, best), r"\hline"]
    previous_group: Optional[str] = None
    for condition_id in _CONDITION_ORDER:
        group = get_condition(condition_id).ablation_group
        if previous_group is not None and group != previous_group:
            lines.append(r"\hline")
        previous_group = group
        lines.append(_latex_row(condition_id, means, best))
    return lines


def _caption(metric: str, has_baseline: bool) -> Tuple[str, str]:
    """Return the caption and label for a metric, noting B0 when present."""
    if metric == "f1":
        caption, label = _CAPTION, r"\label{tab:main_results}"
    else:
        caption = _RANKING_CAPTION.format(name=_RANKING_NAMES[metric])
        label = rf"\label{{tab:ranking_{metric}}}"
    if has_baseline:
        caption = caption.removesuffix("}") + _BASELINE_CAPTION
    return caption, label


def _header_rows(metric: str = "f1", has_baseline: bool = False) -> List[str]:
    """Render the table preamble, caption and column headings."""
    heads = ["Cond", "Strategy", "Model", "Grp", *_COLUMN_HEADS, r"Mean\tnote{*}"]
    return [
        r"\begin{table*}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{threeparttable}",
        *_caption(metric, has_baseline),
        r"\begin{tabular}{llll" + "r" * (len(_DATASETS) + 1) + "}",
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
    results_dir: str, metric: str = "f1", baseline_dir: Optional[str] = None
) -> List[str]:
    """
    Build the complete LaTeX results table for one metric.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.
    metric : str
        ``f1`` (the main table), ``mrr`` or ``recall_at_1``.
    baseline_dir : Optional[str]
        Root of the B0 n-gram baseline results; ``None`` omits the B0 row.

    Returns
    -------
    List[str]
        The table's lines, from ``\\begin{table*}`` to ``\\end{table*}``.
    """
    means = load_cell_means(results_dir, metric)
    if baseline_dir is not None:
        means.update(load_baseline_means(baseline_dir, metric))
    has_baseline = any(condition_id == _BASELINE_ID for condition_id, _ in means)
    return _header_rows(metric, has_baseline) + _body_rows(means) + _footer_rows()


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
    parser.add_argument("--metric", default="f1", choices=_METRICS, help="Metric to tabulate.")
    return parser


def main() -> None:
    """Read every result file and print the main results table."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    print("\n".join(build_table(args.results_dir, args.metric, args.baseline_dir)))


if __name__ == "__main__":
    main()
