"""Command-line entry point for the Section 4.1 main results table.

Reads every result file and prints a complete LaTeX ``table*`` of mean F1 per
condition per dataset, ready to paste as ``Table~\\ref{tab:main_results}``.

Each cell is the mean F1 over the pairs a condition was run on for that
dataset, so D3 cells average the 21 Conference pairs.  Conditions follow the
paper order and a rule separates consecutive ablation groups.  The best value
in each dataset column is set in bold; values that tie at four decimal places
are all bolded.  The final column averages the non-D5 cells a condition has.

Strategy, model and ablation group come from the condition registry, so the
table cannot drift from the conditions that were actually run.  Results are
read through ``kgsemembed.evaluation.load_all_results``, so malformed files,
unregistered conditions and duplicate results are skipped with a warning
exactly as in the generated report.

Usage
-----
python scripts/generate_results_table.py \\
    --results_dir data/results/
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from kgsemembed.evaluation import load_all_results
from kgsemembed.pipeline.conditions import get_condition

CellMeans = Dict[Tuple[str, str], float]

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_results_table")

_CONDITION_ORDER = (
    "C1", "C2", "C17",
    "C9", "C12", "C3", "C10", "C18",
    "C4", "C5", "C6", "C7", "C8", "C11", "C13", "C16",
    "C14", "C15", "C19",
)
_DATASETS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
_COLUMN_HEADS = (
    "D1", "D2", "D3", r"D4$_{\mathrm{sc}}$", r"D4$_{\mathrm{in}}$", "D5"
)
_MEAN_EXCLUDED_DATASET = "D5"
_D3_PAIR_COUNT = 21
_MODEL_MARKERS = {"M2_uncapped": r"M2$^\dagger$", "M4": r"M4$^\ddagger$"}
_MISSING = "---"
_CAPTION = (
    r"\caption{Mean F1 per condition per dataset. "
    r"D3 values are means over 21 pairs. "
    r"Dashes indicate conditions not run on that dataset. "
    r"Bold: best result per dataset column. "
    r"$\dagger$: M2\_uncapped (same weights as M2, "
    r"\texttt{ppas\_budget=None}); C19 is identical to C10 by design. "
    r"$\ddagger$: M4 (BioLORD) evaluated on D1 and D2 only.}"
)
_FOOTNOTE = r"$^*$Mean excludes D5 (recall@20 ceiling 26.67\%)."


def load_cell_means(results_dir: str) -> CellMeans:
    """
    Average pair-level F1 for every condition and dataset with results.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.

    Returns
    -------
    CellMeans
        Mean F1 keyed by ``(condition_id, dataset_id)``.
    """
    df = load_all_results(results_dir)
    grouped = df.groupby(["condition_id", "dataset_id"])["f1"]
    _warn_on_partial_d3(grouped.size().to_dict())
    return {key: float(value) for key, value in grouped.mean().items()}


def _warn_on_partial_d3(pair_counts: Dict[Tuple[str, str], int]) -> None:
    """Warn for any D3 cell not averaged over all Conference pairs."""
    for (condition_id, dataset_id), count in sorted(pair_counts.items()):
        if dataset_id == "D3" and count != _D3_PAIR_COUNT:
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
        if condition_id in _CONDITION_ORDER and dataset_id in _DATASETS:
            best[dataset_id] = max(value, best.get(dataset_id, value))
    return {dataset_id: _format_f1(value) for dataset_id, value in best.items()}


def _mean_excluding_d5(values: Dict[str, Optional[float]]) -> Optional[float]:
    """Average the cells a condition has, leaving out D5."""
    kept = [
        value for dataset_id, value in values.items()
        if dataset_id != _MEAN_EXCLUDED_DATASET and value is not None
    ]
    return sum(kept) / len(kept) if kept else None


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
    condition = get_condition(condition_id)
    values = {ds: means.get((condition_id, ds)) for ds in _DATASETS}
    cells = [_format_f1(values[ds]) for ds in _DATASETS]
    cells = [
        rf"\textbf{{{cell}}}" if cell == best.get(ds) else cell
        for ds, cell in zip(_DATASETS, cells)
    ]
    model = _MODEL_MARKERS.get(condition.model_key, condition.model_key)
    leading = [condition_id, condition.strategy_name, model, condition.ablation_group]
    trailing = [_format_f1(_mean_excluding_d5(values))]
    return " & ".join(leading + cells + trailing) + r" \\"


def _body_rows(means: CellMeans) -> List[str]:
    """Render every condition in paper order with rules between groups."""
    best = best_per_dataset(means)
    lines: List[str] = []
    previous_group: Optional[str] = None
    for condition_id in _CONDITION_ORDER:
        group = get_condition(condition_id).ablation_group
        if previous_group is not None and group != previous_group:
            lines.append(r"\hline")
        previous_group = group
        lines.append(_latex_row(condition_id, means, best))
    return lines


def _header_rows() -> List[str]:
    """Render the table preamble, caption and column headings."""
    heads = ["Cond", "Strategy", "Model", "Grp", *_COLUMN_HEADS, "Mean$^*$"]
    return [
        r"\begin{table*}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        _CAPTION,
        r"\label{tab:main_results}",
        r"\begin{tabular}{llll" + "r" * (len(_DATASETS) + 1) + "}",
        r"\hline",
        " & ".join(rf"\textbf{{{head}}}" for head in heads) + r" \\",
        r"\hline",
    ]


def _footer_rows() -> List[str]:
    """Render the closing rule, mean footnote and environment ends."""
    n_columns = 4 + len(_DATASETS) + 1
    return [
        r"\hline",
        rf"\multicolumn{{{n_columns}}}{{l}}{{{_FOOTNOTE}}} \\",
        r"\end{tabular}",
        r"\end{table*}",
    ]


def build_table(results_dir: str) -> List[str]:
    """
    Build the complete LaTeX main results table.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.

    Returns
    -------
    List[str]
        The table's lines, from ``\\begin{table*}`` to ``\\end{table*}``.
    """
    means = load_cell_means(results_dir)
    return _header_rows() + _body_rows(means) + _footer_rows()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the main results table."""
    parser = argparse.ArgumentParser(
        description="Print the main results table of mean F1 as LaTeX."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result files."
    )
    return parser


def main() -> None:
    """Read every result file and print the main results table."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    print("\n".join(build_table(args.results_dir)))


if __name__ == "__main__":
    main()
