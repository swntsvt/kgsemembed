"""Command-line entry point for the Section 4.3 entity-type table.

Prints the D3 class-versus-predicate table as LaTeX (``tab:entity_type``) from
:func:`kgsemembed.evaluation.entity_types.entity_type_summary`, the summary
the report's "Entity-Type Analysis" section renders, so the two always print
the same values (#166).  D3 only; MRR and Hits@1 (primary) and F1 (secondary)
are unweighted means over the pairs reporting each entity type, and F1$_\\mu$
pools TP, FP and FN over them.  The caption states the pairs and test
references behind each entity type.  Conditions are listed in registry order;
one without both entity types is left out.

Results are read through ``kgsemembed.evaluation.load_all_results``, so
malformed files, unregistered conditions and duplicate results are skipped with
a warning exactly as in the generated report.

Usage
-----
python scripts/generate_entity_type_table.py \\
    --results_dir data/results/ > artifacts/issue-166/entity_type_table.tex
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import List

import numpy as np
import pandas as pd

from kgsemembed.evaluation import load_all_results
from kgsemembed.evaluation import entity_types

_TABLE_METRICS = ("mrr", "hits_at_1", "f1", "f1_micro")
_METRIC_HEADS = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1", "f1_micro": r"F1$_{\mu}$"}
_TYPE_HEADS = {"class": "cls", "predicate": "prd"}
_CAPTION = (
    r"\caption{{Class versus predicate (property) correspondences on D3. MRR "
    r"and Hits@1 are primary, F1 secondary; each is the unweighted mean over "
    r"the pairs reporting the entity type, and F1$_{{\mu}}$ pools TP, FP and FN "
    r"over them. A pair reports an entity type only with at least {min_refs} "
    r"test references of it: class values rest on {class_pairs} pairs and "
    r"{class_refs} test references, predicate values on {predicate_pairs} "
    r"pairs and {predicate_refs}. D4$_{{\mathrm{{sc}}}}$, a single pair from "
    r"another benchmark, is not included.}}"
)


def _heads() -> List[str]:
    """Return the column headings, one per metric and entity type."""
    metric_heads = [
        rf"{_METRIC_HEADS[metric]}$_{{\mathrm{{{_TYPE_HEADS[entity_type]}}}}}$"
        if metric != "f1_micro" else rf"F1$_{{\mu,\mathrm{{{_TYPE_HEADS[entity_type]}}}}}$"
        for metric in _TABLE_METRICS for entity_type in entity_types.ENTITY_TYPES
    ]
    return ["Cond", "Strategy", "Model", *metric_heads]


def latex_row(row: pd.Series) -> str:
    """
    Render one condition of the shared summary as a LaTeX table row.

    Parameters
    ----------
    row : pd.Series
        One row of :func:`entity_types.entity_type_summary`.

    Returns
    -------
    str
        ``condition & strategy & model & values \\\\``, values to four decimals.
    """
    values = [
        f"{row[f'{entity_type}_{metric}']:.4f}"
        for metric in _TABLE_METRICS for entity_type in entity_types.ENTITY_TYPES
    ]
    return " & ".join([row["condition_id"], row["strategy"], row["model_key"], *values]) + r" \\"


def _caption(complete: pd.DataFrame) -> str:
    """Fill the caption with the pair and reference counts of each entity type."""
    counts = {
        f"{entity_type}_{noun}": entity_types.count_text(complete[f"{entity_type}_n_{noun}"])
        for entity_type in entity_types.ENTITY_TYPES for noun in ("pairs", "refs")
    }
    return _CAPTION.format(min_refs=entity_types.MIN_REFS, **counts)


def build_table(results_dir: str) -> List[str]:
    """
    Build the LaTeX entity-type table.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.

    Returns
    -------
    List[str]
        The table's lines, or an empty list when no condition reports both
        entity types on D3.
    """
    df = load_all_results(results_dir, include_entity_type_breakdown=True)
    complete = entity_types.complete_rows(entity_types.entity_type_summary(df))
    if complete.empty:
        return []
    heads = _heads()
    return [
        r"\begin{table*}[t]", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        _caption(complete), r"\label{tab:entity_type}",
        r"\begin{tabular}{lll" + "r" * (len(heads) - 3) + "}", r"\hline",
        " & ".join(rf"\textbf{{{head}}}" for head in heads) + r" \\", r"\hline",
        *(latex_row(row) for _, row in complete.iterrows()),
        r"\hline", r"\end{tabular}", r"\end{table*}",
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the entity-type table."""
    parser = argparse.ArgumentParser(
        description="Print the D3 class-versus-predicate table as LaTeX."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result files."
    )
    return parser


def main() -> None:
    """Read the D3 results and print the entity-type table."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    for line in build_table(args.results_dir):
        print(line)


if __name__ == "__main__":
    main()
