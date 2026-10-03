"""Command-line entry point for the Section 4.3 entity-type table.

Reads the D3 result files and prints one LaTeX row per condition comparing
class F1 with predicate F1, ready to paste into ``Table~\\ref{tab:entity_type}``.

Each F1 is the mean over the 21 Conference pairs weighted by the bucket's test
reference count, so a pair contributes in proportion to the references it
holds.  Buckets with fewer than three references are excluded, matching the
threshold the pipeline applies when writing ``per_entity_type``.  A condition
missing either entity type is omitted from the table.

Results are read through ``kgsemembed.evaluation.load_all_results``, so
malformed files, unregistered conditions and duplicate results are skipped with
a warning exactly as in the generated report.

Usage
-----
python scripts/generate_entity_type_table.py \\
    --results_dir data/results/
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import List, Optional

import numpy as np
import pandas as pd

from kgsemembed.evaluation import load_all_results
from kgsemembed.pipeline.conditions import get_condition

_DATASET_ID = "D3"
_ENTITY_TYPES = ("class", "predicate")
_MIN_REFS = 3
_CONDITION_ORDER = ("C1", "C2", "C3", "C4", "C7", "C9", "C10", "C12", "C13", "C18")
_TABLE_HEADER = "% Paste into Table~\\ref{tab:entity_type}"


def _entity_type_rows(results_dir: str) -> pd.DataFrame:
    """
    Load the D3 class and predicate rows with enough references to report.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.

    Returns
    -------
    pd.DataFrame
        One row per condition, pair and entity type, with numeric ``n_refs``.
    """
    df = load_all_results(results_dir, include_entity_type_breakdown=True)
    df = df.assign(n_refs=pd.to_numeric(df["n_refs"], errors="coerce"))
    return df[
        (df["dataset_id"] == _DATASET_ID)
        & df["entity_type"].isin(_ENTITY_TYPES)
        & (df["n_refs"] >= _MIN_REFS)
    ]


def _weighted_f1(rows: pd.DataFrame) -> Optional[float]:
    """
    Average F1 over pairs, weighting each pair by its test reference count.

    Parameters
    ----------
    rows : pd.DataFrame
        Entity-type rows of one condition and one entity type.

    Returns
    -------
    Optional[float]
        The weighted mean F1, or ``None`` when the rows hold no references.
    """
    total_refs = rows["n_refs"].sum()
    if total_refs == 0:
        return None
    return float((rows["f1"] * rows["n_refs"]).sum() / total_refs)


def _latex_row(condition_id: str, rows: pd.DataFrame) -> Optional[str]:
    """
    Render one condition as a LaTeX table row.

    Parameters
    ----------
    condition_id : str
        Registered condition identifier, e.g. ``"C1"``.
    rows : pd.DataFrame
        Entity-type rows as returned by :func:`_entity_type_rows`.

    Returns
    -------
    Optional[str]
        ``condition & model & class F1 & predicate F1 & gap & n_pairs \\\\``,
        where ``n_pairs`` counts the pairs contributing to class F1, or
        ``None`` when either entity type has no data for the condition.
    """
    own = rows[rows["condition_id"] == condition_id]
    class_rows = own[own["entity_type"] == "class"]
    class_f1 = _weighted_f1(class_rows)
    predicate_f1 = _weighted_f1(own[own["entity_type"] == "predicate"])
    if class_f1 is None or predicate_f1 is None:
        return None
    model_key = get_condition(condition_id).model_key
    return (
        f"{condition_id} & {model_key} & {class_f1:.4f} & {predicate_f1:.4f} & "
        f"{class_f1 - predicate_f1:+.4f} & {len(class_rows)} \\\\"
    )


def build_table_rows(results_dir: str) -> List[str]:
    """
    Build the LaTeX rows of the entity-type table in paper order.

    Parameters
    ----------
    results_dir : str
        Root directory of the experiment result files.

    Returns
    -------
    List[str]
        One row per condition in :data:`_CONDITION_ORDER` that has both class
        and predicate data; conditions without it are left out.
    """
    rows = _entity_type_rows(results_dir)
    latex_rows = [_latex_row(condition_id, rows) for condition_id in _CONDITION_ORDER]
    return [row for row in latex_rows if row is not None]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the entity-type table."""
    parser = argparse.ArgumentParser(
        description="Print the D3 class-versus-predicate F1 rows as LaTeX."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result files."
    )
    return parser


def main() -> None:
    """Read the D3 results and print the entity-type table rows."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    print(_TABLE_HEADER)
    for row in build_table_rows(args.results_dir):
        print(row)


if __name__ == "__main__":
    main()
