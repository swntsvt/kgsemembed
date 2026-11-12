"""LaTeX table of the registered ablation comparisons under MRR (#188).

Writes the paper's ``tab:stats`` (a ``table*``): one row per registered
comparison (``stats.registered_comparisons()``, groups A-E, registry order),
one column per dataset column.  Each cell is the re-ranking MRR difference
B - A of the per-dataset paired test (#140), signed to three decimals, with
``$^{*}$`` when it is significant after Holm adjustment within the ablation
group and column; ``---`` marks a column the comparison does not cover.

The values come from ``kgsemembed.evaluation.paired_tests.run_paired_tests``
on the per-source sidecars, as ``scripts/generate_report.py`` computes them
for ``report.md``, never from parsing the report.  A covered cell that cannot
be tested (missing or stale sidecar) is refused rather than left blank.

Usage
-----
python scripts/generate_ablation_tests_table.py --results_dir data/results/ \\
    --output artifacts/issue-188/tables/ablation_tests.tex
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from kgsemembed.evaluation.paired_tests import DECISION_METRIC, run_paired_tests
from kgsemembed.evaluation.stats import expand_dataset_ids, registered_comparisons

COLUMNS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
COLUMN_HEADERS = {
    "D1": "D1", "D2": "D2", "D3": "D3", "D4_schema": r"D4$_{\mathrm{sc}}$",
    "D4_instance": r"D4$_{\mathrm{in}}$", "D5": "D5",
}
NOT_COVERED = "---"
SIGNIFICANT = "$^{*}$"
CAPTION = (
    r"Registered ablation comparisons, re-ranking MRR difference B~$-$~A per dataset column "
    r"(positive when B is better), from the per-dataset paired tests "
    r"(Section~\ref{sec:statistics}). $^{*}$: significant after Holm adjustment within the "
    r"ablation group and column ($\alpha = 0.05$). ---: comparison not covered by that column. "
    r"Intervals, $p$-values, Hits@1 and F1 are in the released results report."
)
HEAD = [
    r"\begin{table*}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
    rf"\caption{{{CAPTION}}}", r"\label{tab:stats}", r"\begin{tabular}{lllrrrrrr}", r"\hline",
]
TAIL = [r"\end{tabular}", r"\end{table*}"]

CellKey = Tuple[str, str, str, str]


def format_delta(delta: float, significant: bool) -> str:
    """
    Render one MRR difference as a signed three-decimal LaTeX cell.

    Parameters
    ----------
    delta : float
        B - A of the column's MRR.
    significant : bool
        Whether the Holm decision rejects the null.

    Returns
    -------
    str
        ``$+0.039$`` or ``$-0.035$``, ``$0.000$`` when it rounds to zero,
        followed by ``$^{*}$`` when significant.
    """
    text = f"{delta:+.3f}"
    if float(text) == 0.0:
        text = "0.000"
    return f"${text}$" + (SIGNIFICANT if significant else "")


def decision_cells(rows: List[dict]) -> Dict[CellKey, str]:
    """
    Map each tested MRR row to its rendered cell.

    Parameters
    ----------
    rows : List[dict]
        Rows of :func:`kgsemembed.evaluation.paired_tests.run_paired_tests`.

    Returns
    -------
    Dict[CellKey, str]
        ``(group, condition_a, condition_b, dataset_id)`` to the cell text.

    Raises
    ------
    ValueError
        When an MRR row was not tested, naming the cell and its reason.
    """
    cells: Dict[CellKey, str] = {}
    for row in (r for r in rows if r["metric"] == DECISION_METRIC):
        key = (row["group"], row["condition_a"], row["condition_b"], row["dataset_id"])
        if not row["tested"]:
            raise ValueError(f"Comparison {key} was not tested: {row.get('reason')}")
        cells[key] = format_delta(row["delta"], row["significant"])
    return cells


def comparison_line(group: str, comparison: tuple, cells: Dict[CellKey, str]) -> str:
    """
    Render one registered comparison as a table row.

    Parameters
    ----------
    group : str
        Ablation group letter.
    comparison : tuple
        ``(condition_a, condition_b, datasets)`` from the registry.
    cells : Dict[CellKey, str]
        Output of :func:`decision_cells`.

    Returns
    -------
    str
        The LaTeX row, ``---`` in every column the comparison does not cover.

    Raises
    ------
    KeyError
        When a covered column has no tested cell.
    """
    condition_a, condition_b, datasets = comparison
    covered = set(expand_dataset_ids(list(datasets)))
    values = [cells[(group, condition_a, condition_b, column)] if column in covered else NOT_COVERED
              for column in COLUMNS]
    return " & ".join([group, condition_a, condition_b, *values]) + r" \\"


def render_table(cells: Dict[CellKey, str]) -> str:
    """
    Render the complete ``table*``, one rule after each ablation group.

    Parameters
    ----------
    cells : Dict[CellKey, str]
        Output of :func:`decision_cells`.

    Returns
    -------
    str
        The LaTeX table, ending with a newline.
    """
    headers = [r"\textbf{Grp}", r"\textbf{A}", r"\textbf{B}"]
    headers += [rf"\textbf{{{COLUMN_HEADERS[column]}}}" for column in COLUMNS]
    lines = [*HEAD, " & ".join(headers) + r" \\", r"\hline"]
    for group, comparisons in registered_comparisons().items():
        lines += [comparison_line(group, comparison, cells) for comparison in comparisons]
        lines.append(r"\hline")
    return "\n".join(lines + TAIL) + "\n"


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        ``results_dir`` and ``output``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", default="data/results/")
    parser.add_argument("--output", default="artifacts/issue-188/tables/ablation_tests.tex")
    return parser.parse_args()


def main() -> None:
    """Run the paired tests and write the ablation-tests table."""
    np.random.seed(42)
    random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    table = render_table(decision_cells(run_paired_tests(args.results_dir)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Table written to {output}")


if __name__ == "__main__":
    main()
