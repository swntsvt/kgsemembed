"""Command-line entry point for the Phase 2 experiment report.

Loads previously generated result files, runs the per-group Wilcoxon
comparisons, and renders a Markdown report.  Statistical comparison for one
ablation group failing does not abort the report: the failure is logged and the
remaining groups continue.

Usage
-----
python scripts/generate_report.py \\
    --results_dir data/results/ \\
    --output data/results/report.md
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import List

import numpy as np

from kgsemembed.evaluation import run_group_comparisons
from kgsemembed.evaluation.aggregator import generate_markdown_report, load_all_results

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_report")
_ABLATION_GROUPS = ("A", "B", "C", "D")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the report generator."""
    parser = argparse.ArgumentParser(
        description="Generate the Phase 2 experiment Markdown report."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result JSON files."
    )
    parser.add_argument(
        "--output", default="data/results/report.md", help="Destination Markdown report path."
    )
    return parser


def _collect_comparisons(results_dir: str) -> List[dict]:
    """Run every ablation group's comparisons, skipping failures with a warning."""
    comparisons: List[dict] = []
    for group in _ABLATION_GROUPS:
        try:
            comparisons.extend(run_group_comparisons(group, results_dir))
        except Exception as exc:
            _LOGGER.warning("Skipping ablation group %s: %s", group, exc)
    return comparisons


def main() -> None:
    """Load results, gather statistical comparisons, and write the report."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    df = load_all_results(args.results_dir)
    comparisons = _collect_comparisons(args.results_dir)
    generate_markdown_report(df, comparisons or None, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
