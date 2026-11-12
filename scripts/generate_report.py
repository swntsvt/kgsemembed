"""Command-line entry point for the Phase 2 experiment report.

Loads previously generated result files, runs the per-dataset paired tests of
every registered comparison (#140; the primary analysis, read from the
per-source sidecars), the paired tests of every condition against the stage-1
baseline B0 (#165; read from ``--baseline_dir``), the cross-dataset Wilcoxon
consistency summary on MRR and on F1, and renders a Markdown report.  Statistical comparison for one
ablation group failing does not abort the report: the failure is logged and the
remaining groups continue.  A comparison without sidecars is listed as not
tested.

Usage
-----
python scripts/generate_report.py \\
    --results_dir data/results/ \\
    --baseline_dir data/results_ngram_baseline/ \\
    --output data/results/report.md
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from kgsemembed.evaluation import run_group_comparisons
from kgsemembed.evaluation.aggregator import generate_markdown_report, load_all_results
from kgsemembed.evaluation.paired_tests import run_baseline_tests, run_paired_tests
from kgsemembed.evaluation.reference_counts import DEFAULT_COUNTS_PATH, load_reference_counts

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_report")
_ABLATION_GROUPS = ("A", "B", "C", "D", "E")
_WILCOXON_METRICS = ("mrr", "f1")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the report generator."""
    parser = argparse.ArgumentParser(
        description="Generate the Phase 2 experiment Markdown report."
    )
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result JSON files."
    )
    parser.add_argument(
        "--baseline_dir", default="data/results_ngram_baseline/",
        help="Root of the B0 results (pipeline/ngram_baseline.py); absent omits the B0 section.",
    )
    parser.add_argument(
        "--output", default="data/results/report.md", help="Destination Markdown report path."
    )
    parser.add_argument(
        "--ceilings", default=str(DEFAULT_COUNTS_PATH),
        help="Per-pair reference counts and ceilings (scripts/metric_ceilings.py).",
    )
    return parser


def _collect_comparisons(results_dir: str) -> List[dict]:
    """Run every ablation group's comparisons, skipping failures with a warning."""
    comparisons: List[dict] = []
    for metric in _WILCOXON_METRICS:
        for group in _ABLATION_GROUPS:
            try:
                comparisons.extend(run_group_comparisons(group, results_dir, metric=metric))
            except Exception as exc:
                _LOGGER.warning("Skipping ablation group %s on %s: %s", group, metric, exc)
    return comparisons


def _load_counts(path: str) -> Optional[dict]:
    """Load the reference counts, or return ``None`` with a warning when absent."""
    if not Path(path).is_file():
        _LOGGER.warning("No reference counts at %s; ceilings and older results' Hits@1 omitted.", path)
        return None
    return load_reference_counts(path)


def _paired_results(results_dir: str, counts: Optional[dict]) -> Optional[List[dict]]:
    """Run the per-dataset paired tests, logging and omitting them if the run fails."""
    try:
        return run_paired_tests(results_dir, counts)
    except Exception as exc:
        _LOGGER.warning("Skipping the per-dataset paired tests: %s", exc)
        return None


def _baseline_results(
    results_dir: str, baseline_dir: str, counts: Optional[dict]
) -> Optional[List[dict]]:
    """Run the tests against B0, omitting them with a warning if B0 or the run is missing."""
    if not Path(baseline_dir).is_dir():
        _LOGGER.warning("No B0 results at %s; the section against B0 is omitted.", baseline_dir)
        return None
    try:
        return run_baseline_tests(results_dir, baseline_dir, counts)
    except Exception as exc:
        _LOGGER.warning("Skipping the paired tests against B0: %s", exc)
        return None


def main() -> None:
    """Load results, gather statistical comparisons, and write the report."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    counts = _load_counts(args.ceilings)
    df = load_all_results(args.results_dir, include_entity_type_breakdown=True, counts=counts)
    comparisons = _collect_comparisons(args.results_dir)
    paired = _paired_results(args.results_dir, counts)
    baseline = _baseline_results(args.results_dir, args.baseline_dir, counts)
    generate_markdown_report(
        df, comparisons or None, args.output, paired_results=paired, baseline_results=baseline
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
