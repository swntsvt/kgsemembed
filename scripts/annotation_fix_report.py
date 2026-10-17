"""Old-vs-new results of the V1/V2 annotation predicate fix (#136).

Reads the stored main-pipeline results (``--old_dir``, written before the fix)
and the re-runs (``--new_dir``) of every condition whose strategy contains V1
or V2, on D1 and D2, and tabulates MRR, Recall@1 and F1 before and after.
Reads results only; no model is loaded.  The main pipeline writes no
per-source ranks, so the differences are descriptive: no interval or test.

Usage
-----
python scripts/annotation_fix_report.py --old_dir data/results/ \\
    --new_dir data/results_issue136/ --output artifacts/issue-136/results_comparison.md
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS, ExperimentCondition
from kgsemembed.verbalisation.base import annotation_predicates_digest

_PAIRS = {"D1": "d1_snomed_fma", "D2": "d2_anatomy"}
_METRICS = ("mrr", "recall_at_1", "f1")


def affected_conditions() -> List[ExperimentCondition]:
    """Return the registered conditions whose strategy contains V1 or V2."""
    return [
        c for c in EXPERIMENT_CONDITIONS
        if {"V1", "V2"} & set(c.strategy_name.split("+")) and set(_PAIRS) & set(c.datasets)
    ]


def read_result(results_dir: Path, condition_id: str, dataset_id: str) -> Optional[dict]:
    """Return a stored result, or ``None`` when it does not exist."""
    path = results_dir / condition_id / dataset_id / f"{_PAIRS[dataset_id]}_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_new_result(result: dict) -> None:
    """Refuse a re-run written by other annotation predicates or a dirty tree."""
    if result.get("annotation_predicates_sha256") != annotation_predicates_digest():
        raise ValueError(f"{result['condition_id']}/{result['dataset_id']}: annotation digest differs.")
    if result.get("git_dirty"):
        raise ValueError(f"{result['condition_id']}/{result['dataset_id']}: written from a dirty tree.")


def _cells(old: dict, new: dict) -> List[str]:
    """Format old, new and difference of each metric."""
    cells = []
    for metric in _METRICS:
        before, after = old["metrics"][metric], new["metrics"][metric]
        cells.append(f"{before:.4f} → {after:.4f} ({after - before:+.4f})")
    return cells


def comparison_rows(old_dir: Path, new_dir: Path) -> Dict[str, List[str]]:
    """Return table rows, plus the cells missing on either side."""
    rows: Dict[str, List[str]] = {"rows": [], "missing": [], "commits": []}
    for condition in affected_conditions():
        for dataset_id in sorted(set(_PAIRS) & set(condition.datasets)):
            old, new = (read_result(d, condition.condition_id, dataset_id) for d in (old_dir, new_dir))
            label = f"{condition.condition_id}/{dataset_id}"
            if old is None or new is None:
                rows["missing"].append(label)
                continue
            check_new_result(new)
            rows["commits"].append(new["git_commit"])
            prefix = f"| {condition.condition_id} | {condition.strategy_name} | {condition.model_key} | {dataset_id} | "
            rows["rows"].append(prefix + " | ".join(_cells(old, new)) + " |")
    return rows


def report_lines(rows: Dict[str, List[str]]) -> List[str]:
    """Assemble the markdown report."""
    commits = sorted(set(rows["commits"]))
    return [
        "# #136 annotation predicate fix: results before and after", "",
        f"- Re-runs: commit(s) {', '.join(f'`{c[:7]}`' for c in commits) or 'none'}; "
        f"annotation predicates `{annotation_predicates_digest()[:12]}…`.",
        "- Each cell: before → after (difference). One pair per dataset; the main pipeline "
        "stores no per-source ranks, so no intervals or tests.",
        f"- Missing on either side: {', '.join(rows['missing']) or 'none'}.", "",
        "| condition | strategy | model | dataset | MRR | Recall@1 | F1 |",
        "|---|---|---|---|---|---|---|", *rows["rows"], "",
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare results before and after the #136 fix.")
    parser.add_argument("--old_dir", default="data/results/", help="Results written before the fix.")
    parser.add_argument("--new_dir", default="data/results_issue136/", help="Re-runs after the fix.")
    parser.add_argument("--output", default="artifacts/issue-136/results_comparison.md")
    return parser


def main() -> None:
    """Write and print the before/after comparison."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    lines = report_lines(comparison_rows(Path(args.old_dir), Path(args.new_dir)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
