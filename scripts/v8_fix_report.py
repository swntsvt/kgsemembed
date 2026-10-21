"""Old-vs-new results of the V8 annotation-predicate exclusion (#145).

Reads the stored main-pipeline results (``--old_dir``, written before the fix)
and the re-runs (``--new_dir``) of every condition whose strategy contains V8,
on the columns whose V8 text changed (D1, D2, D4_instance, D5) plus D4_schema,
whose text is unchanged and so must reproduce.  D3 texts are byte-identical
(``scripts/v8_inventory.py``), so D3 is not re-run.  Reads results only; no
model is loaded.  The main pipeline writes no per-source ranks, so per-column
differences are descriptive: no interval or test.

The second table sets C10 (V2+V8/M2) against its controlled baseline C20
(V2/M2, no V8, unchanged) per column, before and after.  The group-E
Wilcoxon test over dataset columns (F1, as in ``report.md``) is computed on
the stored results and on a view where the re-runs replace them.

Usage
-----
python scripts/v8_fix_report.py --old_dir data/results/ \\
    --new_dir data/results_issue145/ --output artifacts/issue-145/results_comparison.md
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from kgsemembed.evaluation.reference_counts import load_reference_counts, result_metric
from kgsemembed.evaluation.stats import load_results_for_condition, run_group_comparisons
from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS, ExperimentCondition
from kgsemembed.verbalisation.v8 import v8_exclusions_digest

RERUN_COLUMNS = ("D1", "D2", "D4_schema", "D4_instance", "D5")
COLUMNS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
UNCHANGED_COLUMNS = ("D4_schema",)
METRICS = ("mrr", "hits_at_1", "f1")
_METRIC_NAMES = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1"}


def affected_conditions() -> List[ExperimentCondition]:
    """Return the registered conditions whose strategy contains V8."""
    return [c for c in EXPERIMENT_CONDITIONS if "V8" in c.strategy_name.split("+")]


def _condition_columns(condition: ExperimentCondition) -> List[str]:
    """Return the re-run columns a condition is configured for."""
    datasets = set(condition.datasets)
    return [c for c in RERUN_COLUMNS if c.split("_")[0] in datasets]


def check_new_result(result: dict) -> None:
    """Refuse a re-run written by other V8 exclusions or a dirty tree."""
    label = f"{result['condition_id']}/{result['dataset_id']}/{result['pair_name']}"
    if result.get("v8_exclusions_sha256") != v8_exclusions_digest():
        raise ValueError(f"{label}: V8 exclusion digest differs.")
    if result.get("git_dirty"):
        raise ValueError(f"{label}: written from a dirty tree.")


def column_metric(results: Optional[dict], metric: str, counts: dict) -> Optional[float]:
    """Return a column's mean metric over its pairs (D3: macro), or ``None``."""
    if not results:
        return None
    return float(np.mean([result_metric(payload, metric, counts) for payload in results.values()]))


def _cell(before: Optional[float], after: Optional[float]) -> str:
    """Format before → after (difference), or a dash when either is missing."""
    if before is None or after is None:
        return "—"
    return f"{before:.4f} → {after:.4f} ({after - before:+.4f})"


def _new_results(new_dir: Path, condition_id: str, column: str) -> Optional[dict]:
    """Load and check the re-runs of one condition and column."""
    results = load_results_for_condition(condition_id, column, str(new_dir))
    for payload in (results or {}).values():
        check_new_result(payload)
    return results


def comparison_rows(old_dir: Path, new_dir: Path, counts: dict) -> Dict[str, List[str]]:
    """Return the per-condition table rows, missing cells, commits and failed reproductions."""
    rows: Dict[str, List[str]] = {"rows": [], "missing": [], "commits": [], "not_reproduced": []}
    for condition in affected_conditions():
        for column in _condition_columns(condition):
            old = load_results_for_condition(condition.condition_id, column, str(old_dir))
            new = _new_results(new_dir, condition.condition_id, column)
            label = f"{condition.condition_id}/{column}"
            if not old or not new:
                rows["missing"].append(label)
                continue
            rows["commits"] += [payload["git_commit"] for payload in new.values()]
            values = [(column_metric(old, m, counts), column_metric(new, m, counts)) for m in METRICS]
            if column in UNCHANGED_COLUMNS and any(abs(a - b) > 1e-12 for a, b in values):
                rows["not_reproduced"].append(label)
            prefix = f"| {condition.condition_id} | {condition.strategy_name} | {condition.model_key} | {column} | "
            rows["rows"].append(prefix + " | ".join(_cell(a, b) for a, b in values) + " |")
    return rows


def _c10_minus_c20(c10: Optional[dict], c20: Optional[dict], metric: str, counts: dict) -> Optional[float]:
    """Return C10 − C20 of one column and metric, or ``None`` when either is missing."""
    a, b = column_metric(c10, metric, counts), column_metric(c20, metric, counts)
    return None if a is None or b is None else a - b


def c20_c10_rows(old_dir: Path, new_dir: Path, counts: dict) -> List[str]:
    """Return per-column C10 − C20 rows, before and after, for every metric."""
    rows = []
    for column in COLUMNS:
        c20 = load_results_for_condition("C20", column, str(old_dir))
        c10_old = load_results_for_condition("C10", column, str(old_dir))
        c10_new = _new_results(new_dir, "C10", column) if column in RERUN_COLUMNS else c10_old
        cells = [_cell(_c10_minus_c20(c10_old, c20, m, counts), _c10_minus_c20(c10_new, c20, m, counts))
                 for m in METRICS]
        rows.append(f"| {column} | " + " | ".join(cells) + " |")
    return rows


def merged_view(old_dir: Path, new_dir: Path, view: Path) -> Path:
    """Copy C10 and C20 from ``old_dir`` into ``view``, the C10 re-runs replacing theirs."""
    for condition_id in ("C10", "C20"):
        shutil.copytree(old_dir / condition_id, view / condition_id)
    for column in RERUN_COLUMNS:
        source = new_dir / "C10" / column
        if source.is_dir():
            shutil.rmtree(view / "C10" / column)
            shutil.copytree(source, view / "C10" / column)
    return view


def _group_e_line(label: str, result: dict) -> str:
    """Format one group-E Wilcoxon result."""
    p_value = "not tested" if result["p_value"] is None else f"p {result['p_value']:.4f}"
    return (f"- {label}: pair W/T/L {result['pair_wins']}/{result['pair_ties']}/{result['pair_losses']}, "
            f"{p_value}, ΔF1 (C10 − C20) {result['delta_f1']:+.4f}, "
            f"significant after correction: {result['corrected_significant']}.")


def group_e_lines(old_dir: Path, new_dir: Path) -> List[str]:
    """Return the group-E Wilcoxon results before and after the re-runs."""
    before = run_group_comparisons("E", str(old_dir))[0]
    with tempfile.TemporaryDirectory() as tmp:
        after = run_group_comparisons("E", str(merged_view(old_dir, new_dir, Path(tmp) / "view")))[0]
    return [_group_e_line("Before", before), _group_e_line("After", after)]


def _metric_header() -> str:
    """Return the metric part of a table header."""
    return " | ".join(_METRIC_NAMES[m] for m in METRICS)


def report_lines(rows: Dict[str, List[str]], c20_rows: List[str], group_e: List[str]) -> List[str]:
    """Assemble the markdown report."""
    commits = sorted(set(rows["commits"]))
    return [
        "# #145 V8 annotation-predicate exclusion: results before and after", "",
        f"- Re-runs: commit(s) {', '.join(f'`{c[:7]}`' for c in commits) or 'none'}; "
        f"V8 exclusions `{v8_exclusions_digest()[:12]}…`.",
        "- Each cell: before → after (difference). One pair per column (D3: macro mean of 21); "
        "the main pipeline stores no per-source ranks, so no intervals or tests per column.",
        "- Hits@1 of results stored before #139 is derived from Recall@1 (`evaluation/reference_counts.py`).",
        "- D3 is not re-run: its V8 texts are byte-identical (`v8_text_inventory.md`).",
        f"- D4_schema texts are unchanged, so its re-runs must reproduce; not reproduced: "
        f"{', '.join(rows['not_reproduced']) or 'none'}.",
        f"- Missing on either side: {', '.join(rows['missing']) or 'none'}.", "",
        f"| condition | strategy | model | column | {_metric_header()} |",
        "|---|---|---|---|---|---|---|", *rows["rows"], "",
        "## C10 (V2+V8/M2) − C20 (V2/M2), before → after", "",
        "C20 has no V8 component and is unchanged; D3 is the stored (unchanged) C10.", "",
        f"| column | {_metric_header()} |", "|---|---|---|---|", *c20_rows, "",
        "## Group E (C20 vs C10), Wilcoxon over dataset columns, F1", "", *group_e, "",
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare results before and after the #145 fix.")
    parser.add_argument("--old_dir", default="data/results/", help="Results written before the fix.")
    parser.add_argument("--new_dir", default="data/results_issue145/", help="Re-runs after the fix.")
    parser.add_argument("--output", default="artifacts/issue-145/results_comparison.md")
    return parser


def main() -> None:
    """Write and print the before/after comparison."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    old_dir, new_dir, counts = Path(args.old_dir), Path(args.new_dir), load_reference_counts()
    lines = report_lines(
        comparison_rows(old_dir, new_dir, counts), c20_c10_rows(old_dir, new_dir, counts),
        group_e_lines(old_dir, new_dir),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
