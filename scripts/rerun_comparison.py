"""Compare a #183 re-run with the stored results and check its guarantees.

For every result in ``--new_dir`` (and B0 in ``--new_baseline_dir``), this
compares every metric and TP/FP/FN count with the stored result of the same
condition, column and pair, and classifies the cell:

- **expected to change** — a cell the #183 corrections touch: a budgeted
  condition (C3, C4, C5, C6, C7, C8, C16: budgets counted exactly), an M4
  condition (C6, C11, C16, C17: input limit 512; C6/C16 also write V7's
  annotation once), or a D5 condition whose strategy has a V3, V4 or V6
  component (C3, C4, C5, C7, C14: datatype suffix removed);
- **must reproduce** — every other cell, B0 included; any difference fails.

It also fails unless every budgeted result reports no text over its budget
and none truncated, every result records the encoder input and its sidecar
the per-entity input lengths, and C19 equals C10 wherever both ran.

Usage
-----
python scripts/rerun_comparison.py --new_dir data/results_issue183/ \\
    --new_baseline_dir data/results_ngram_baseline_issue183/ \\
    --output artifacts/issue-183/rerun_comparison.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BUDGETED = {"C3", "C4", "C5", "C6", "C7", "C8", "C16"}
M4_CONDITIONS = {"C6", "C11", "C16", "C17"}
D5_SUFFIX = {"C3", "C4", "C5", "C7", "C14"}
COMPARED = ("metrics", "counts")
_RESULT_SUFFIX = "_results.json"


def expected_change(condition_id: str, column: str) -> Optional[str]:
    """
    Name the #183 correction that may change a cell, or ``None`` if none does.

    Parameters
    ----------
    condition_id : str
        Condition, or ``"B0"``.
    column : str
        Dataset column.

    Returns
    -------
    Optional[str]
        The reasons, comma-separated, or ``None`` for a cell that must reproduce.
    """
    reasons = [name for name, applies in (
        ("budget", condition_id in BUDGETED),
        ("M4 512", condition_id in M4_CONDITIONS),
        ("D5 suffix", column == "D5" and condition_id in D5_SUFFIX),
    ) if applies]
    return ", ".join(reasons) or None


def result_files(root: Path) -> Dict[tuple, Path]:
    """
    Index the result files of a results tree by condition, column and pair.

    Parameters
    ----------
    root : Path
        ``<root>/<condition>/<column>/<pair>_results.json``.

    Returns
    -------
    Dict[tuple, Path]
        ``(condition, column, pair)`` to file.
    """
    return {(p.parent.parent.name, p.parent.name, p.name[: -len(_RESULT_SUFFIX)]): p
            for p in sorted(root.glob(f"*/*/*{_RESULT_SUFFIX}"))}


def differences(old: dict, new: dict) -> Dict[str, tuple]:
    """
    Return every compared metric or count that differs, as ``(old, new)``.

    Parameters
    ----------
    old, new : dict
        Result payloads.

    Returns
    -------
    Dict[str, tuple]
        Differing keys.
    """
    return {key: (old[block].get(key), value) for block in COMPARED
            for key, value in new[block].items() if old[block].get(key) != value}


def guarantee_problems(key: tuple, payload: dict, path: Path) -> List[str]:
    """
    List the #183 guarantees a new result breaks.

    Parameters
    ----------
    key : tuple
        ``(condition, column, pair)``.
    payload : dict
        The new result.
    path : Path
        Its file, beside which the sidecar lies.

    Returns
    -------
    List[str]
        Problems, empty when the result keeps every guarantee.
    """
    record = payload.get("encoder_input")
    if record is None:
        return [f"{key}: no encoder_input record"]
    sidecar = json.loads(path.with_name(path.name.replace(_RESULT_SUFFIX, "_entities.json")).read_text())
    problems = [] if "inputs" in sidecar else [f"{key}: sidecar has no per-entity inputs"]
    if record["token_budget"] is not None:
        problems += [f"{key}: {role} over budget or truncated" for role in ("source", "candidate")
                     if record[role]["n_over_budget"] or record[role]["n_truncated"]]
    return problems


def compare(old_root: Path, new_root: Path) -> tuple:
    """
    Compare every new result with its stored counterpart.

    Parameters
    ----------
    old_root, new_root : Path
        Stored and new results trees.

    Returns
    -------
    tuple
        Rows (one per cell) and the list of failures.
    """
    old_files, rows, failures = result_files(old_root), [], []
    for key, path in result_files(new_root).items():
        new, reason = json.loads(path.read_text()), expected_change(key[0], key[1])
        if key not in old_files:
            failures.append(f"{key}: no stored result to compare with")
            continue
        diff = differences(json.loads(old_files[key].read_text()), new)
        if diff and reason is None:
            failures.append(f"{key}: must reproduce but differs in {sorted(diff)}")
        if key[0] != "B0":
            failures += guarantee_problems(key, new, path)
        rows.append({"key": key, "reason": reason, "diff": diff, "new": new})
    return rows, failures


def c19_problems(rows: List[dict]) -> List[str]:
    """
    Check that C19 equals C10 on every pair where both ran.

    Parameters
    ----------
    rows : List[dict]
        Rows from :func:`compare`.

    Returns
    -------
    List[str]
        A problem per differing pair, or one if no pair has both.
    """
    by_key = {row["key"]: row["new"] for row in rows}
    pairs = [(c, k) for k in by_key if k[0] == "C19" and (c := ("C10", k[1], k[2])) in by_key]
    if not pairs:
        return ["C19 = C10: no pair where both ran"]
    return [f"C19 differs from C10 on {k[1:]}" for c, k in pairs
            if any(by_key[c][b] != by_key[k][b] for b in COMPARED)]


def _row_line(row: dict) -> str:
    condition, column, pair = row["key"]
    metrics = row["new"]["metrics"]
    old_mrr = row["diff"].get("mrr", (metrics["mrr"],))[0]
    status = "identical" if not row["diff"] else f"changed ({len(row['diff'])} values)"
    budget = (row["new"].get("encoder_input") or {}).get("token_budget")
    return (f"| {condition} | {column} | {pair} | {row['reason'] or '—'} | {status} | {old_mrr:.4f} "
            f"| {metrics['mrr']:.4f} | {metrics['mrr'] - old_mrr:+.4f} | {budget if budget is not None else '—'} |")


def render_markdown(rows: List[dict], failures: List[str], commits: List[str]) -> str:
    """
    Render the comparison and every check as Markdown.

    Parameters
    ----------
    rows : List[dict]
        Rows from :func:`compare`, B0 included.
    failures : List[str]
        Every failed check.
    commits : List[str]
        Distinct commits of the new results.

    Returns
    -------
    str
        The Markdown document.
    """
    lines = ["# #183 re-run against the stored results", "",
             f"New results from commit(s) {', '.join(c[:7] for c in commits)}. A cell *expected to change* is "
             "touched by a #183 correction; every other cell must reproduce every metric and count exactly. "
             "Budgeted results must report no text over the budget and none truncated; C19 must equal C10.", "",
             f"**Checks: {'all passed' if not failures else f'{len(failures)} failed'}.**", ""]
    lines += [f"- {failure}" for failure in failures] + ([""] if failures else [])
    lines += ["| Cond. | Column | Pair | Expected change | Result | MRR stored | MRR new | Δ | Budget |",
              "|---|---|---|---|---|---:|---:|---:|---:|"]
    return "\n".join(lines + [_row_line(row) for row in rows]) + "\n"


def main() -> None:
    """Compare the new trees with the stored ones and write the report."""
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old_dir", default="data/results/")
    parser.add_argument("--new_dir", required=True)
    parser.add_argument("--old_baseline_dir", default="data/results_ngram_baseline/")
    parser.add_argument("--new_baseline_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, failures = compare(Path(args.old_dir), Path(args.new_dir))
    b0_rows, b0_failures = compare(Path(args.old_baseline_dir), Path(args.new_baseline_dir))
    failures += b0_failures + c19_problems(rows)
    commits = sorted({row["new"].get("git_commit", "unknown") for row in rows + b0_rows})
    Path(args.output).write_text(render_markdown(b0_rows + rows, failures, commits))
    print("all checks passed" if not failures else "\n".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
