"""Inventory of what the #183 corrections change, cell by cell, before the re-run.

For every condition and column it lists the corrections that touch the cell
(``rerun_comparison.expected_change``), and from the two tokenizer
inventories, before #183 (``artifacts/issue-181/``) and after it
(``artifacts/issue-183/``), the encoded tokens and the texts the encoder
truncates or a budget would exceed.  Where the D2 dry run covered the cell,
its MRR against the stored result is shown.

Usage
-----
python scripts/change_inventory.py --dryrun artifacts/issue-183/dryrun_comparison.md \\
    --output artifacts/issue-183/change_inventory.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerun_comparison import expected_change  # noqa: E402

_DRYRUN_ROW = re.compile(r"^\| (\w+) \| (\w+) \| [^|]+ \| [^|]+ \| ([^|]+) \| ([\d.]+) \| ([\d.]+) \|")


def inventory(path: str) -> Dict[tuple, dict]:
    """
    Read a tokenizer inventory keyed by condition and column.

    Parameters
    ----------
    path : str
        ``encoded_input_size.json``.

    Returns
    -------
    Dict[tuple, dict]
        ``(condition, column)`` to its record.
    """
    return {(r["condition_id"], r["column"]): r for r in json.loads(Path(path).read_text())}


def dryrun_rows(path: str) -> Dict[tuple, str]:
    """
    Read the dry run's MRR per condition and column from its Markdown report.

    Parameters
    ----------
    path : str
        ``dryrun_comparison.md``.

    Returns
    -------
    Dict[tuple, str]
        ``(condition, column)`` to ``"stored → new (status)"``.
    """
    rows = {}
    for line in Path(path).read_text().splitlines():
        if match := _DRYRUN_ROW.match(line):
            condition, column, status, old, new = match.groups()
            rows[(condition, column)] = f"{old} → {new} ({status.strip()})"
    return rows


def _line(key: tuple, before: dict, after: dict, dry: str) -> str:
    b, a = before["total"], after["total"]
    over = "—" if after["token_budget"] is None else f"{a['n_over_budget']:,}"
    return (f"| {key[0]} | {key[1]} | {after['strategy']} | {after['model_key']} | {expected_change(*key) or '—'} "
            f"| {b['encoded_tokens'] / 1e6:.2f} → {a['encoded_tokens'] / 1e6:.2f} "
            f"| {b['n_over_limit']:,} / {before['max_seq_length']} → {a['n_over_limit']:,} / {after['max_seq_length']} "
            f"| {over} | {dry} |")


def render(before: Dict[tuple, dict], after: Dict[tuple, dict], dry: Dict[tuple, str]) -> List[str]:
    """
    Render one table row per condition and column, affected cells first.

    Parameters
    ----------
    before, after : Dict[tuple, dict]
        Inventories before and after #183.
    dry : Dict[tuple, str]
        Dry-run outcomes.

    Returns
    -------
    List[str]
        Table rows.
    """
    keys = sorted(after, key=lambda k: (expected_change(*k) is None, k[1], int(k[0][1:])))
    return [_line(key, before[key], after[key], dry.get(key, "")) for key in keys]


def main() -> None:
    """Write the change inventory."""
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", default="artifacts/issue-181/encoded_input_size.json")
    parser.add_argument("--after", default="artifacts/issue-183/encoded_input_size.json")
    parser.add_argument("--dryrun", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    before, after = inventory(args.before), inventory(args.after)
    affected = sum(expected_change(*key) is not None for key in after)
    header = [
        "# #183 change inventory", "",
        f"{affected} of {len(after)} condition × column cells are touched by a #183 correction (C19 is not "
        "listed: it equals C10). *Tokens*: encoded tokens in millions, before #183 (`artifacts/issue-181/`) "
        "→ after (`artifacts/issue-183/`). *Truncated*: texts over the encoder's limit / the limit. *Over "
        "budget*: budgeted texts above their budget after #183. *Dry run*: MRR stored → new on D2 (and C10 "
        "on D1); every untouched cell must reproduce exactly in the re-run.", "",
        "| Cond. | Column | Strategy | Model | Correction | Tokens (M) | Truncated / limit | Over budget | Dry run |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    Path(args.output).write_text("\n".join(header + render(before, after, dryrun_rows(args.dryrun))) + "\n")


if __name__ == "__main__":
    main()
