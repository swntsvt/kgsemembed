"""Old-vs-new inventory of the V1/V2 annotation predicate fix (#136).

Loads only the dataset graphs and candidate files; no model or tokenizer is
loaded and no alignment is scored.  ``dump`` records, for every entity the
main pipeline verbalises (each pair's source entities and the union of its
candidate targets), a digest and the word count of its V1 and V2 text under
whichever ``kgsemembed`` is importable.  Run it once against the code before
the fix (``PYTHONPATH`` pointing at a checkout of the earlier commit) and once
against the current code, then ``compare`` the two dumps.

V2+V6, V2+V7, V2+V8 and V2+V8+V7 embed the V2 text unchanged, so an entity
whose V2 text is unchanged keeps its text under every V2 composition.

Usage
-----
PYTHONPATH=<old checkout>/src python scripts/annotation_inventory.py dump \\
    --datasets D1 D2 D3 D4 D5 --output old.json
python scripts/annotation_inventory.py dump --datasets D1 D2 D3 D4 D5 --output new.json
python scripts/annotation_inventory.py compare --old old.json --new new.json \\
    --output artifacts/issue-136/annotation_inventory.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdflib import Graph, URIRef

import kgsemembed
from kgsemembed.candidates import load_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.verbalisation.registry import build_verbaliser

_STRATEGIES = ("V1", "V2")
_SIDES = ("source", "candidate")


def text_record(texts: List[str]) -> List[object]:
    """Return a short digest and the word count of each strategy's text."""
    record: List[object] = []
    for text in texts:
        record += [hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], len(text.split())]
    return record


def side_records(graph: Graph, uris: List[str]) -> Dict[str, List[object]]:
    """Map each URI to its V1/V2 text record on one side of a pair."""
    verbalisers = [build_verbaliser(name, "M2") for name in _STRATEGIES]
    return {
        uri: text_record([v.verbalise(graph, URIRef(uri), "class") for v in verbalisers])
        for uri in uris
    }


def pair_records(pair: AlignmentPair, data_dir: Path) -> Dict[str, Dict[str, List[object]]]:
    """Return the source and candidate records of one alignment pair."""
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    targets = sorted({uri for source in pair.source_entities for uri in candidates.get(source, [])})
    return {
        "source": side_records(pair.source_graph, pair.source_entities),
        "candidate": side_records(pair.target_graph, targets),
    }


def _code_identity() -> Dict[str, str]:
    """Describe which kgsemembed tree produced a dump."""
    package_dir = Path(kgsemembed.__file__).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(package_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"package_dir": str(package_dir), "git_commit": commit}


def dump(dataset_ids: List[str], data_dir: Path, output: Path) -> None:
    """Write the text records of every requested dataset's pairs to ``output``."""
    pairs: Dict[str, Dict[str, Dict[str, List[object]]]] = {}
    for dataset_id in dataset_ids:
        for pair in load_dataset(dataset_id, data_dir):
            pairs[f"{pair.dataset_id}/{pair.pair_name}"] = pair_records(pair, data_dir)
            print(f"dumped {pair.dataset_id}/{pair.pair_name}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"code": _code_identity(), "pairs": pairs}), encoding="utf-8")


def _column(pair_key: str) -> str:
    """Return the dataset column of a pair key (all D3 pairs share one)."""
    return pair_key.split("/", 1)[0]


def _side_counts(old: Dict[str, list], new: Dict[str, list], index: int) -> List[int]:
    """Return entities, changed texts and old/new word totals for one strategy."""
    if set(old) != set(new):
        raise ValueError("Old and new dumps verbalise different entities.")
    digest, words = 2 * index, 2 * index + 1
    changed = sum(old[uri][digest] != new[uri][digest] for uri in old)
    return [len(old), changed, sum(old[u][words] for u in old), sum(new[u][words] for u in new)]


def compare_counts(old: dict, new: dict) -> Dict[tuple, List[int]]:
    """Sum the per-side counts of every (column, side, strategy) over its pairs."""
    if set(old["pairs"]) != set(new["pairs"]):
        raise ValueError("Old and new dumps cover different pairs.")
    totals: Dict[tuple, List[int]] = {}
    for pair_key in sorted(old["pairs"]):
        for side in _SIDES:
            for index, strategy in enumerate(_STRATEGIES):
                counts = _side_counts(old["pairs"][pair_key][side], new["pairs"][pair_key][side], index)
                key = (_column(pair_key), side, strategy)
                totals[key] = [a + b for a, b in zip(totals.get(key, [0, 0, 0, 0]), counts)]
    return totals


def _row(key: tuple, counts: List[int]) -> str:
    """Format one inventory row."""
    column, side, strategy = key
    n, changed, old_words, new_words = counts
    share = changed / n if n else 0.0
    return (f"| {column} | {side} | {strategy} | {n:,} | {changed:,} | {share:.1%} | "
            f"{old_words / max(n, 1):.1f} | {new_words / max(n, 1):.1f} |")


def compare(old_path: Path, new_path: Path, output: Path) -> None:
    """Write the old-vs-new inventory table to ``output`` and print it."""
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (old_path, new_path))
    rows = [_row(key, counts) for key, counts in compare_counts(old, new).items()]
    lines = [
        "# #136 annotation predicate fix: V1/V2 text inventory", "",
        f"- Before: `{old['code']['git_commit']}`; after: `{new['code']['git_commit']}` "
        "(plus the uncommitted fix, if the tree was dirty).",
        "- Entities: each pair's source entities and the union of its candidate targets, "
        "as the main pipeline verbalises them. D3 sums its 21 pairs.",
        "- Words: mean whitespace-separated words per text.", "",
        "| dataset | side | strategy | entities | text changed | share | words before | words after |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory the V1/V2 annotation predicate fix.")
    commands = parser.add_subparsers(dest="command", required=True)
    dump_parser = commands.add_parser("dump", help="Record V1/V2 texts of the importable code.")
    dump_parser.add_argument("--datasets", nargs="+", default=["D1", "D2", "D3", "D4", "D5"])
    dump_parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    dump_parser.add_argument("--output", required=True, help="Dump file to write.")
    compare_parser = commands.add_parser("compare", help="Tabulate two dumps.")
    compare_parser.add_argument("--old", required=True, help="Dump of the code before the fix.")
    compare_parser.add_argument("--new", required=True, help="Dump of the code after the fix.")
    compare_parser.add_argument("--output", default="artifacts/issue-136/annotation_inventory.md")
    return parser


def main() -> None:
    """Dump or compare V1/V2 text records."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    if args.command == "dump":
        dump(args.datasets, Path(args.data_dir), Path(args.output))
    else:
        compare(Path(args.old), Path(args.new), Path(args.output))


if __name__ == "__main__":
    main()
