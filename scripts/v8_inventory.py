"""Inventory of the V8 annotation-predicate exclusion (#145).

Loads only the dataset graphs and candidate files; no model or tokenizer is
loaded and no alignment is scored.  Entities are what the main pipeline
verbalises: each pair's source entities and the union of its candidate
targets, each with the entity type the pipeline gives it.

``dump`` records a digest and the word count of every entity's V8 text under
whichever ``kgsemembed`` is importable.  Run it once against the code before
the fix (``PYTHONPATH`` pointing at a checkout of the earlier commit) and once
against the current code, then ``compare`` the two dumps.  V2+V8 and
V2+V8+V7 embed the V8 text unchanged, so an entity whose V8 text is unchanged
keeps its text under every V8 composition.

``predicates`` classifies, under the current code, every predicate that
touches a verbalised entity in either direction: the exclusion family that
removes it, or ``kept``.  It also lists any predicate a graph declares an
``owl:AnnotationProperty`` that V8 keeps (expected: none).

Usage
-----
PYTHONPATH=<old checkout>/src python scripts/v8_inventory.py dump \\
    --datasets D1 D2 D3 D4 D5 --output old.json
python scripts/v8_inventory.py dump --datasets D1 D2 D3 D4 D5 --output new.json
python scripts/v8_inventory.py compare --old old.json --new new.json
python scripts/v8_inventory.py predicates --datasets D1 D2 D3 D4 D5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

import kgsemembed
from kgsemembed.candidates import load_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.pipeline.run_experiment import _entity_type_for
from kgsemembed.verbalisation.registry import build_verbaliser

_SIDES = ("source", "candidate")
_EXCLUDED_BEFORE_145 = frozenset({
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://purl.obolibrary.org/obo/IAO_0000115",
})
_ARTIFACTS = Path("artifacts/issue-145")


def side_entities(pair: AlignmentPair, data_dir: Path) -> Iterator[Tuple[str, Graph, List[str]]]:
    """Yield each side's name, graph and verbalised entity URIs."""
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    targets = sorted({uri for source in pair.source_entities for uri in candidates.get(source, [])})
    yield "source", pair.source_graph, list(pair.source_entities)
    yield "candidate", pair.target_graph, targets


def text_record(text: str) -> List[object]:
    """Return a short digest and the word count of one V8 text."""
    return [hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], len(text.split())]


def pair_records(pair: AlignmentPair, data_dir: Path) -> Dict[str, Dict[str, List[object]]]:
    """Map each side's URIs to their V8 text records."""
    verbaliser = build_verbaliser("V8", "M2")
    records: Dict[str, Dict[str, List[object]]] = {}
    for side, graph, uris in side_entities(pair, data_dir):
        records[side] = {
            uri: text_record(verbaliser.verbalise(graph, URIRef(uri), _entity_type_for(pair.entity_type, graph, uri)))
            for uri in uris
        }
    return records


def _code_identity() -> Dict[str, str]:
    """Describe which kgsemembed tree produced a dump."""
    package_dir = Path(kgsemembed.__file__).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(package_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {"package_dir": str(package_dir), "git_commit": commit}


def dump(dataset_ids: List[str], data_dir: Path, output: Path) -> None:
    """Write the V8 text records of every requested dataset's pairs to ``output``."""
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


def _side_counts(old: Dict[str, list], new: Dict[str, list]) -> List[int]:
    """Return entities, changed texts and old/new word totals of one side."""
    if set(old) != set(new):
        raise ValueError("Old and new dumps verbalise different entities.")
    changed = sum(old[uri][0] != new[uri][0] for uri in old)
    return [len(old), changed, sum(old[u][1] for u in old), sum(new[u][1] for u in new)]


def compare_counts(old: dict, new: dict) -> Dict[tuple, List[int]]:
    """Sum the per-side counts of every (column, side) over its pairs."""
    if set(old["pairs"]) != set(new["pairs"]):
        raise ValueError("Old and new dumps cover different pairs.")
    totals: Dict[tuple, List[int]] = {}
    for pair_key in sorted(old["pairs"]):
        for side in _SIDES:
            counts = _side_counts(old["pairs"][pair_key][side], new["pairs"][pair_key][side])
            key = (_column(pair_key), side)
            totals[key] = [a + b for a, b in zip(totals.get(key, [0, 0, 0, 0]), counts)]
    return totals


def _compare_row(key: tuple, counts: List[int]) -> str:
    """Format one inventory row."""
    n, changed, old_words, new_words = counts
    share = changed / n if n else 0.0
    return (f"| {key[0]} | {key[1]} | {n:,} | {changed:,} | {share:.1%} | "
            f"{old_words / max(n, 1):.1f} | {new_words / max(n, 1):.1f} |")


def compare(old_path: Path, new_path: Path, output: Path) -> None:
    """Write the old-vs-new V8 text inventory to ``output`` and print it."""
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (old_path, new_path))
    rows = [_compare_row(key, counts) for key, counts in compare_counts(old, new).items()]
    lines = [
        "# #145 V8 annotation-predicate exclusion: text inventory", "",
        f"- Before: `{old['code']['git_commit']}`; after: `{new['code']['git_commit']}`.",
        "- Entities: each pair's source entities and the union of its candidate targets, "
        "as the main pipeline verbalises them. D3 sums its 21 pairs.",
        "- Words: mean whitespace-separated words per V8 text.", "",
        "| dataset | side | entities | V8 text changed | share | words before | words after |",
        "|---|---|---|---|---|---|---|", *rows, "",
    ]
    _write(output, lines)


def entity_predicates(graph: Graph, uri: str) -> set[Tuple[str, str]]:
    """Return the (direction, predicate) pairs that touch one entity."""
    entity = URIRef(uri)
    outgoing = {("out", str(p)) for p in graph.predicates(subject=entity)}
    return outgoing | {("in", str(p)) for p in graph.predicates(object=entity)}


def declared_annotation_properties(graph: Graph) -> set[str]:
    """Return the predicates a graph declares ``owl:AnnotationProperty``."""
    return {str(p) for p in graph.subjects(RDF.type, OWL.AnnotationProperty) if isinstance(p, URIRef)}


def predicate_counts(dataset_ids: List[str], data_dir: Path) -> Tuple[Counter, set[str]]:
    """Count entities per (column, side, direction, predicate), and collect declared annotations."""
    counts: Counter = Counter()
    declared: set[str] = set()
    for dataset_id in dataset_ids:
        for pair in load_dataset(dataset_id, data_dir):
            for side, graph, uris in side_entities(pair, data_dir):
                declared |= declared_annotation_properties(graph)
                for uri in uris:
                    counts.update((pair.dataset_id, side, d, p) for d, p in entity_predicates(graph, uri))
            print(f"counted {pair.dataset_id}/{pair.pair_name}", flush=True)
    return counts, declared


def write_classification_csv(counts: Counter, output: Path) -> None:
    """Write one row per (column, side, direction, predicate) with its decision."""
    from kgsemembed.verbalisation.v8 import exclusion_family

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["dataset", "side", "direction", "predicate", "entities", "decision", "before_145"])
        for (column, side, direction, predicate), n in sorted(counts.items()):
            before = "excluded" if predicate in _EXCLUDED_BEFORE_145 else "kept"
            writer.writerow([column, side, direction, predicate, n, exclusion_family(predicate) or "kept", before])


def _excluded_rows(counts: Counter) -> List[str]:
    """Format the excluded predicates, one row per column, side and direction."""
    from kgsemembed.verbalisation.v8 import exclusion_family

    rows = []
    for (column, side, direction, predicate), n in sorted(counts.items()):
        family = exclusion_family(predicate)
        if family and family != "type_and_label":
            before = "excluded" if predicate in _EXCLUDED_BEFORE_145 else "kept"
            rows.append(f"| {column} | {side} | {direction} | `{predicate}` | {family} | {before} | {n:,} |")
    return rows


def _kept_rows(counts: Counter) -> List[str]:
    """Summarise the kept predicates per column and side."""
    from kgsemembed.verbalisation.v8 import exclusion_family

    kept: Dict[tuple, set[str]] = {}
    for (column, side, _direction, predicate) in counts:
        kept.setdefault((column, side), set())
        if exclusion_family(predicate) is None:
            kept[(column, side)].add(predicate)
    return [f"| {column} | {side} | {len(preds):,} |" for (column, side), preds in sorted(kept.items())]


def classification_lines(counts: Counter, declared: set[str]) -> List[str]:
    """Assemble the predicate classification report."""
    from kgsemembed.verbalisation.v8 import EXCLUDED_PREDICATES, v8_exclusions_digest

    seen = {predicate for (_c, _s, _d, predicate) in counts}
    leaked = sorted((declared & seen) - EXCLUDED_PREDICATES)
    return [
        "# #145 V8 predicate classification", "",
        f"- Exclusion families: `v8_exclusions_sha256` `{v8_exclusions_digest()[:12]}…` "
        "(`EXCLUSION_FAMILIES`, `verbalisation/v8.py`).",
        "- Entities: each pair's source entities and the union of its candidate targets; "
        "a count is the number of entities the predicate touches in that direction. D3 sums its 21 pairs.",
        "- Every predicate's decision, per dataset, side and direction: `predicate_classification.csv`.",
        f"- Declared `owl:AnnotationProperty` but kept by V8: {', '.join(f'`{p}`' for p in leaked) or 'none'}.",
        "", "## Excluded (annotation) predicates", "",
        "`rdf:type` and `rdfs:label` (family `type_and_label`, excluded before #145) are omitted.", "",
        "| dataset | side | direction | predicate | family | before #145 | entities |",
        "|---|---|---|---|---|---|---|", *_excluded_rows(counts),
        "", "## Kept predicates (relations and attributes)", "",
        "| dataset | side | distinct kept predicates |", "|---|---|---|", *_kept_rows(counts), "",
    ]


def predicates(dataset_ids: List[str], data_dir: Path, output_dir: Path) -> None:
    """Write the predicate classification report and CSV."""
    counts, declared = predicate_counts(dataset_ids, data_dir)
    write_classification_csv(counts, output_dir / "predicate_classification.csv")
    _write(output_dir / "predicate_classification.md", classification_lines(counts, declared))


def _write(output: Path, lines: List[str]) -> None:
    """Write markdown lines to ``output`` and print them."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory the V8 annotation-predicate exclusion.")
    commands = parser.add_subparsers(dest="command", required=True)
    dump_parser = commands.add_parser("dump", help="Record V8 texts of the importable code.")
    compare_parser = commands.add_parser("compare", help="Tabulate two dumps.")
    predicates_parser = commands.add_parser("predicates", help="Classify every predicate V8 sees.")
    for sub in (dump_parser, predicates_parser):
        sub.add_argument("--datasets", nargs="+", default=["D1", "D2", "D3", "D4", "D5"])
        sub.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    dump_parser.add_argument("--output", required=True, help="Dump file to write.")
    compare_parser.add_argument("--old", required=True, help="Dump of the code before the fix.")
    compare_parser.add_argument("--new", required=True, help="Dump of the code after the fix.")
    compare_parser.add_argument("--output", default=str(_ARTIFACTS / "v8_text_inventory.md"))
    predicates_parser.add_argument("--output_dir", default=str(_ARTIFACTS))
    return parser


def main() -> None:
    """Dump, compare or classify."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    if args.command == "dump":
        dump(args.datasets, Path(args.data_dir), Path(args.output))
    elif args.command == "compare":
        compare(Path(args.old), Path(args.new), Path(args.output))
    else:
        predicates(args.datasets, Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
