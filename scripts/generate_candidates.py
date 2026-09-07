"""Command-line entry point for offline stage-1 candidate generation.

Loads each requested OAEI dataset, extracts the source and target entity URIs
implied by the pair's entity type, scores them with character n-gram overlap,
and writes the per-source candidate lists that the Phase 2 embedding pipeline
reads back through ``kgsemembed.candidates.load_candidates``.

One dataset failing does not abort the run: the failure is logged and the
remaining datasets continue.

Usage
-----
python scripts/generate_candidates.py \\
    --datasets D3 D4 \\
    --data_dir data/
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdflib import Graph

from kgsemembed.candidates.ngram import (
    NgramConfig,
    build_ngram_index,
    dump_candidates,
    generate_candidates_for_pair,
    get_entity_label,
)
from kgsemembed.datasets import (
    AlignmentPair,
    extract_class_uris,
    extract_instance_uris,
    extract_predicate_uris,
    load_dataset,
)
from kgsemembed.utils.errors import DataError

_LOGGER = logging.getLogger("kgsemembed.scripts.generate_candidates")
_ALL_DATASETS = ("D1", "D2", "D3", "D4", "D5")


def _extract_class_uris(graph: Graph) -> List[str]:
    """Return sorted class URIs of *graph*."""
    return extract_class_uris(graph)


def _extract_predicate_uris(graph: Graph) -> List[str]:
    """Return sorted predicate URIs of *graph*."""
    return extract_predicate_uris(graph)


def _extract_instance_uris(graph: Graph) -> List[str]:
    """Return sorted instance URIs of *graph*."""
    return extract_instance_uris(graph)


def extract_entity_uris(graph: Graph, entity_type: str) -> List[str]:
    """
    Return the URIs of *graph* that match a pair-level entity type.

    Parameters
    ----------
    graph : Graph
        RDF graph to inspect.
    entity_type : str
        One of ``"class"``, ``"predicate"``, ``"instance"``, ``"mixed"``.

    Returns
    -------
    List[str]
        Sorted, deduplicated entity URIs.

    Raises
    ------
    DataError
        If ``entity_type`` is not a recognised descriptor.
    """
    if entity_type == "class":
        return _extract_class_uris(graph)
    if entity_type == "predicate":
        return _extract_predicate_uris(graph)
    if entity_type == "mixed":
        # Used for D3 Conference pairs and D4_schema — covers classes and predicates.
        # "mixed" is a pair-level descriptor only; individual entities stay class or predicate.
        return sorted(set(_extract_class_uris(graph)) | set(_extract_predicate_uris(graph)))
    if entity_type == "instance":
        return _extract_instance_uris(graph)
    raise DataError(f"Unknown entity type: {entity_type!r}.")


def _entity_labels(graph: Graph, uris: List[str]) -> Dict[str, str]:
    """Return a label for every URI, falling back to its local name."""
    return {uri: get_entity_label(graph, uri) for uri in uris}


def _candidate_map(pair: AlignmentPair, config: NgramConfig) -> Dict[str, List[str]]:
    """Rank targets for every source URI through the inverted n-gram index."""
    source_labels = _entity_labels(pair.source_graph, pair.source_entities)
    target_uris = extract_entity_uris(pair.target_graph, pair.entity_type)
    index = build_ngram_index(_entity_labels(pair.target_graph, target_uris), config)

    candidates: Dict[str, List[str]] = {}
    for scored in generate_candidates_for_pair(source_labels, index, config):
        candidates.setdefault(scored.source_uri, []).append(scored.target_uri)
    return candidates


def _write_candidates(pair: AlignmentPair, candidates: Dict[str, List[str]], data_dir: Path) -> Path:
    """Persist *candidates* to the location expected by ``load_candidates``."""
    path = data_dir / "candidates" / pair.dataset_id / f"{pair.pair_name}_candidates.json"
    dump_candidates(candidates, path)
    return path


def _process_dataset(dataset_id: str, args: argparse.Namespace) -> None:
    """Generate and persist candidates for every pair of one dataset."""
    data_dir = Path(args.data_dir)
    config = NgramConfig(n=args.n, metric=args.metric, top_k=args.top_k)
    for pair in load_dataset(dataset_id, data_dir):
        candidates = _candidate_map(pair, config)
        path = _write_candidates(pair, candidates, data_dir)
        _LOGGER.info(
            "%s/%s: %d source entities -> %s", dataset_id, pair.pair_name, len(candidates), path
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for candidate generation."""
    parser = argparse.ArgumentParser(description="Generate stage-1 n-gram candidate lists.")
    parser.add_argument(
        "--datasets", nargs="+", default=list(_ALL_DATASETS), help="Dataset identifiers to process."
    )
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--top_k", type=int, default=20, help="Candidates kept per source entity.")
    parser.add_argument("--n", type=int, default=3, help="Character n-gram size.")
    parser.add_argument(
        "--metric", default="cosine", choices=("cosine", "jaccard"), help="n-gram similarity metric."
    )
    return parser


def main() -> None:
    """Generate candidate files for every requested dataset."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    for dataset_id in args.datasets:
        try:
            _process_dataset(dataset_id, args)
        except Exception as exc:
            _LOGGER.warning("Skipping dataset %s: %s", dataset_id, exc)


if __name__ == "__main__":
    main()
