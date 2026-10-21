"""Command-line entry point for stage-1 candidate recall@k.

Loads each requested OAEI dataset together with its generated candidate file
and reports the fraction of held-out test references whose gold target appears
in the source entity's top-k candidates.  The result is the ceiling on recall
that no re-ranking stage can exceed.

Recall is computed with ``kgsemembed.evaluation.compute_recall_at_k``, the same
function the experiment pipeline uses, so the figures reported here are
directly comparable with the ``recall_at_k`` entries in the result JSON.

One dataset or pair failing does not abort the run: the failure is logged and
the remaining pairs continue.

Usage
-----
python scripts/compute_recall_at_20.py \\
    --datasets D1 D5 \\
    --data_dir data/
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from kgsemembed.candidates import CandidateMap, load_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.evaluation import RankedList, compute_recall_at_k
from kgsemembed.utils.errors import DataError

_LOGGER = logging.getLogger("kgsemembed.scripts.compute_recall_at_20")
_ALL_DATASETS = ("D1", "D2", "D3", "D4", "D5")


@dataclass(frozen=True)
class PairRecall:
    """
    Candidate recall@k for a single alignment pair.

    Attributes
    ----------
    dataset_id : str
        OAEI track identifier the pair was loaded under.
    pair_name : str
        Ontology pair name, e.g. ``"d1_snomed_fma"``.
    recall : float
        Fraction of test references recovered within the top-k candidates.
    n_refs : int
        Number of test references the recall was computed over.
    """

    dataset_id: str
    pair_name: str
    recall: float
    n_refs: int


def _as_ranked_lists(candidates: CandidateMap) -> List[RankedList]:
    """
    Convert a candidate map into the ranked-list form the metrics expect.

    Candidate files store rank by list position and carry no scores, so each
    position is given a descending placeholder score. ``compute_recall_at_k``
    reads order only and never compares score magnitudes.

    Parameters
    ----------
    candidates : CandidateMap
        Mapping from source URI to its ordered candidate target URIs.

    Returns
    -------
    List[RankedList]
        One ranked ``(source, target, score)`` list per source entity.
    """
    return [
        [(source, target, float(len(targets) - rank)) for rank, target in enumerate(targets)]
        for source, targets in candidates.items()
    ]


def _pair_recall(pair: AlignmentPair, data_dir: str, k: int) -> PairRecall:
    """
    Compute candidate recall@k for one alignment pair.

    Parameters
    ----------
    pair : AlignmentPair
        Loaded alignment pair supplying the test references.
    data_dir : str
        Dataset root directory holding ``candidates/``.
    k : int
        Number of leading candidates considered for each source entity.

    Returns
    -------
    PairRecall
        Recall and reference count for the pair.

    Raises
    ------
    DataError
        If the pair has no candidate file or the file is malformed.
    """
    candidates = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    recall = compute_recall_at_k(_as_ranked_lists(candidates), pair.test_refs, k)
    return PairRecall(pair.dataset_id, pair.pair_name, recall, len(pair.test_refs))


def _dataset_recalls(dataset_id: str, data_dir: str, k: int) -> List[PairRecall]:
    """
    Compute recall@k for every pair of one dataset, skipping unusable pairs.

    Parameters
    ----------
    dataset_id : str
        OAEI track identifier, one of ``"D1"`` through ``"D5"``.
    data_dir : str
        Dataset root directory.
    k : int
        Number of leading candidates considered for each source entity.

    Returns
    -------
    List[PairRecall]
        One record per pair whose candidate file could be read.
    """
    records: List[PairRecall] = []
    for pair in load_dataset(dataset_id, data_dir):
        try:
            records.append(_pair_recall(pair, data_dir, k))
        except DataError as exc:
            _LOGGER.warning("Skipping %s/%s: %s", pair.dataset_id, pair.pair_name, exc)
    return records


def _collect_recalls(datasets: List[str], args: argparse.Namespace) -> List[PairRecall]:
    """
    Gather recall records across datasets, skipping failures with a warning.

    Parameters
    ----------
    datasets : List[str]
        Dataset identifiers to process.
    args : argparse.Namespace
        Parsed command-line arguments supplying ``data_dir`` and ``k``.

    Returns
    -------
    List[PairRecall]
        Recall records in dataset and pair order.
    """
    records: List[PairRecall] = []
    for dataset_id in datasets:
        try:
            records.extend(_dataset_recalls(dataset_id, args.data_dir, args.k))
        except DataError as exc:
            _LOGGER.warning("Skipping dataset %s: %s", dataset_id, exc)
    return records


def _group_by_dataset(records: List[PairRecall]) -> Dict[str, List[PairRecall]]:
    """
    Group recall records by the dataset identifier they were loaded under.

    Parameters
    ----------
    records : List[PairRecall]
        Recall records to group.

    Returns
    -------
    Dict[str, List[PairRecall]]
        Mapping from dataset identifier to its records, in insertion order.
    """
    grouped: Dict[str, List[PairRecall]] = {}
    for record in records:
        grouped.setdefault(record.dataset_id, []).append(record)
    return grouped


def _micro_average(records: List[PairRecall]) -> float:
    """
    Pool references across pairs and return the reference-weighted recall.

    Parameters
    ----------
    records : List[PairRecall]
        Recall records of a single dataset.

    Returns
    -------
    float
        Recall over the pooled references, or ``0.0`` when there are none.
    """
    n_refs = sum(record.n_refs for record in records)
    if n_refs == 0:
        return 0.0
    return sum(record.recall * record.n_refs for record in records) / n_refs


def _print_pair_table(records: List[PairRecall], k: int) -> None:
    """
    Print one row per alignment pair.

    Parameters
    ----------
    records : List[PairRecall]
        Recall records to render.
    k : int
        Cut-off the recall was computed at, used in the column header.
    """
    header = f"{'Dataset':<13}{'Pair':<36}{'Refs':>7}{f'Recall@{k}':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for record in records:
        print(
            f"{record.dataset_id:<13}{record.pair_name:<36}"
            f"{record.n_refs:>7}{record.recall:>12.4f}"
        )


def _print_dataset_summary(records: List[PairRecall], k: int) -> None:
    """
    Print a pooled recall line per dataset, with the spread over its pairs.

    Parameters
    ----------
    records : List[PairRecall]
        Recall records across every processed dataset.
    k : int
        Cut-off the recall was computed at, used in the column header.
    """
    header = f"{'Dataset':<13}{'Pairs':>7}{'Refs':>9}{f'Recall@{k}':>12}{'Min':>9}{'Max':>9}"
    print(f"\n{header}")
    print("-" * len(header))
    for dataset_id, group in _group_by_dataset(records).items():
        recalls = [record.recall for record in group]
        print(
            f"{dataset_id:<13}{len(group):>7}{sum(r.n_refs for r in group):>9}"
            f"{_micro_average(group):>12.4f}{min(recalls):>9.4f}{max(recalls):>9.4f}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for candidate recall."""
    parser = argparse.ArgumentParser(
        description="Compute stage-1 candidate recall@k against the test splits."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(_ALL_DATASETS), help="Dataset identifiers to process."
    )
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--k", type=int, default=20, help="Candidate cut-off rank.")
    return parser


def main() -> None:
    """Load candidates and references, then report recall@k per pair and dataset."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()
    records = _collect_recalls(args.datasets, args)
    if not records:
        raise DataError("No candidate files could be read for the requested datasets.")
    _print_pair_table(records, args.k)
    _print_dataset_summary(records, args.k)


if __name__ == "__main__":
    main()
