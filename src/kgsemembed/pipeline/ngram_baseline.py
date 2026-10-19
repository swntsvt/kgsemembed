"""Stage-1 character n-gram ranking baseline (B0) under the Phase 2 protocol.

Every Phase 2 condition re-ranks the stored top-20 n-gram candidate lists.
B0 scores those same lists in their stored n-gram order, so the difference
between a condition and B0 is what embedding re-ranking adds (or costs) on
identical candidates, populations, thresholds and metric definitions.

The n-gram scores are not stored in the candidate files, so they are recomputed
with the configuration that produced them (:data:`NGRAM_CONFIG`).  The
recomputed order must equal the stored order for every source; any difference
raises :class:`CandidateOrderMismatchError` before anything is written.
Evaluation is :func:`kgsemembed.pipeline.run_experiment._evaluate_pair`
unchanged: thresholds are tuned on the validation population only.

Results are written to ``{results_dir}/B0/{dataset_id}/{pair}_results.json``
with the C-condition schema (no model fields), beside a
``{pair}_entities.json`` sidecar holding each test reference source's best
gold rank.  B0 is not a registered condition and never writes into
``data/results/``.

Usage
-----
python -m kgsemembed.pipeline.ngram_baseline --datasets D1 D2 D3 D4 D5 \\
    --data_dir data/ --results_dir data/results_ngram_baseline/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kgsemembed.candidates.ngram import NgramConfig, load_candidates, score_pair_candidates
from kgsemembed.datasets import AlignmentPair, load_dataset
from kgsemembed.evaluation.metrics import EntityPair, RankedList, _reference_map, compute_recall_at_k
from kgsemembed.pipeline.run_experiment import (
    _COUNT_KEYS,
    _METRIC_KEYS,
    _PER_ENTITY_TYPE_KEY,
    _POPULATION_KEY,
    _candidate_count,
    _candidate_digest,
    _evaluate_pair,
    _provenance,
    _result_path,
    _seed_random_state,
)
from kgsemembed.utils.errors import DataError

CONDITION_ID = "B0"
STRATEGY = "ngram"
NGRAM_CONFIG = NgramConfig(n=3, metric="cosine", top_k=20)
DEFAULT_RESULTS_DIR = "data/results_ngram_baseline/"
ALL_DATASETS = ("D1", "D2", "D3", "D4", "D5")

_HISTORICAL_RESULTS_DIR = Path("data/results")
_LOGGER = logging.getLogger("kgsemembed.pipeline.ngram_baseline")

ScoredCandidates = Dict[str, List[Tuple[str, float]]]


class CandidateOrderMismatchError(DataError):
    """Recomputed n-gram ranking differs from the stored candidate lists."""


def check_results_dir(results_dir: str | Path) -> None:
    """
    Refuse to write B0 into the historical C1–C20 results tree.

    Parameters
    ----------
    results_dir : str | Path
        Requested output directory.

    Raises
    ------
    ValueError
        If ``results_dir`` resolves to ``data/results/``.
    """
    if Path(results_dir).resolve() == _HISTORICAL_RESULTS_DIR.resolve():
        raise ValueError(
            f"B0 results must not be written to {_HISTORICAL_RESULTS_DIR}; use {DEFAULT_RESULTS_DIR}."
        )


def scored_candidates(pair: AlignmentPair) -> ScoredCandidates:
    """
    Recompute every source's stage-1 candidates with their n-gram scores.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to score.

    Returns
    -------
    ScoredCandidates
        Mapping from source URI to ``(target URI, score)`` in n-gram rank order.
    """
    scored: ScoredCandidates = {}
    for candidate in score_pair_candidates(pair, NGRAM_CONFIG):
        scored.setdefault(candidate.source_uri, []).append((candidate.target_uri, candidate.score))
    return scored


def count_order_mismatches(scored: ScoredCandidates, stored: Dict[str, List[str]]) -> int:
    """
    Count sources whose recomputed candidate order differs from the stored one.

    A source present on one side only counts as a mismatch.

    Parameters
    ----------
    scored : ScoredCandidates
        Recomputed candidates with scores.
    stored : Dict[str, List[str]]
        Candidate lists read from the candidate file.

    Returns
    -------
    int
        Number of mismatching sources.
    """
    sources = set(scored) | set(stored)
    return sum(
        [target for target, _ in scored.get(source, [])] != stored.get(source) for source in sources
    )


def check_stored_order(scored: ScoredCandidates, stored: Dict[str, List[str]], label: str) -> None:
    """
    Raise unless the recomputed ranking reproduces the stored candidate lists.

    Parameters
    ----------
    scored : ScoredCandidates
        Recomputed candidates with scores.
    stored : Dict[str, List[str]]
        Candidate lists read from the candidate file.
    label : str
        Pair label used in the error message.

    Raises
    ------
    CandidateOrderMismatchError
        If any source's order differs or a source is missing on either side.
    """
    mismatches = count_order_mismatches(scored, stored)
    if mismatches:
        raise CandidateOrderMismatchError(
            f"{label}: recomputed n-gram order differs from the stored candidates "
            f"for {mismatches} of {len(stored)} sources."
        )


def build_ranked_lists(pair: AlignmentPair, scored: ScoredCandidates) -> List[RankedList]:
    """
    Turn the n-gram candidates into ranked lists, in source-entity order.

    Sources without candidates are skipped, as in the embedding pipeline.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair whose ``source_entities`` fix the list order.
    scored : ScoredCandidates
        Candidates with scores, already in n-gram rank order.

    Returns
    -------
    List[RankedList]
        One ``(source, target, score)`` list per source with candidates.
    """
    return [
        [(source, target, score) for target, score in scored[source]]
        for source in pair.source_entities
        if scored.get(source)
    ]


def gold_rank_records(ranked_lists: List[RankedList], references: List[EntityPair]) -> List[dict]:
    """
    Record the best gold rank of every test reference source.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        Ranked lists of the pair's sources.
    references : List[EntityPair]
        Test reference pairs.

    Returns
    -------
    List[dict]
        ``source``, ``ranked`` (whether the source has a ranked list),
        ``gold_targets``, ``gold_ranks`` (1-based rank of each gold
        target, ``None`` when it was not ranked) and ``best_gold_rank``
        (``None`` when no gold target was ranked), sorted by source URI.
    """
    ranked = {lst[0][0]: [target for _, target, _ in lst] for lst in ranked_lists if lst}
    records = []
    for source, gold in sorted(_reference_map(references).items()):
        positions = {target: rank for rank, target in enumerate(ranked.get(source, []), start=1)}
        ranks = [positions.get(target) for target in gold]
        best = min((rank for rank in ranks if rank is not None), default=None)
        records.append({"source": source, "ranked": source in ranked, "gold_targets": gold,
                        "gold_ranks": ranks, "best_gold_rank": best})
    return records


def result_payload(pair: AlignmentPair, metrics: Dict[str, object], stored: Dict[str, List[str]]) -> dict:
    """
    Assemble a B0 result in the C-condition result schema.

    Parameters
    ----------
    pair : AlignmentPair
        Evaluated alignment pair.
    metrics : Dict[str, object]
        Output of ``_evaluate_pair``.
    stored : Dict[str, List[str]]
        Candidate lists the result was computed from.

    Returns
    -------
    dict
        JSON-serialisable result payload.
    """
    return {
        "condition_id": CONDITION_ID, "dataset_id": pair.dataset_id, "pair_name": pair.pair_name,
        "strategy": STRATEGY, "model_key": None, "model_id": None,
        "apply_ppas": False, "ppas_effective": False, "token_budget": None,
        "ngram_config": asdict(NGRAM_CONFIG), "n_order_mismatches": 0,
        "metrics": {key: metrics[key] for key in _METRIC_KEYS},
        "counts": {key: metrics[key] for key in _COUNT_KEYS},
        "per_entity_type": metrics.get(_PER_ENTITY_TYPE_KEY, {}),
        "population": metrics[_POPULATION_KEY],
        "n_source_entities": len(pair.source_entities),
        "n_candidates_per_entity": _candidate_count(stored),
        "candidates_sha256": _candidate_digest(stored),
        **_provenance(None),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_pair(pair: AlignmentPair, data_dir: str | Path) -> Tuple[dict, List[dict]]:
    """
    Score one pair's stored candidates in n-gram order under the protocol.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to evaluate.
    data_dir : str | Path
        Dataset and candidate root directory.

    Returns
    -------
    Tuple[dict, List[dict]]
        Result payload, with ``candidate_recall_at_20`` (the ceiling no
        re-ranking can exceed), and the test-source gold-rank records.
    """
    stored = load_candidates(pair.dataset_id, pair.pair_name, data_dir)
    scored = scored_candidates(pair)
    check_stored_order(scored, stored, f"{pair.dataset_id}/{pair.pair_name}")
    ranked_lists = build_ranked_lists(pair, scored)
    payload = result_payload(pair, _evaluate_pair(pair, ranked_lists), stored)
    payload["candidate_recall_at_20"] = compute_recall_at_k(ranked_lists, pair.test_refs, NGRAM_CONFIG.top_k)
    return payload, gold_rank_records(ranked_lists, pair.test_refs)


def run_pair(pair: AlignmentPair, data_dir: str | Path, results_dir: str | Path, force: bool) -> None:
    """
    Evaluate one pair and write its result and sidecar, skipping existing ones.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to evaluate.
    data_dir : str | Path
        Dataset and candidate root directory.
    results_dir : str | Path
        Output root; files go under ``B0/{dataset_id}/``.
    force : bool
        Overwrite an existing result when ``True``.
    """
    path = _result_path(results_dir, CONDITION_ID, pair.dataset_id, pair.pair_name)
    if path.exists() and not force:
        _LOGGER.info("Skipping existing result: %s", path)
        return
    payload, records = evaluate_pair(pair, data_dir)
    _write_json(path, payload)
    _write_json(path.with_name(f"{pair.pair_name}_entities.json"), {"test_sources": records})
    _LOGGER.info("%s/%s: MRR %.4f F1 %.4f", pair.dataset_id, pair.pair_name,
                 payload["metrics"]["mrr"], payload["metrics"]["f1"])


def run_baseline(dataset_ids: List[str], data_dir: str | Path, results_dir: str | Path,
                 force: bool = False) -> None:
    """
    Run B0 on every pair of the requested datasets.

    Parameters
    ----------
    dataset_ids : List[str]
        Dataset identifiers (``"D4"`` covers both D4 pairs).
    data_dir : str | Path
        Dataset and candidate root directory.
    results_dir : str | Path
        Output root, never ``data/results/``.
    force : bool
        Overwrite existing results when ``True``.
    """
    check_results_dir(results_dir)
    for dataset_id in dataset_ids:
        for pair in load_dataset(dataset_id, data_dir):
            run_pair(pair, data_dir, results_dir, force)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the stage-1 n-gram ranking baseline (B0).")
    parser.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), help="Dataset IDs.")
    parser.add_argument("--data_dir", default="data/", help="Dataset and candidate root.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Result output root.")
    parser.add_argument("--force_recompute", action="store_true", help="Overwrite existing results.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the baseline."""
    _seed_random_state()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    run_baseline(args.datasets, args.data_dir, args.results_dir, args.force_recompute)


if __name__ == "__main__":
    main()
