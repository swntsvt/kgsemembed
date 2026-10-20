"""Per-source outcome records written beside every result (#140).

The paired significance tests compare two conditions source by source, so each
result carries a ``{pair}_entities.json`` sidecar with what every test source
contributed to its metrics:

- ``test_sources``: one record per test reference source, with its gold
  targets, the rank of each and the best gold rank, counted on the same
  judgeable lists as MRR (known pairs removed).  Its reciprocal best rank is
  the source's MRR term and ``best_gold_rank == 1`` its Hits@1 term.
- ``predictions``: the top-1 candidate of every ranked test-population source
  (reference and unmatched), its score and whether it was predicted (score at
  or above the tuned threshold) and correct.  With each reference source's
  gold count this decomposes TP, FP and FN by source, so F1 can be resampled.

:func:`per_source_record` checks that the predictions reproduce the result's
TP and FP counts before anything is written.
"""

from typing import Dict, List

from kgsemembed.evaluation.metrics import EntityPair, RankedList, _reference_map
from kgsemembed.utils.errors import PipelineStageError


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


def prediction_records(
    ranked_lists: List[RankedList], references: List[EntityPair], threshold: float
) -> List[dict]:
    """
    Record the top-1 candidate of every ranked source and how it was judged.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        Judgeable ranked lists of the test population.
    references : List[EntityPair]
        Test reference pairs.
    threshold : float
        Decision threshold tuned on the validation population.

    Returns
    -------
    List[dict]
        ``source``, ``target``, ``score``, ``predicted`` (score at or above
        ``threshold``) and ``correct`` (predicted and a reference pair),
        sorted by source URI.
    """
    reference_set = set(references)
    records = []
    for source, target, score in sorted(lst[0] for lst in ranked_lists if lst):
        predicted = score >= threshold
        records.append({"source": source, "target": target, "score": float(score),
                        "predicted": predicted,
                        "correct": predicted and (source, target) in reference_set})
    return records


def _check_counts(predictions: List[dict], metrics: Dict[str, object]) -> None:
    """Raise unless the prediction records reproduce the result's TP and FP."""
    tp = sum(1 for record in predictions if record["correct"])
    fp = sum(1 for record in predictions if record["predicted"] and not record["correct"])
    if (tp, fp) != (metrics["tp"], metrics["fp"]):
        raise PipelineStageError(
            f"Per-source predictions give TP {tp} / FP {fp}, the result "
            f"{metrics['tp']} / {metrics['fp']}."
        )


def per_source_record(
    test_lists: List[RankedList], references: List[EntityPair], metrics: Dict[str, object]
) -> dict:
    """
    Build the sidecar of one result from its test population.

    Parameters
    ----------
    test_lists : List[RankedList]
        Judgeable ranked lists of the test population, as metrics were
        computed from.
    references : List[EntityPair]
        Test reference pairs.
    metrics : Dict[str, object]
        The result's metrics, holding ``threshold``, ``tp`` and ``fp``.

    Returns
    -------
    dict
        ``threshold``, ``n_test_refs``, ``test_sources`` and ``predictions``.

    Raises
    ------
    PipelineStageError
        If the predictions do not reproduce the result's TP and FP counts.
    """
    predictions = prediction_records(test_lists, references, float(metrics["threshold"]))
    _check_counts(predictions, metrics)
    return {
        "threshold": float(metrics["threshold"]),
        "n_test_refs": len(set(references)),
        "test_sources": gold_rank_records(test_lists, references),
        "predictions": predictions,
    }
