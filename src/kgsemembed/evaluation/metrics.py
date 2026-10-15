"""Phase 2 evaluation metrics for knowledge graph matching.

Central location for the in-memory metrics used to evaluate scored candidate
pairs and ranked candidate lists.  The module provides threshold-based
precision/recall/F1, automatic threshold tuning, Mean Reciprocal Rank,
Recall@k, and a combined evaluation summary.

Evaluation protocol
-------------------
Each source predicts at most one target: its best-ranked candidate, kept when
its score reaches the decision threshold.  This matches the BERTMapLt baseline,
which emits one mapping per source, and keeps precision from being swamped by
the lower-ranked candidates of every source.  Ranking metrics are averaged over
reference sources only: a source without a gold target has no rank to recover,
while a reference source the candidate stage missed contributes zero.

URIs are compared using exact string equality only; no normalisation is
applied.  The module performs no file I/O and no statistical significance
testing, and every returned floating-point metric lies within ``[0.0, 1.0]``.
"""

from typing import Dict, List, Optional, Tuple

EVALUATION_PROTOCOL_VERSION = 2

EntityPair = Tuple[str, str]
ScoredPair = Tuple[str, str, float]
RankedList = List[ScoredPair]


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Divide ``numerator`` by ``denominator``, returning ``0.0`` when undefined.

    Parameters
    ----------
    numerator : float
        Dividend.
    denominator : float
        Divisor; a value of zero yields ``0.0`` instead of raising.

    Returns
    -------
    float
        The quotient, or ``0.0`` when ``denominator`` is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_f1_at_threshold(
    scored_pairs: List[ScoredPair],
    references: List[EntityPair],
    threshold: float,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for a fixed decision threshold.

    Every scored pair whose score is greater than or equal to ``threshold`` is
    treated as a predicted match.  Predictions and references are compared as
    sets of entity pairs using exact string equality, so duplicate scored pairs
    collapse to a single prediction.

    Parameters
    ----------
    scored_pairs : List[ScoredPair]
        Candidate pairs as ``(source, target, score)`` triples.
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.
    threshold : float
        Minimum score for a pair to count as a predicted match.

    Returns
    -------
    Dict[str, float]
        Mapping with ``precision``, ``recall``, ``f1``, ``threshold``, and the
        integer counts ``tp``, ``fp``, and ``fn``.
    """
    reference_set = set(references)
    predicted_set = {
        (source, target)
        for source, target, score in scored_pairs
        if score >= threshold
    }
    tp = len(predicted_set & reference_set)
    fp = len(predicted_set - reference_set)
    fn = len(reference_set - predicted_set)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


_GRID_START = 0.10
_GRID_STEP = 0.01
_GRID_SIZE = 90


def _default_thresholds() -> List[float]:
    """
    Build the default tuning grid from ``0.10`` to ``0.99`` in steps of ``0.01``.

    The grid reaches ``0.99`` because top-1 cosine scores of the larger models
    cluster above ``0.9``; a grid capped at ``0.95`` left the tuned threshold
    pinned at its upper edge.  Values are rounded to two decimals so a tuned
    threshold reproduces its reported metrics exactly.

    Returns
    -------
    List[float]
        Candidate thresholds ``[0.10, 0.11, ..., 0.99]``.
    """
    return [round(_GRID_START + _GRID_STEP * step, 2) for step in range(_GRID_SIZE)]


def _median_threshold(thresholds: List[float]) -> float:
    """
    Return the lower median of thresholds that tie for the best F1.

    Tied thresholds form a plateau when validation references are few, which
    is the norm on D3.  Taking its middle rather than its lowest end keeps the
    choice away from the plateau edge, where a test score just beside the
    validation scores would flip a prediction.

    Parameters
    ----------
    thresholds : List[float]
        Tied thresholds in ascending grid order; must be non-empty.

    Returns
    -------
    float
        The element at position ``(len - 1) // 2``.
    """
    return float(thresholds[(len(thresholds) - 1) // 2])


def tune_threshold(
    scored_pairs: List[ScoredPair],
    references: List[EntityPair],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Select the threshold that maximises F1 over a grid of candidate values.

    Parameters
    ----------
    scored_pairs : List[ScoredPair]
        Candidate pairs as ``(source, target, score)`` triples.
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.
    thresholds : Optional[List[float]]
        Candidate thresholds to evaluate, in ascending order.  Defaults to the
        grid ``[0.10, 0.11, ..., 0.99]`` when ``None``.

    Returns
    -------
    Dict[str, float]
        Mapping with ``best_threshold``, ``best_f1``, ``precision``, and
        ``recall`` for the winning threshold.

    Raises
    ------
    ValueError
        If ``thresholds`` is an empty sequence.

    Notes
    -----
    Threshold tuning is appropriate only for validation/development datasets.
    A tuned threshold must never be derived from the test dataset; held-out
    test evaluation must always use a fixed threshold selected independently.
    When several thresholds achieve the same maximum F1, the lower median of
    the tied thresholds, in grid order, is chosen to keep the result
    deterministic and away from the edge of the plateau.
    """
    grid = _default_thresholds() if thresholds is None else list(thresholds)
    if not grid:
        raise ValueError("Threshold grid must contain at least one value.")
    results = [compute_f1_at_threshold(scored_pairs, references, t) for t in grid]
    best_f1 = max(result["f1"] for result in results)
    tied = [t for t, result in zip(grid, results) if result["f1"] == best_f1]
    best = results[grid.index(_median_threshold(tied))]
    return {
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
    }


def _reference_map(references: List[EntityPair]) -> Dict[str, List[str]]:
    """
    Map each source URI to every gold target aligned with it.

    A source may carry several gold targets under a 1:N reference alignment, so
    all of them are retained in first-occurrence order rather than collapsing
    to the first.

    Parameters
    ----------
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from source URI to its gold target URIs.
    """
    mapping: Dict[str, List[str]] = {}
    for source, target in references:
        targets = mapping.setdefault(source, [])
        if target not in targets:
            targets.append(target)
    return mapping


def _reciprocal_rank(
    ranked_list: RankedList, reference_map: Dict[str, List[str]]
) -> float:
    """
    Compute the reciprocal rank of a source's best-ranked gold target.

    Under a 1:N alignment the source is credited with its highest-ranked gold
    target, so recovering any one of several equivalent targets at rank one
    scores the same as an unambiguous 1:1 match.

    Parameters
    ----------
    ranked_list : RankedList
        Ranked candidate pairs for a single source entity.
    reference_map : Dict[str, List[str]]
        Mapping from source URI to its gold target URIs.

    Returns
    -------
    float
        ``1 / rank`` of the earliest gold target, or ``0.0`` when the source has
        no reference or none of its gold targets appear in the ranked list.
    """
    if not ranked_list:
        return 0.0
    targets = reference_map.get(ranked_list[0][0])
    if not targets:
        return 0.0
    for rank, (_, candidate, _) in enumerate(ranked_list, start=1):
        if candidate in targets:
            return 1.0 / rank
    return 0.0


def compute_mrr(
    ranked_lists: List[RankedList],
    references: List[EntityPair],
) -> float:
    """
    Compute the Mean Reciprocal Rank over per-source ranked candidate lists.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.

    Returns
    -------
    float
        Arithmetic mean of the reciprocal ranks across every source carrying a
        reference, in ``[0.0, 1.0]``.  Each reference source contributes the
        reciprocal rank of its best-ranked gold target, or zero when it has no
        ranked list or none of its gold targets was ranked.  Ranked sources
        without a reference are not part of the mean, and ``0.0`` is returned
        when ``references`` is empty.
    """
    reference_map = _reference_map(references)
    if not reference_map:
        return 0.0
    ranked_by_source = {ranked[0][0]: ranked for ranked in ranked_lists if ranked}
    reciprocal_ranks = [
        _reciprocal_rank(ranked_by_source.get(source, []), reference_map)
        for source in reference_map
    ]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _ranked_targets_by_source(
    ranked_lists: List[RankedList],
) -> Dict[str, List[str]]:
    """
    Map each source URI to its ordered list of candidate target URIs.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from source URI to its ranked candidate target URIs.
    """
    mapping: Dict[str, List[str]] = {}
    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        source = ranked_list[0][0]
        mapping[source] = [target for _, target, _ in ranked_list]
    return mapping


def compute_recall_at_k(
    ranked_lists: List[RankedList],
    references: List[EntityPair],
    k: int,
) -> float:
    """
    Compute the fraction of references whose gold target appears in the top-k.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.
    k : int
        Number of leading candidates considered for each source.

    Returns
    -------
    float
        ``recovered / len(references)`` in ``[0.0, 1.0]``, or ``0.0`` when
        ``references`` is empty.
    """
    if not references:
        return 0.0
    ranked_by_source = _ranked_targets_by_source(ranked_lists)
    recovered = 0
    for source, target in references:
        if target in ranked_by_source.get(source, [])[:k]:
            recovered += 1
    return recovered / len(references)


def top_ranked_pairs(ranked_lists: List[RankedList]) -> List[ScoredPair]:
    """
    Keep each source's best-ranked candidate as its only prediction.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity, best candidate first.

    Returns
    -------
    List[ScoredPair]
        The first scored pair of every non-empty ranked list.
    """
    return [ranked_list[0] for ranked_list in ranked_lists if ranked_list]


def compute_all_metrics(
    ranked_lists: List[RankedList],
    references: List[EntityPair],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute the full Phase 2 evaluation summary for ranked candidate lists.

    When ``threshold`` is ``None`` the threshold is tuned on the supplied data;
    otherwise the supplied threshold is used directly.  Threshold-based metrics
    operate on each source's top-ranked candidate (see the module notes), while
    the ranking metrics operate on the full per-source ranked lists.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.
    references : List[EntityPair]
        Ground-truth ``(source, target)`` pairs.
    threshold : Optional[float]
        Fixed decision threshold, or ``None`` to tune on the supplied data.

    Returns
    -------
    Dict[str, float]
        Mapping with ``f1``, ``precision``, ``recall``, ``threshold``, ``mrr``,
        ``recall_at_1``, ``recall_at_5``, ``recall_at_10``, and the integer
        counts ``tp``, ``fp``, and ``fn`` behind the threshold metrics.
    """
    scored_pairs = top_ranked_pairs(ranked_lists)
    if threshold is None:
        threshold = tune_threshold(scored_pairs, references)["best_threshold"]
    threshold_metrics = compute_f1_at_threshold(scored_pairs, references, threshold)
    return {
        "f1": threshold_metrics["f1"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "threshold": threshold_metrics["threshold"],
        "mrr": compute_mrr(ranked_lists, references),
        "recall_at_1": compute_recall_at_k(ranked_lists, references, 1),
        "recall_at_5": compute_recall_at_k(ranked_lists, references, 5),
        "recall_at_10": compute_recall_at_k(ranked_lists, references, 10),
        "tp": threshold_metrics["tp"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
    }
