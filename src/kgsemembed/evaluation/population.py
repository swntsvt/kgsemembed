"""Evaluation populations: which source entities each split is scored on.

Every ranked source belongs to exactly one population, so no source is seen
both while the threshold is tuned and when the test metrics are computed.

Assignment rules
----------------
* A source with a test reference is scored on test.
* Otherwise, a source with a validation reference is scored on validation.
* Otherwise, a source with a training reference is excluded from both: it is
  a known match, so counting its prediction as a false positive would be wrong.
  Training references are D5's OpenEA training links and D1's Bio-ML
  validation mappings (``input.rdf``).
* A known (training) pair is also removed from every ranked list, so for a
  scored source it is neither a distractor ranked above the gold target nor a
  prediction; Bio-ML's evaluation ignores these mappings in the same way.  See
  :func:`judgeable_ranked_lists`.
* A source with no reference in any slice is *unmatched*.  Under the OAEI
  convention that the reference alignment is complete, predicting a target for
  it is a false positive.  Unmatched sources are shared between validation and
  test in proportion to the reference sources of each slice, using a stable
  ``zlib.crc32`` draw per URI, so tuning faces the same false-positive pressure
  that the test population does.
* When the pair's reference is partial (``reference_complete`` is ``False``,
  the OAEI Knowledge Graph track), an unmatched source's prediction is judged
  only if its target appears in the reference, where it conflicts with a gold
  mapping and is a false positive; otherwise it is ignored, as the track's
  own evaluation does.  See :func:`judgeable_ranked_lists`.
"""

import zlib
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Set

from kgsemembed.datasets import AlignmentPair
from kgsemembed.evaluation.metrics import EntityPair, RankedList

_CRC32_RANGE = 2**32


@dataclass(frozen=True)
class EvaluationPopulations:
    """
    Source entities scored during threshold tuning and during test evaluation.

    Attributes
    ----------
    val_sources : FrozenSet[str]
        Validation reference sources plus their share of unmatched sources.
    test_sources : FrozenSet[str]
        Test reference sources plus their share of unmatched sources.
    n_unmatched_val : int
        Unmatched sources drawn into the validation population.
    n_unmatched_test : int
        Unmatched sources drawn into the test population.
    """

    val_sources: FrozenSet[str]
    test_sources: FrozenSet[str]
    n_unmatched_val: int
    n_unmatched_test: int


def _reference_sources(refs: Iterable[EntityPair]) -> Set[str]:
    """Return the distinct source URIs of ``refs``."""
    return {source for source, _ in refs}


def validation_share(n_val: int, n_test: int) -> float:
    """
    Return the fraction of reference sources that sit in the validation slice.

    Parameters
    ----------
    n_val : int
        Number of validation reference sources.
    n_test : int
        Number of test reference sources.

    Returns
    -------
    float
        ``n_val / (n_val + n_test)``, or ``0.0`` when both are zero.
    """
    total = n_val + n_test
    return n_val / total if total else 0.0


def draws_validation(uri: str, share: float) -> bool:
    """
    Decide, stably across processes, whether an unmatched source joins validation.

    Parameters
    ----------
    uri : str
        Source entity URI.
    share : float
        Probability of joining validation, in ``[0, 1]``.

    Returns
    -------
    bool
        ``True`` when the URI's ``crc32`` draw falls below ``share``.
    """
    return zlib.crc32(uri.encode("utf-8")) / _CRC32_RANGE < share


def build_populations(pair: AlignmentPair) -> EvaluationPopulations:
    """
    Assign every source entity of a pair to the validation or test population.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair with its train, validation, and test references.

    Returns
    -------
    EvaluationPopulations
        Disjoint validation and test source sets; see the module notes.
    """
    test = _reference_sources(pair.test_refs)
    val = _reference_sources(pair.val_refs) - test
    known = test | val | _reference_sources(pair.train_refs)
    share = validation_share(len(val), len(test))
    unmatched = [uri for uri in pair.source_entities if uri not in known]
    drawn_val = {uri for uri in unmatched if draws_validation(uri, share)}
    drawn_test = set(unmatched) - drawn_val
    return EvaluationPopulations(
        val_sources=frozenset(val | drawn_val),
        test_sources=frozenset(test | drawn_test),
        n_unmatched_val=len(drawn_val),
        n_unmatched_test=len(drawn_test),
    )


def restrict_ranked_lists(
    ranked_lists: List[RankedList], sources: FrozenSet[str]
) -> List[RankedList]:
    """
    Keep the ranked lists whose source belongs to ``sources``.

    Parameters
    ----------
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.
    sources : FrozenSet[str]
        Source URIs of the population being scored.

    Returns
    -------
    List[RankedList]
        The non-empty ranked lists of the population, in their original order.
    """
    return [ranked for ranked in ranked_lists if ranked and ranked[0][0] in sources]



def without_known_pairs(
    pair: AlignmentPair, ranked_lists: List[RankedList]
) -> List[RankedList]:
    """
    Remove every known ``(source, target)`` training pair from the ranked lists.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair supplying ``train_refs``.
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity.

    Returns
    -------
    List[RankedList]
        The lists without known pairs; ``ranked_lists`` itself when the pair
        has none.
    """
    known = set(pair.train_refs)
    if not known:
        return ranked_lists
    return [[entry for entry in ranked if entry[:2] not in known] for ranked in ranked_lists]


def judgeable_ranked_lists(
    pair: AlignmentPair, ranked_lists: List[RankedList]
) -> List[RankedList]:
    """
    Drop the candidates and predictions the reference alignment cannot judge.

    Known training pairs are first removed from every list: they are correct,
    so they can be neither a distractor nor a false positive.  Pairs without
    training references are unaffected.  Then, under a complete reference,
    every ranked list is kept.  Under a partial one, an unreferenced source's
    list is kept only when its top-ranked target is a reference target, i.e.
    its prediction contradicts a gold mapping.  Reference sources are always
    kept.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair supplying the references and ``reference_complete``.
    ranked_lists : List[RankedList]
        One ranked candidate list per source entity, best candidate first.

    Returns
    -------
    List[RankedList]
        The ranked lists, without known pairs, whose predictions the
        reference can judge.
    """
    ranked_lists = without_known_pairs(pair, ranked_lists)
    if pair.reference_complete:
        return ranked_lists
    refs = [*pair.train_refs, *pair.val_refs, *pair.test_refs]
    sources = _reference_sources(refs)
    targets = {target for _, target in refs}
    return [
        ranked for ranked in ranked_lists
        if ranked and (ranked[0][0] in sources or ranked[0][1] in targets)
    ]
