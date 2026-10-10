"""Evaluation populations: which source entities each split is scored on.

Every ranked source belongs to exactly one population, so no source is seen
both while the threshold is tuned and when the test metrics are computed.

Assignment rules
----------------
* A source with a test reference is scored on test.
* Otherwise, a source with a validation reference is scored on validation.
* Otherwise, a source with a training reference is excluded from both: it is
  a known match, so counting its prediction as a false positive would be wrong.
* A source with no reference in any slice is *unmatched*.  Under the OAEI
  convention that the reference alignment is complete, predicting a target for
  it is a false positive.  Unmatched sources are shared between validation and
  test in proportion to the reference sources of each slice, using a stable
  ``zlib.crc32`` draw per URI, so tuning faces the same false-positive pressure
  that the test population does.
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
