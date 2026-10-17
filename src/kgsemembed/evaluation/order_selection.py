"""Frozen selection rule and paired bootstraps of the serialization experiment (#134).

The rule chooses one serialization order from D2 validation sources before any
test rank exists.  It is a deterministic selection heuristic: given the
per-source reciprocal ranks it always returns the same order.  Its intervals
describe validation uncertainty only; they are not adjusted for selection and
support no inferential claim, and the chosen order's validation estimate is
optimistically biased by having been chosen.

Rule (fixed in #134 before any validation result)
-------------------------------------------------
1. One index matrix of :data:`SELECTION_REPLICATES` x n source indices, drawn
   uniformly with replacement from ``numpy.random.default_rng(SELECTION_SEED)``,
   is shared by every contrast.
2. ``best`` is the candidate with the highest MRR; an exact tie goes to the
   earlier candidate in :data:`PRECEDENCE`.
3. ``E`` holds ``best`` and every candidate whose 95% percentile interval of
   ``MRR(best) - MRR(o)`` has a lower bound <= 0: insufficient evidence of a
   difference under this procedure, not evidence of equivalence.
4. The frozen order is the first member of ``E`` in :data:`PRECEDENCE`.

Every other interval of the experiment uses :func:`paired_interval`: 2,000
replicates from a fresh ``default_rng(42)`` per contrast, as in #131.
"""

from typing import Dict, Tuple

import numpy as np

SELECTION_REPLICATES = 10_000
SELECTION_SEED = 20261016
REFERENCE_ORDER = "uri"
PRECEDENCE: Tuple[str, ...] = ("label", "core", "role")
"""Candidates by how much of the canonical text each moves: one predicate,
one role, every role."""

REPORT_REPLICATES = 2_000
REPORT_SEED = 42
_LOW, _HIGH = 0.025, 0.975


def index_matrix(n_sources: int, replicates: int, seed: int) -> np.ndarray:
    """
    Draw the bootstrap index matrix shared by a family of paired contrasts.

    Parameters
    ----------
    n_sources : int
        Number of paired units (reference sources).
    replicates : int
        Number of bootstrap replicates.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        ``(replicates, n_sources)`` indices drawn uniformly with replacement.
    """
    return np.random.default_rng(seed).integers(0, n_sources, size=(replicates, n_sources))


def interval(a: np.ndarray, b: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    """
    Return the paired mean difference ``a - b`` and its 95% percentile interval.

    Parameters
    ----------
    a, b : np.ndarray
        Per-source values aligned by source.
    indices : np.ndarray
        Bootstrap index matrix from :func:`index_matrix`.

    Returns
    -------
    Dict[str, float]
        ``estimate``, ``low`` and ``high`` (``numpy.quantile``, linear method).
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    means = diff[indices].mean(axis=1)
    return {"estimate": float(diff.mean()), "low": float(np.quantile(means, _LOW)),
            "high": float(np.quantile(means, _HIGH))}


def paired_interval(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Report-convention paired interval: 2,000 replicates, fresh seed 42.

    Parameters
    ----------
    a, b : np.ndarray
        Per-source values aligned by source.

    Returns
    -------
    Dict[str, float]
        ``estimate``, ``low``, ``high`` and ``n``.
    """
    n = len(a)
    if n == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    return {**interval(a, b, index_matrix(n, REPORT_REPLICATES, REPORT_SEED)), "n": n}


def _best(mrr: Dict[str, float]) -> str:
    """Return the highest-MRR candidate, exact ties to the earlier in precedence."""
    return min(PRECEDENCE, key=lambda o: (-mrr[o], PRECEDENCE.index(o)))


def choose_order(rr: Dict[str, np.ndarray]) -> Dict[str, object]:
    """
    Apply the frozen rule to per-source reciprocal ranks.

    Parameters
    ----------
    rr : Dict[str, np.ndarray]
        Reciprocal rank per source for ``uri`` and every candidate in
        :data:`PRECEDENCE`, aligned by source.

    Returns
    -------
    Dict[str, object]
        ``chosen``, ``best``, ``eligible`` (E, in precedence order), and per
        candidate its MRR, its difference from ``uri`` and from ``best``.
    """
    lengths = {len(v) for v in rr.values()}
    if len(lengths) != 1 or set(rr) != {REFERENCE_ORDER, *PRECEDENCE}:
        raise ValueError("Reciprocal ranks must cover uri and every candidate, aligned by source.")
    indices = index_matrix(lengths.pop(), SELECTION_REPLICATES, SELECTION_SEED)
    mrr = {order: float(np.mean(values)) for order, values in rr.items()}
    best = _best(mrr)
    versus_best = {o: interval(rr[best], rr[o], indices) for o in PRECEDENCE}
    eligible = [o for o in PRECEDENCE if o == best or versus_best[o]["low"] <= 0]
    candidates = {
        o: {"mrr": mrr[o], "minus_uri": interval(rr[o], rr[REFERENCE_ORDER], indices),
            "best_minus": versus_best[o], "in_E": o in eligible}
        for o in PRECEDENCE
    }
    return {"chosen": eligible[0], "best": best, "eligible": eligible, "uri_mrr": mrr[REFERENCE_ORDER],
            "candidates": candidates}
