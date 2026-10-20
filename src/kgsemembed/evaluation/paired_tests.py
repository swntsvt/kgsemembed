"""Per-dataset paired tests of the registered C1-C20 comparisons (#140).

The dataset-level Wilcoxon in :mod:`kgsemembed.evaluation.stats` has at most
six units, so no comparison outside group E can be significant.  This module
tests each registered comparison within each dataset column instead, over the
test sources recorded in the per-source sidecars
(:mod:`kgsemembed.evaluation.per_source`).

Method
------
- **Unit**: a test reference source for MRR and Hits@1 (a source with several
  gold targets counts once, at its best gold rank).  For F1, every test
  source that contributes to TP, FP or FN: the test reference sources plus the
  unmatched sources either condition made a judgeable prediction for.  Each
  unit carries its TP, FP and FN, and F1 is recomputed from their sums.
- **Statistic**: the column value of B minus that of A (positive when B is
  better), as the results table reports it: the metric over the column's one
  pair, or for D3 the mean of the 21 pairs' values.
- **Interval**: percentile bootstrap, resampling units with replacement within
  each pair (D3 pairs are strata, never pooled), :data:`N_REPLICATES`
  replicates, :data:`SEED`.
- **p-value**: paired randomisation test, swapping the two conditions'
  outcomes of each unit with probability 1/2 (within each pair), two-sided,
  ``(1 + #{|Δ*| >= |Δ|}) / (R + 1)`` with the same replicate count and seed.
- **Multiplicity**: Holm within each family of one ablation group, one dataset
  column and one metric (:func:`holm_adjust`).  MRR and Hits@1 are primary;
  F1 is secondary.  Intervals are not adjusted.

Every result also carries the column's top-20 ceiling, the most the metric
can reach on the stored candidate lists.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from kgsemembed.evaluation.stats import expand_dataset_ids, registered_comparisons
from kgsemembed.utils.errors import DataError

N_REPLICATES = 10_000
SEED = 42
ALPHA = 0.05
CONFIDENCE = 0.95
PRIMARY_METRICS = ("mrr", "hits_at_1")
SECONDARY_METRICS = ("f1",)
METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS)
CEILING_KEYS = {"mrr": "ranking_ceiling", "hits_at_1": "ranking_ceiling", "f1": "f1_recall_ceiling"}

_RESULT_SUFFIX = "_results.json"
_SIDECAR_SUFFIX = "_entities.json"
_CHUNK = 100
_TOLERANCE = 1e-12

Values = Dict[str, Tuple[float, ...]]
PairArrays = Tuple[np.ndarray, np.ndarray]


class MissingSidecarError(DataError):
    """A result has no per-source sidecar, so it cannot be tested source by source."""


@dataclass(frozen=True)
class TestSettings:
    """
    Resampling settings shared by every test of one run.

    Attributes
    ----------
    n_replicates : int
        Bootstrap and randomisation replicates.
    seed : int
        Seed of every test's generator, so each test is reproducible alone.
    """

    n_replicates: int = N_REPLICATES
    seed: int = SEED


def load_sidecars(results_dir: str | Path, condition_id: str, dataset_id: str) -> Dict[str, dict]:
    """
    Load the per-source sidecar of every result of a condition on a column.

    Parameters
    ----------
    results_dir : str | Path
        Root of the result tree.
    condition_id : str
        Condition, e.g. ``"C10"``.
    dataset_id : str
        Result column, e.g. ``"D4_schema"``.

    Returns
    -------
    Dict[str, dict]
        Sidecar by pair name.

    Raises
    ------
    MissingSidecarError
        If the column has no result, or a result has no current sidecar.
    """
    results = sorted((Path(results_dir) / condition_id / dataset_id).glob(f"*{_RESULT_SUFFIX}"))
    if not results:
        raise MissingSidecarError(f"No results for {condition_id}/{dataset_id}.")
    sidecars = {}
    for path in results:
        pair_name = path.name.removesuffix(_RESULT_SUFFIX)
        sidecar = path.with_name(f"{pair_name}{_SIDECAR_SUFFIX}")
        payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        if "predictions" not in payload:
            raise MissingSidecarError(f"{condition_id}/{dataset_id}/{pair_name} has no per-source sidecar.")
        sidecars[pair_name] = payload
    return sidecars


def _ranking_values(sidecar: dict, metric: str) -> Values:
    """Return each test reference source's MRR or Hits@1 term and a unit count."""
    values = {}
    for record in sidecar["test_sources"]:
        best = record["best_gold_rank"]
        if metric == "mrr":
            values[record["source"]] = (1.0 / best if best else 0.0, 1.0)
        else:
            values[record["source"]] = (float(best == 1), 1.0)
    return values


def _f1_values(sidecar: dict) -> Values:
    """
    Decompose a result's TP, FP and FN by test source.

    Parameters
    ----------
    sidecar : dict
        Per-source sidecar of one result.

    Returns
    -------
    Values
        ``(tp, fp, fn)`` per test reference source and per unmatched source
        with a judgeable top-1 prediction.

    Raises
    ------
    DataError
        If the gold targets do not add up to the result's test references.
    """
    predictions = {record["source"]: record for record in sidecar["predictions"]}
    values = {}
    for source, record in predictions.items():
        values[source] = (float(record["correct"]), float(record["predicted"] and not record["correct"]), 0.0)
    for record in sidecar["test_sources"]:
        tp, fp, _ = values.get(record["source"], (0.0, 0.0, 0.0))
        values[record["source"]] = (tp, fp, len(record["gold_targets"]) - tp)
    if sum(len(r["gold_targets"]) for r in sidecar["test_sources"]) != sidecar["n_test_refs"]:
        raise DataError("Sidecar gold targets do not add up to its test references.")
    return values


def unit_values(sidecar: dict, metric: str) -> Values:
    """
    Return every unit's contribution to ``metric`` in one result.

    Parameters
    ----------
    sidecar : dict
        Per-source sidecar of one result.
    metric : str
        One of :data:`METRICS`.

    Returns
    -------
    Values
        Component tuple per unit: ``(term, 1)`` for MRR and Hits@1,
        ``(tp, fp, fn)`` for F1.
    """
    return _f1_values(sidecar) if metric == "f1" else _ranking_values(sidecar, metric)


def _gold_map(sidecar: dict) -> Dict[str, List[str]]:
    """Return each test reference source's gold targets."""
    return {record["source"]: sorted(record["gold_targets"]) for record in sidecar["test_sources"]}


def check_same_references(side_a: Dict[str, dict], side_b: Dict[str, dict]) -> None:
    """
    Require two conditions to share pairs, test reference sources and gold targets.

    Parameters
    ----------
    side_a, side_b : Dict[str, dict]
        Sidecars of the two conditions by pair name.

    Raises
    ------
    DataError
        If the pair names or any pair's test references differ.
    """
    if set(side_a) != set(side_b):
        raise DataError(f"Pairs differ: {sorted(set(side_a) ^ set(side_b))}.")
    for pair_name in sorted(side_a):
        if _gold_map(side_a[pair_name]) != _gold_map(side_b[pair_name]):
            raise DataError(f"Test references differ on {pair_name}.")


def aligned_arrays(values_a: Values, values_b: Values) -> PairArrays:
    """
    Align two conditions' unit values on the union of their units.

    A unit missing on one side (an unmatched source only one condition made a
    judgeable prediction for) contributes zeros there.

    Parameters
    ----------
    values_a, values_b : Values
        Unit values of the two conditions on one pair.

    Returns
    -------
    PairArrays
        ``(a, b)``, each ``(n_units, n_components)``, rows in source order.
    """
    units = sorted(set(values_a) | set(values_b))
    width = len(next(iter({**values_a, **values_b}.values())))
    zero = (0.0,) * width
    a = np.array([values_a.get(unit, zero) for unit in units], dtype=float)
    b = np.array([values_b.get(unit, zero) for unit in units], dtype=float)
    return a, b


def statistic(sums: np.ndarray, metric: str) -> np.ndarray:
    """
    Turn summed unit components into the metric, along the last axis.

    Parameters
    ----------
    sums : np.ndarray
        Component sums, shape ``(..., n_components)``.
    metric : str
        One of :data:`METRICS`.

    Returns
    -------
    np.ndarray
        The metric, 0 where it is undefined (no unit, or no TP, FP or FN).
    """
    if metric == "f1":
        numerator, denominator = 2 * sums[..., 0], 2 * sums[..., 0] + sums[..., 1] + sums[..., 2]
    else:
        numerator, denominator = sums[..., 0], sums[..., 1]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _replicate_chunks(n_replicates: int) -> Iterator[int]:
    """Yield chunk sizes that add up to ``n_replicates``."""
    for start in range(0, n_replicates, _CHUNK):
        yield min(_CHUNK, n_replicates - start)


def _bootstrap_deltas(pairs: List[PairArrays], metric: str, settings: TestSettings) -> np.ndarray:
    """Resample units within each pair and return the column delta of every replicate."""
    rng = np.random.default_rng(settings.seed)
    deltas = []
    for size in _replicate_chunks(settings.n_replicates):
        per_pair = []
        for a, b in pairs:
            index = rng.integers(0, len(a), size=(size, len(a)))
            per_pair.append(statistic(b[index].sum(axis=1), metric) - statistic(a[index].sum(axis=1), metric))
        deltas.append(np.mean(per_pair, axis=0))
    return np.concatenate(deltas)


def _randomisation_deltas(pairs: List[PairArrays], metric: str, settings: TestSettings) -> np.ndarray:
    """Swap each unit's two outcomes with probability 1/2 and return every replicate's delta."""
    rng = np.random.default_rng(settings.seed)
    deltas = []
    for size in _replicate_chunks(settings.n_replicates):
        per_pair = []
        for a, b in pairs:
            swap = (rng.random((size, len(a))) < 0.5).astype(float)
            sums_a = b.sum(axis=0) + swap @ (a - b)
            sums_b = a.sum(axis=0) + b.sum(axis=0) - sums_a
            per_pair.append(statistic(sums_b, metric) - statistic(sums_a, metric))
        deltas.append(np.mean(per_pair, axis=0))
    return np.concatenate(deltas)


def _observed(pairs: List[PairArrays], metric: str) -> Tuple[float, float]:
    """Return the column values of A and B: the mean over pairs of each pair's metric."""
    value_a = float(np.mean([statistic(a.sum(axis=0), metric) for a, _ in pairs]))
    value_b = float(np.mean([statistic(b.sum(axis=0), metric) for _, b in pairs]))
    return value_a, value_b


def paired_test(pairs: List[PairArrays], metric: str, settings: TestSettings) -> dict:
    """
    Estimate a column's B - A difference with its interval and p-value.

    Parameters
    ----------
    pairs : List[PairArrays]
        Aligned unit arrays of every pair of the column.
    metric : str
        One of :data:`METRICS`.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    dict
        ``value_a``, ``value_b``, ``delta``, ``ci_low``, ``ci_high``,
        ``p_value``, ``n_units`` and ``n_pairs``.
    """
    value_a, value_b = _observed(pairs, metric)
    delta = value_b - value_a
    tail = 100 * (1 - CONFIDENCE) / 2
    low, high = np.percentile(_bootstrap_deltas(pairs, metric, settings), [tail, 100 - tail])
    null = _randomisation_deltas(pairs, metric, settings)
    extreme = int(np.sum(np.abs(null) >= abs(delta) - _TOLERANCE))
    return {
        "value_a": value_a, "value_b": value_b, "delta": delta,
        "ci_low": float(low), "ci_high": float(high),
        "p_value": (1 + extreme) / (settings.n_replicates + 1),
        "n_units": sum(len(a) for a, _ in pairs), "n_pairs": len(pairs),
    }


def holm_adjust(p_values: List[float]) -> List[float]:
    """
    Return Holm step-down adjusted p-values, in the input order.

    Parameters
    ----------
    p_values : List[float]
        Unadjusted p-values of one family.

    Returns
    -------
    List[float]
        Adjusted p-values, monotone in the unadjusted order and capped at 1.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def column_ceiling(counts: Optional[dict], dataset_id: str, metric: str) -> Optional[float]:
    """
    Return a column's top-20 ceiling of ``metric``: the mean over its pairs.

    Parameters
    ----------
    counts : Optional[dict]
        Output of :func:`kgsemembed.evaluation.reference_counts.load_reference_counts`,
        or ``None`` when unavailable.
    dataset_id : str
        Result column.
    metric : str
        One of :data:`METRICS`.

    Returns
    -------
    Optional[float]
        The ceiling, or ``None`` when no count covers the column.
    """
    rows = [row for (column, _), row in (counts or {}).items() if column == dataset_id]
    return float(np.mean([row[CEILING_KEYS[metric]] for row in rows])) if rows else None


def _column_pairs(side_a: Dict[str, dict], side_b: Dict[str, dict], metric: str) -> List[PairArrays]:
    """Align both conditions' unit values on every pair of a column."""
    return [
        aligned_arrays(unit_values(side_a[name], metric), unit_values(side_b[name], metric))
        for name in sorted(side_a)
    ]


def _comparison_rows(context: dict, settings: TestSettings) -> List[dict]:
    """
    Test one comparison on one column for every metric.

    Parameters
    ----------
    context : dict
        ``group``, ``condition_a``, ``condition_b``, ``dataset_id``,
        ``results_dir`` and ``counts``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per metric; ``tested`` is ``False`` with a ``reason`` when a
        sidecar is missing.
    """
    keys = {k: context[k] for k in ("group", "condition_a", "condition_b", "dataset_id")}
    try:
        side_a = load_sidecars(context["results_dir"], keys["condition_a"], keys["dataset_id"])
        side_b = load_sidecars(context["results_dir"], keys["condition_b"], keys["dataset_id"])
    except MissingSidecarError as exc:
        return [{**keys, "metric": m, "tested": False, "reason": str(exc)} for m in METRICS]
    check_same_references(side_a, side_b)
    return [
        {**keys, "metric": metric, "tested": True, "primary": metric in PRIMARY_METRICS,
         "ceiling": column_ceiling(context["counts"], keys["dataset_id"], metric),
         **paired_test(_column_pairs(side_a, side_b, metric), metric, settings)}
        for metric in METRICS
    ]


def apply_holm(rows: List[dict], alpha: float = ALPHA) -> List[dict]:
    """
    Add Holm-adjusted p-values and decisions within each family.

    A family is one ablation group, one dataset column and one metric; only
    tested rows take part.

    Parameters
    ----------
    rows : List[dict]
        Rows from :func:`run_paired_tests` before adjustment.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    List[dict]
        The same rows, tested ones extended with ``family_size``,
        ``p_holm`` and ``significant``.
    """
    families: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in rows:
        if row["tested"]:
            families.setdefault((row["group"], row["dataset_id"], row["metric"]), []).append(row)
    for family in families.values():
        for row, adjusted in zip(family, holm_adjust([row["p_value"] for row in family])):
            row.update(family_size=len(family), p_holm=adjusted, significant=adjusted < alpha)
    return rows


def run_paired_tests(
    results_dir: str | Path, counts: Optional[dict] = None, settings: TestSettings = TestSettings()
) -> List[dict]:
    """
    Test every registered comparison on every dataset column it covers.

    Parameters
    ----------
    results_dir : str | Path
        Root of the result tree, with per-source sidecars.
    counts : Optional[dict]
        Per-pair reference counts and ceilings, or ``None`` to omit ceilings.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per comparison, column and metric, in registry order, with
        Holm decisions (:func:`apply_holm`).

    Raises
    ------
    DataError
        If two compared conditions have different test references.
    """
    rows = []
    for group, comparisons in registered_comparisons().items():
        for condition_a, condition_b, datasets in comparisons:
            for dataset_id in expand_dataset_ids(list(datasets)):
                context = {"group": group, "condition_a": condition_a, "condition_b": condition_b,
                           "dataset_id": dataset_id, "results_dir": results_dir, "counts": counts}
                rows += _comparison_rows(context, settings)
    return apply_holm(rows)
