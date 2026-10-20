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
  sources outside the reference that either condition predicted (top-1 score
  at or above its threshold).  Each unit carries its TP, FP and FN, and F1 is
  recomputed from their sums.
- **Sidecar check**: a sidecar is used only when its ``result`` link, its
  threshold, its TP/FP and its test reference count match the result file
  beside it (:func:`check_sidecar`); any failure of a comparison is reported
  as not tested, never fatal to the run.
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

from kgsemembed.evaluation.per_source import RESULT_LINK_KEYS, SIDECAR_SUFFIX
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
_RANKING_WIDTH = 2
_F1_WIDTH = 3
_UNTESTABLE = (DataError, ValueError, KeyError, OSError)
_CHUNK = 100
_TOLERANCE = 1e-12

Values = Dict[str, Tuple[float, ...]]
PairArrays = Tuple[np.ndarray, np.ndarray]


class MissingSidecarError(DataError):
    """A result has no per-source sidecar, so it cannot be tested source by source."""


class StaleSidecarError(DataError):
    """A sidecar does not belong to the result file beside it."""


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


def check_sidecar(sidecar: dict, payload: dict, label: str) -> None:
    """
    Refuse a sidecar that was not written with the result beside it.

    Parameters
    ----------
    sidecar : dict
        Per-source sidecar.
    payload : dict
        The result file beside it.
    label : str
        ``condition/column/pair``, for the error message.

    Raises
    ------
    StaleSidecarError
        If its result link, threshold, TP/FP or test reference count differ
        from the result's.
    """
    predictions = sidecar["predictions"]
    found = {
        "result": sidecar.get("result"),
        "threshold": sidecar["threshold"],
        "tp_fp": [sum(r["correct"] for r in predictions),
                  sum(r["predicted"] and not r["correct"] for r in predictions)],
        "n_test_refs": sidecar["n_test_refs"],
    }
    expected = {
        "result": {key: payload.get(key) for key in RESULT_LINK_KEYS},
        "threshold": payload["metrics"]["threshold"],
        "tp_fp": [payload["counts"]["tp"], payload["counts"]["fp"]],
        "n_test_refs": payload["population"]["n_test_refs"],
    }
    differing = sorted(key for key in expected if found[key] != expected[key])
    if differing:
        raise StaleSidecarError(f"{label}: sidecar differs from its result in {differing}.")


def _read_json(path: Path) -> dict:
    """Parse a JSON file, or return an empty mapping when it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_sidecars(results_dir: str | Path, condition_id: str, dataset_id: str) -> Dict[str, dict]:
    """
    Load and check the per-source sidecar of every result of a condition on a column.

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
    StaleSidecarError
        If a sidecar was not written with its result (:func:`check_sidecar`).
    """
    results = sorted((Path(results_dir) / condition_id / dataset_id).glob(f"*{_RESULT_SUFFIX}"))
    if not results:
        raise MissingSidecarError(f"No results for {condition_id}/{dataset_id}.")
    sidecars = {}
    for path in results:
        pair_name = path.name.removesuffix(_RESULT_SUFFIX)
        label = f"{condition_id}/{dataset_id}/{pair_name}"
        sidecar = _read_json(path.with_name(f"{pair_name}{SIDECAR_SUFFIX}"))
        if "predictions" not in sidecar:
            raise MissingSidecarError(f"{label} has no per-source sidecar.")
        check_sidecar(sidecar, _read_json(path), label)
        sidecars[pair_name] = sidecar
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
        ``(tp, fp, fn)`` per test reference source and per other source whose
        top-1 candidate was predicted.

    Raises
    ------
    DataError
        If the gold targets do not add up to the result's test references.
    """
    predictions = {record["source"]: record for record in sidecar["predictions"] if record["predicted"]}
    values = {}
    for source, record in predictions.items():
        values[source] = (float(record["correct"]), float(not record["correct"]), 0.0)
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


def aligned_arrays(values_a: Values, values_b: Values, metric: str) -> PairArrays:
    """
    Align two conditions' unit values on the union of their units.

    A unit missing on one side (a source only one condition predicted)
    contributes zeros there.  A pair without units gives empty arrays.

    Parameters
    ----------
    values_a, values_b : Values
        Unit values of the two conditions on one pair.
    metric : str
        One of :data:`METRICS`, which fixes the number of components.

    Returns
    -------
    PairArrays
        ``(a, b)``, each ``(n_units, n_components)``, rows in source order.
    """
    units = sorted(set(values_a) | set(values_b))
    width = _F1_WIDTH if metric == "f1" else _RANKING_WIDTH
    zero = (0.0,) * width
    a = np.array([values_a.get(unit, zero) for unit in units], dtype=float).reshape(-1, width)
    b = np.array([values_b.get(unit, zero) for unit in units], dtype=float).reshape(-1, width)
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
            if not len(a):
                per_pair.append(np.zeros(size))
                continue
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


class SidecarCache:
    """
    Load each condition's sidecars and unit values once per run.

    A condition takes part in several comparisons (C10 in five), so its
    sidecars are read, checked and decomposed once, and a loading failure is
    remembered and raised again for every comparison that needs them.
    """

    def __init__(self, results_dir: str | Path) -> None:
        self._results_dir = results_dir
        self._sidecars: Dict[Tuple[str, str], object] = {}
        self._values: Dict[Tuple[str, str, str], Dict[str, Values]] = {}

    def sidecars(self, condition_id: str, dataset_id: str) -> Dict[str, dict]:
        """Return the checked sidecars of a condition on a column, or raise the load error."""
        key = (condition_id, dataset_id)
        if key not in self._sidecars:
            try:
                self._sidecars[key] = load_sidecars(self._results_dir, condition_id, dataset_id)
            except _UNTESTABLE as exc:
                self._sidecars[key] = exc
        if isinstance(self._sidecars[key], Exception):
            raise self._sidecars[key]
        return self._sidecars[key]

    def values(self, condition_id: str, dataset_id: str, metric: str) -> Dict[str, Values]:
        """Return the unit values of every pair of a condition on a column."""
        key = (condition_id, dataset_id, metric)
        if key not in self._values:
            side = self.sidecars(condition_id, dataset_id)
            self._values[key] = {name: unit_values(side[name], metric) for name in side}
        return self._values[key]


def _column_pairs(cache: SidecarCache, keys: dict, metric: str) -> List[PairArrays]:
    """Align both conditions' unit values on every pair of a column."""
    values_a = cache.values(keys["condition_a"], keys["dataset_id"], metric)
    values_b = cache.values(keys["condition_b"], keys["dataset_id"], metric)
    return [aligned_arrays(values_a[name], values_b[name], metric) for name in sorted(values_a)]


def _tested_rows(context: dict, keys: dict, settings: TestSettings) -> List[dict]:
    """Run every metric's test of one comparison on one column."""
    cache = context["cache"]
    check_same_references(cache.sidecars(keys["condition_a"], keys["dataset_id"]),
                          cache.sidecars(keys["condition_b"], keys["dataset_id"]))
    return [
        {**keys, "metric": metric, "tested": True, "primary": metric in PRIMARY_METRICS,
         "ceiling": column_ceiling(context["counts"], keys["dataset_id"], metric),
         **paired_test(_column_pairs(cache, keys, metric), metric, settings)}
        for metric in METRICS
    ]


def _comparison_rows(context: dict, settings: TestSettings) -> List[dict]:
    """
    Test one comparison on one column for every metric.

    Parameters
    ----------
    context : dict
        ``group``, ``condition_a``, ``condition_b``, ``dataset_id``,
        ``cache`` (a :class:`SidecarCache`) and ``counts``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per metric; ``tested`` is ``False`` with a ``reason`` when the
        sidecars are missing, stale, malformed or describe different
        references, so one bad comparison never stops the others.
    """
    keys = {k: context[k] for k in ("group", "condition_a", "condition_b", "dataset_id")}
    try:
        return _tested_rows(context, keys, settings)
    except _UNTESTABLE as exc:
        return [{**keys, "metric": m, "tested": False, "reason": str(exc)} for m in METRICS]


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

    A comparison that cannot be tested is reported with its reason instead
    of raising.
    """
    rows, cache = [], SidecarCache(results_dir)
    for group, comparisons in registered_comparisons().items():
        for condition_a, condition_b, datasets in comparisons:
            for dataset_id in expand_dataset_ids(list(datasets)):
                context = {"group": group, "condition_a": condition_a, "condition_b": condition_b,
                           "dataset_id": dataset_id, "cache": cache, "counts": counts}
                rows += _comparison_rows(context, settings)
    return apply_holm(rows)
