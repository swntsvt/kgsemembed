"""Class-versus-predicate analysis on D3 (#166).

One definition shared by the report's "Entity-Type Analysis" section and the
paper table printed by ``scripts/generate_entity_type_table.py``, so the two
always show the same values.

- **Scope**: D3 only.  D4_schema, the other mixed-type column, is a single
  pair whose class and property references (12 and 32 on test) come from a
  different benchmark, so pooling it with D3 would make rows of conditions run
  on one, the other or both incomparable.  It is not reported here.
- **Buckets**: a pair records an entity type's metrics only when it has at
  least :data:`MIN_REFS` test references of that type (``run_experiment``), so
  predicate values rest on the pairs that reach it.  Every row states its pair
  and reference counts.
- **Aggregation**: the D3 convention of #139.  MRR, Hits@1 and F1 are
  unweighted means over the pairs reporting the type; ``f1_micro`` pools TP, FP
  and FN over the same pairs, as the F1 table's D3$_\\mu$ column does.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pandas as pd

from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS

DATASET_ID = "D3"
ENTITY_TYPES = ("class", "predicate")
MIN_REFS = 3
MACRO_METRICS = ("mrr", "hits_at_1", "f1")
TYPE_STATS = ("n_pairs", "n_refs", *MACRO_METRICS, "f1_micro")
GAP_METRICS = ("mrr", "f1")
KEY_COLUMNS = ("condition_id", "strategy", "model_key")
COUNT_FIELDS = ("tp", "fp", "fn")
SUMMARY_COLUMNS = (
    *KEY_COLUMNS,
    *(f"{entity_type}_{stat}" for entity_type in ENTITY_TYPES for stat in TYPE_STATS),
    *(f"{metric}_gap" for metric in GAP_METRICS),
)
_REQUIRED_COLUMNS = frozenset({"dataset_id", "pair_name", "entity_type", "n_refs", *COUNT_FIELDS})
_REGISTRY_ORDER = {c.condition_id: index for index, c in enumerate(EXPERIMENT_CONDITIONS)}


def entity_type_rows(df_expanded: pd.DataFrame) -> pd.DataFrame:
    """
    Select the D3 class and predicate rows that recorded enough references.

    Parameters
    ----------
    df_expanded : pd.DataFrame
        Results loaded with ``include_entity_type_breakdown=True``.  A frame
        without the breakdown columns yields no rows.

    Returns
    -------
    pd.DataFrame
        One row per condition, D3 pair and entity type, with numeric ``n_refs``.
    """
    if not _REQUIRED_COLUMNS.issubset(df_expanded.columns):
        return df_expanded.iloc[0:0]
    rows = df_expanded.assign(n_refs=pd.to_numeric(df_expanded["n_refs"], errors="coerce"))
    return rows[
        (rows["dataset_id"] == DATASET_ID)
        & rows["entity_type"].isin(ENTITY_TYPES)
        & (rows["n_refs"] >= MIN_REFS)
    ]


def micro_f1(tp: int, fp: int, fn: int) -> float:
    """
    Compute F1 from pooled counts.

    Parameters
    ----------
    tp, fp, fn : int
        True positives, false positives and false negatives summed over pairs.

    Returns
    -------
    float
        ``2 tp / (2 tp + fp + fn)``, or 0 when every count is 0.
    """
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _strict_mean(values: pd.Series) -> float:
    """Mean that is missing when any value is, so a mean never covers fewer pairs."""
    return float(pd.to_numeric(values, errors="coerce").astype(float).mean(skipna=False))


def type_summary(rows: pd.DataFrame) -> Dict[str, float]:
    """
    Summarise one condition's rows of one entity type.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of one condition and one entity type, one per pair.

    Returns
    -------
    Dict[str, float]
        The :data:`TYPE_STATS`: pair and reference counts, pair means of
        :data:`MACRO_METRICS`, and micro F1.
    """
    tp, fp, fn = (int(rows[field].astype(int).sum()) for field in COUNT_FIELDS)
    return {
        "n_pairs": rows["pair_name"].nunique(),
        "n_refs": int(rows["n_refs"].sum()),
        **{metric: _strict_mean(rows[metric]) for metric in MACRO_METRICS},
        "f1_micro": micro_f1(tp, fp, fn),
    }


def _summary_records(rows: pd.DataFrame) -> List[dict]:
    """Return one record per condition holding each entity type's statistics."""
    records: Dict[str, dict] = {}
    if rows.empty:
        return []
    for (condition_id, strategy, model_key, entity_type), group in rows.groupby(
        [*KEY_COLUMNS, "entity_type"], sort=False
    ):
        record = records.setdefault(
            condition_id, {"condition_id": condition_id, "strategy": strategy, "model_key": model_key}
        )
        record.update({f"{entity_type}_{stat}": value for stat, value in type_summary(group).items()})
    return list(records.values())


def entity_type_summary(df_expanded: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-condition class-versus-predicate summary on D3.

    Parameters
    ----------
    df_expanded : pd.DataFrame
        Results loaded with ``include_entity_type_breakdown=True``.

    Returns
    -------
    pd.DataFrame
        One row per condition with D3 entity-type data, in registry order, over
        :data:`SUMMARY_COLUMNS`.  An entity type a condition does not report is
        missing; each gap is class minus predicate.
    """
    records = _summary_records(entity_type_rows(df_expanded))
    summary = pd.DataFrame.from_records(records, columns=list(SUMMARY_COLUMNS))
    for metric in GAP_METRICS:
        summary[f"{metric}_gap"] = summary[f"class_{metric}"] - summary[f"predicate_{metric}"]
    order = summary["condition_id"].map(lambda cid: _REGISTRY_ORDER.get(cid, len(_REGISTRY_ORDER)))
    return summary.assign(_order=order).sort_values(["_order", "condition_id"]).drop(columns="_order")


def complete_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the conditions that report both entity types."""
    return summary.dropna(subset=[f"{entity_type}_mrr" for entity_type in ENTITY_TYPES])


def count_text(values: pd.Series) -> str:
    """
    Render a count that should be one value as that value, or its range.

    Parameters
    ----------
    values : pd.Series
        One count per condition, e.g. ``predicate_n_pairs``.

    Returns
    -------
    str
        ``"8"`` when every condition agrees, ``"7–8"`` otherwise.
    """
    present = [int(value) for value in values if not (isinstance(value, float) and math.isnan(value))]
    low, high = min(present), max(present)
    return str(low) if low == high else f"{low}–{high}"


def mean_contrast(complete: pd.DataFrame, metric: str) -> Tuple[float, float, float]:
    """
    Average one metric of each entity type over complete conditions.

    Parameters
    ----------
    complete : pd.DataFrame
        Rows of :func:`complete_rows`.
    metric : str
        One of :data:`MACRO_METRICS` or ``"f1_micro"``.

    Returns
    -------
    Tuple[float, float, float]
        Mean class value, mean predicate value, and their difference, so the
        quoted gap always equals the difference of the quoted means.
    """
    mean_class = float(complete[f"class_{metric}"].mean())
    mean_predicate = float(complete[f"predicate_{metric}"].mean())
    return mean_class, mean_predicate, mean_class - mean_predicate
