"""Per-pair reference counts and top-20 candidate ceilings (#139).

Results stored before source-level Hits@k was recorded carry only the
reference-level ``recall_at_1``.  Their Hits@1 is recovered exactly with
:func:`kgsemembed.evaluation.metrics.hits_at_1_from_recall_at_1`, which needs
each pair's number of test reference sources.  Those counts depend only on the
dataset and its stored candidates, never on a condition, so they are computed
once by ``scripts/metric_ceilings.py`` into a tracked JSON file and read here.

The same file holds the candidate ceilings every table reports: the
source-level ceiling of MRR and Hits@k, the reference-level ceiling of
Recall@k (candidate recall@20), and the recall ceiling of the one-prediction
-per-source F1.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from kgsemembed.evaluation.metrics import hits_at_1_from_recall_at_1

DEFAULT_COUNTS_PATH = Path("artifacts/issue-139/metric_ceilings.json")

PairKey = Tuple[str, str]
PairCounts = Dict[str, float]


def load_reference_counts(path: str | Path = DEFAULT_COUNTS_PATH) -> Dict[PairKey, PairCounts]:
    """
    Load the per-pair reference counts and ceilings.

    Parameters
    ----------
    path : str | Path
        JSON file written by ``scripts/metric_ceilings.py``.

    Returns
    -------
    Dict[PairKey, PairCounts]
        Counts keyed by ``(dataset_id, pair_name)``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(row["dataset_id"], row["pair_name"]): row for row in payload["pairs"]}


def source_hits_at_1(payload: dict, counts: Dict[PairKey, PairCounts]) -> float:
    """
    Return a result's source-level Hits@1, derived when it was not recorded.

    Parameters
    ----------
    payload : dict
        Parsed result JSON.
    counts : Dict[PairKey, PairCounts]
        Output of :func:`load_reference_counts`.

    Returns
    -------
    float
        ``metrics.hits_at_1`` when present, else the exact conversion of
        ``metrics.recall_at_1``.

    Raises
    ------
    KeyError
        If the pair has neither a recorded Hits@1 nor reference counts.
    ValueError
        If the result's test reference count differs from the counts file.
    """
    metrics = payload["metrics"]
    if "hits_at_1" in metrics:
        return float(metrics["hits_at_1"])
    row = counts[(payload["dataset_id"], payload["pair_name"])]
    n_refs = payload["population"]["n_test_refs"]
    if n_refs != row["n_test_refs"]:
        raise ValueError(
            f"{payload['dataset_id']}/{payload['pair_name']}: result has {n_refs} test "
            f"references, the counts file {row['n_test_refs']}."
        )
    return hits_at_1_from_recall_at_1(metrics["recall_at_1"], n_refs, row["n_test_reference_sources"])


def result_metric(payload: dict, metric: str, counts: Dict[PairKey, PairCounts] | None = None) -> float:
    """
    Read one metric from a result, deriving Hits@1 of older results.

    Parameters
    ----------
    payload : dict
        Parsed result JSON.
    metric : str
        Key of the result's ``metrics`` block, or ``hits_at_1``.
    counts : Dict[PairKey, PairCounts] | None
        Per-pair reference counts; needed only for Hits@1 not recorded.

    Returns
    -------
    float
        The metric value.

    Raises
    ------
    ValueError
        If Hits@1 is neither recorded nor derivable for want of counts.
    """
    if metric != "hits_at_1":
        return float(payload["metrics"][metric])
    try:
        return source_hits_at_1(payload, counts or {})
    except KeyError as exc:
        raise ValueError(
            f"{payload['dataset_id']}/{payload['pair_name']} has no recorded hits_at_1 and no "
            "reference counts; run scripts/metric_ceilings.py or pass --ceilings."
        ) from exc
