"""Tests for the per-source sidecar records (#140)."""

import pytest

from kgsemembed.evaluation.metrics import compute_all_metrics
from kgsemembed.evaluation.per_source import per_source_record, prediction_records
from kgsemembed.utils.errors import PipelineStageError

_LISTS = [
    [("s1", "t1", 0.9), ("s1", "t2", 0.5)],
    [("s2", "t9", 0.8), ("s2", "t2", 0.7)],
    [("s3", "t3", 0.4)],
    [("u1", "t4", 0.95)],
]
_REFS = [("s1", "t1"), ("s2", "t2"), ("s2", "t5"), ("s3", "t3"), ("s4", "t6")]


def _record(threshold: float = 0.6) -> dict:
    metrics = compute_all_metrics(_LISTS, _REFS, threshold=threshold)
    return per_source_record(_LISTS, _REFS, metrics)


def test_predictions_cover_every_ranked_source_once() -> None:
    records = prediction_records(_LISTS, _REFS, 0.6)
    assert [r["source"] for r in records] == ["s1", "s2", "s3", "u1"]


def test_predictions_apply_the_threshold_and_the_references() -> None:
    by_source = {r["source"]: r for r in prediction_records(_LISTS, _REFS, 0.6)}
    assert (by_source["s1"]["predicted"], by_source["s1"]["correct"]) == (True, True)
    assert (by_source["s2"]["predicted"], by_source["s2"]["correct"]) == (True, False)
    assert (by_source["s3"]["predicted"], by_source["s3"]["correct"]) == (False, False)
    assert (by_source["u1"]["predicted"], by_source["u1"]["correct"]) == (True, False)


def test_record_holds_threshold_reference_count_and_both_blocks() -> None:
    record = _record()
    assert set(record) == {"threshold", "n_test_refs", "test_sources", "predictions"}
    assert record["threshold"] == pytest.approx(0.6)
    assert record["n_test_refs"] == 5


def test_test_sources_include_unranked_reference_sources() -> None:
    sources = {r["source"]: r for r in _record()["test_sources"]}
    assert sources["s4"]["ranked"] is False and sources["s4"]["best_gold_rank"] is None
    assert sources["s2"]["best_gold_rank"] == 2


def test_record_refuses_counts_that_differ_from_the_result() -> None:
    metrics = dict(compute_all_metrics(_LISTS, _REFS, threshold=0.6), fp=99)
    with pytest.raises(PipelineStageError, match="FP"):
        per_source_record(_LISTS, _REFS, metrics)
