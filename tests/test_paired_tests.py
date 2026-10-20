"""Tests for the per-dataset paired tests of the registered comparisons (#140)."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from kgsemembed.evaluation import paired_tests as pt
from kgsemembed.evaluation.metrics import compute_all_metrics
from kgsemembed.evaluation.per_source import link_to_result, per_source_record
from kgsemembed.utils.errors import DataError

_FAST = pt.TestSettings(n_replicates=500, seed=7)


def _sidecar(lists, refs, threshold=0.5) -> dict:
    return per_source_record(lists, refs, compute_all_metrics(lists, refs, threshold=threshold))


def _lists(n: int, hit_every: int, prefix: str = "s") -> Tuple[list, list]:
    """``n`` sources; every ``hit_every``-th ranks its gold first, the rest second."""
    lists, refs = [], []
    for i in range(n):
        source, gold = f"{prefix}{i}", f"g{i}"
        first = gold if i % hit_every == 0 else f"x{i}"
        second = f"x{i}" if first == gold else gold
        lists.append([(source, first, 0.9), (source, second, 0.4)])
        refs.append((source, gold))
    return lists, refs


def _payload(sidecar: dict, run: str = "2026-10-20T00:00:00Z") -> dict:
    """A result file consistent with ``sidecar``."""
    predictions = sidecar.get("predictions", [])
    return {
        "run_timestamp": run, "candidates_sha256": "c" * 64, "git_commit": "abc",
        "metrics": {"threshold": sidecar.get("threshold")},
        "counts": {"tp": sum(r["correct"] for r in predictions),
                   "fp": sum(r["predicted"] and not r["correct"] for r in predictions)},
        "population": {"n_test_refs": sidecar.get("n_test_refs")},
    }


def _write(results_dir: Path, condition: str, column: str, pair: str, sidecar: dict,
           payload: dict | None = None) -> None:
    directory = results_dir / condition / column
    directory.mkdir(parents=True, exist_ok=True)
    payload = payload or _payload(sidecar)
    (directory / f"{pair}_results.json").write_text(json.dumps(payload), encoding="utf-8")
    linked = link_to_result(sidecar, payload) if "predictions" in sidecar else sidecar
    (directory / f"{pair}_entities.json").write_text(json.dumps(linked), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit values
# ---------------------------------------------------------------------------
def test_unit_values_reproduce_the_stored_metrics() -> None:
    lists, refs = _lists(12, 3)
    lists.append([("u0", "t0", 0.95)])
    metrics = compute_all_metrics(lists, refs, threshold=0.5)
    sidecar = per_source_record(lists, refs, metrics)
    for metric in pt.METRICS:
        values = np.array(list(pt.unit_values(sidecar, metric).values()))
        assert pt.statistic(values.sum(axis=0), metric) == pytest.approx(metrics[metric])


def test_f1_units_count_a_two_target_source_once_with_one_fn() -> None:
    lists = [[("s", "a", 0.9)]]
    sidecar = _sidecar(lists, [("s", "a"), ("s", "b")])
    assert pt.unit_values(sidecar, "f1") == {"s": (1.0, 0.0, 1.0)}
    assert pt.unit_values(sidecar, "mrr") == {"s": (1.0, 1.0)}


def test_f1_units_include_unmatched_predictions_as_false_positives() -> None:
    sidecar = _sidecar([[("s", "a", 0.9)], [("u", "z", 0.9)]], [("s", "a")])
    assert pt.unit_values(sidecar, "f1")["u"] == (0.0, 1.0, 0.0)


def test_aligned_arrays_fill_a_unit_missing_on_one_side_with_zeros() -> None:
    a, b = pt.aligned_arrays(
        {"s": (1.0, 0.0, 0.0)}, {"s": (1.0, 0.0, 0.0), "u": (0.0, 1.0, 0.0)}, "f1"
    )
    assert a.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert b.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


# ---------------------------------------------------------------------------
# One test
# ---------------------------------------------------------------------------
def _pair_arrays(hit_a: int, hit_b: int, n: int = 60, metric: str = "mrr") -> List[pt.PairArrays]:
    side_a, side_b = _sidecar(*_lists(n, hit_a)), _sidecar(*_lists(n, hit_b))
    return [pt.aligned_arrays(pt.unit_values(side_a, metric), pt.unit_values(side_b, metric), metric)]


def test_identical_conditions_have_no_difference() -> None:
    result = pt.paired_test(_pair_arrays(2, 2), "mrr", _FAST)
    assert result["delta"] == 0 and result["ci_low"] == result["ci_high"] == 0
    assert result["p_value"] == 1.0


def test_a_clear_gain_excludes_zero_and_reaches_the_smallest_p() -> None:
    result = pt.paired_test(_pair_arrays(4, 1, metric="hits_at_1"), "hits_at_1", _FAST)
    assert result["delta"] == pytest.approx(1 - 15 / 60)
    assert result["ci_low"] > 0
    assert result["p_value"] == pytest.approx(1 / 501)


def test_results_are_reproducible_from_the_seed() -> None:
    first = pt.paired_test(_pair_arrays(4, 3), "mrr", _FAST)
    second = pt.paired_test(_pair_arrays(4, 3), "mrr", _FAST)
    assert first == second


def test_column_value_is_the_mean_over_pairs_not_a_pooled_value() -> None:
    small = pt.aligned_arrays(*(pt.unit_values(_sidecar(*_lists(2, 1, "a")), "hits_at_1"),) * 2, "hits_at_1")
    large = pt.aligned_arrays(*(pt.unit_values(_sidecar(*_lists(98, 99, "b")), "hits_at_1"),) * 2, "hits_at_1")
    result = pt.paired_test([small, large], "hits_at_1", _FAST)
    assert result["value_a"] == pytest.approx((1.0 + 1 / 98) / 2)
    assert (result["n_pairs"], result["n_units"]) == (2, 100)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------
def test_holm_adjustment_matches_a_worked_example() -> None:
    assert pt.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one() -> None:
    assert pt.holm_adjust([0.6, 0.9]) == [1.0, 1.0]


def test_families_are_one_group_and_column_of_mrr_tests() -> None:
    rows = [
        {"group": "B", "dataset_id": "D1", "metric": "mrr", "tested": True, "p_value": 0.03},
        {"group": "B", "dataset_id": "D1", "metric": "mrr", "tested": True, "p_value": 0.04},
        {"group": "B", "dataset_id": "D2", "metric": "mrr", "tested": True, "p_value": 0.02},
        {"group": "B", "dataset_id": "D1", "metric": "f1", "tested": False},
    ]
    pt.apply_holm(rows)
    assert [row.get("family_size") for row in rows] == [2, 2, 1, None]
    assert [row.get("significant") for row in rows] == [False, False, True, None]


def test_only_mrr_carries_a_decision() -> None:
    rows = [
        {"group": "E", "dataset_id": "D1", "metric": metric, "tested": True, "p_value": 0.001}
        for metric in pt.METRICS
    ]
    pt.apply_holm(rows)
    assert [("significant" in row) for row in rows] == [True, False, False]
    assert rows[0]["family_size"] == 1


# ---------------------------------------------------------------------------
# Loading and the full run
# ---------------------------------------------------------------------------
def test_missing_sidecar_is_reported() -> None:
    with pytest.raises(pt.MissingSidecarError):
        pt.load_sidecars("does/not/exist", "C1", "D1")


def test_a_result_without_predictions_counts_as_missing(tmp_path: Path) -> None:
    _write(tmp_path, "C1", "D1", "p", {"test_sources": []})
    with pytest.raises(pt.MissingSidecarError, match="no per-source sidecar"):
        pt.load_sidecars(tmp_path, "C1", "D1")


def test_different_references_are_refused() -> None:
    side_a = {"p": _sidecar(*_lists(4, 1))}
    side_b = {"p": _sidecar(*_lists(5, 1))}
    with pytest.raises(DataError, match="differ"):
        pt.check_same_references(side_a, side_b)


def _seed_group_e(results_dir: Path) -> None:
    columns = ("D1", "D2", "D4_schema", "D4_instance", "D5")
    for column in columns:
        _write(results_dir, "C20", column, "p", _sidecar(*_lists(40, 3)))
        _write(results_dir, "C10", column, "p", _sidecar(*_lists(40, 1)))
    for index in range(3):
        _write(results_dir, "C20", "D3", f"p{index}", _sidecar(*_lists(10, 2)))
        _write(results_dir, "C10", "D3", f"p{index}", _sidecar(*_lists(10, 2)))


def test_run_tests_every_column_and_metric_of_a_comparison(tmp_path: Path) -> None:
    _seed_group_e(tmp_path)
    counts = {("D1", "p"): {"ranking_ceiling": 0.9, "f1_recall_ceiling": 0.8}}
    rows = [r for r in pt.run_paired_tests(tmp_path, counts, _FAST) if r["group"] == "E"]
    assert len(rows) == 6 * 3 and all(r["tested"] for r in rows)
    d1_mrr = next(r for r in rows if r["dataset_id"] == "D1" and r["metric"] == "mrr")
    assert d1_mrr["delta"] > 0 and d1_mrr["significant"] and d1_mrr["ceiling"] == 0.9
    d3 = [r for r in rows if r["dataset_id"] == "D3"]
    assert all(r["delta"] == 0 for r in d3)
    assert [r["significant"] for r in d3 if "significant" in r] == [False]
    assert next(r for r in rows if r["dataset_id"] == "D2")["ceiling"] is None


def test_run_marks_comparisons_without_sidecars_untested(tmp_path: Path) -> None:
    rows = pt.run_paired_tests(tmp_path, None, _FAST)
    assert rows and not any(r["tested"] for r in rows)
    assert all("reason" in r for r in rows)


# ---------------------------------------------------------------------------
# Review fixes: sidecar checks, robustness, F1 units, caching
# ---------------------------------------------------------------------------
def test_f1_units_leave_out_sources_neither_condition_predicted() -> None:
    sidecar = _sidecar([[("s", "a", 0.9)], [("u", "z", 0.1)]], [("s", "a")])
    assert set(pt.unit_values(sidecar, "f1")) == {"s"}


def test_a_sidecar_from_another_run_is_refused(tmp_path: Path) -> None:
    sidecar = _sidecar(*_lists(4, 1))
    _write(tmp_path, "C1", "D1", "p", sidecar)
    newer = _payload(sidecar, run="2026-10-21T00:00:00Z")
    (tmp_path / "C1" / "D1" / "p_results.json").write_text(json.dumps(newer), encoding="utf-8")
    with pytest.raises(pt.StaleSidecarError, match="result"):
        pt.load_sidecars(tmp_path, "C1", "D1")


def test_a_sidecar_with_other_counts_is_refused(tmp_path: Path) -> None:
    sidecar = _sidecar(*_lists(4, 1))
    payload = _payload(sidecar)
    payload["counts"]["fp"] += 1
    _write(tmp_path, "C1", "D1", "p", sidecar, payload)
    with pytest.raises(pt.StaleSidecarError, match="tp_fp"):
        pt.load_sidecars(tmp_path, "C1", "D1")


def test_a_sidecar_without_a_result_link_is_refused(tmp_path: Path) -> None:
    sidecar = _sidecar(*_lists(4, 1))
    directory = tmp_path / "C1" / "D1"
    directory.mkdir(parents=True)
    (directory / "p_results.json").write_text(json.dumps(_payload(sidecar)), encoding="utf-8")
    (directory / "p_entities.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(pt.StaleSidecarError):
        pt.load_sidecars(tmp_path, "C1", "D1")


def test_different_references_make_one_comparison_untested_not_the_run(tmp_path: Path) -> None:
    _seed_group_e(tmp_path)
    _write(tmp_path, "C10", "D2", "p", _sidecar(*_lists(41, 1)))
    rows = [r for r in pt.run_paired_tests(tmp_path, None, _FAST) if r["group"] == "E"]
    d2 = [r for r in rows if r["dataset_id"] == "D2"]
    assert all(not r["tested"] and "differ" in r["reason"] for r in d2)
    assert all(r["tested"] for r in rows if r["dataset_id"] != "D2")


def test_a_malformed_sidecar_makes_its_comparisons_untested(tmp_path: Path) -> None:
    _seed_group_e(tmp_path)
    (tmp_path / "C10" / "D5" / "p_entities.json").write_text("{not json", encoding="utf-8")
    rows = [r for r in pt.run_paired_tests(tmp_path, None, _FAST) if r["dataset_id"] == "D5"]
    assert rows and not any(r["tested"] for r in rows)


def test_a_pair_without_units_contributes_no_difference() -> None:
    empty = pt.aligned_arrays({}, {}, "mrr")
    assert empty[0].shape == (0, 2)
    pairs = [empty, *_pair_arrays(4, 1)]
    result = pt.paired_test(pairs, "mrr", _FAST)
    assert result["n_pairs"] == 2 and result["value_a"] == pytest.approx(0.625 / 2)
    assert result["ci_low"] > 0


def test_each_condition_is_loaded_once_per_run(tmp_path: Path, monkeypatch) -> None:
    _seed_group_e(tmp_path)
    loads = []
    original = pt.load_sidecars
    monkeypatch.setattr(pt, "load_sidecars", lambda *a: loads.append(a[1:]) or original(*a))
    pt.run_paired_tests(tmp_path, None, _FAST)
    assert len(loads) == len(set(loads))


def test_results_from_different_setups_are_not_compared(tmp_path: Path) -> None:
    _seed_group_e(tmp_path)
    sidecar = _sidecar(*_lists(40, 1))
    payload = {**_payload(sidecar), "population": {"n_test_refs": 40, "n_train_refs": 726}}
    _write(tmp_path, "C10", "D1", "p", sidecar, payload)
    rows = [r for r in pt.run_paired_tests(tmp_path, None, _FAST)
            if r["dataset_id"] == "D1" and r["group"] == "E"]
    assert len(rows) == 3 and all(not r["tested"] and "setup differs" in r["reason"] for r in rows)


def test_an_unexpected_value_type_makes_a_comparison_untested(tmp_path: Path) -> None:
    _seed_group_e(tmp_path)
    path = tmp_path / "C10" / "D5" / "p_entities.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["test_sources"][0]["best_gold_rank"] = "one"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    rows = [r for r in pt.run_paired_tests(tmp_path, None, _FAST) if r["dataset_id"] == "D5"]
    assert rows and not any(r["tested"] for r in rows)
