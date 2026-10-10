"""Tests for the statistical significance testing module.

All tests operate on synthetic result files written into a temporary directory;
no real experiment outputs, embedding models, or repository datasets are used.
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import pytest
from scipy.stats import PermutationMethod

import kgsemembed.evaluation as evaluation
from kgsemembed.evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    collect_f1_scores,
    export_stats_table,
    load_results_for_condition,
    run_group_comparisons,
    wilcoxon_comparison,
)
from kgsemembed.evaluation.stats import (
    _GROUP_COMPARISONS,
    _MIN_PAIRED_OBSERVATIONS,
    expand_dataset_ids,
    minimum_attainable_p,
)
from kgsemembed.utils.errors import DataError


# ---------------------------------------------------------------------------
# Synthetic result fixtures
# ---------------------------------------------------------------------------
def _write_result(
    results_dir: Path,
    condition_id: str,
    dataset_id: str,
    pair_name: str,
    f1: Optional[float],
    protocol: Optional[int] = EVALUATION_PROTOCOL_VERSION,
) -> None:
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, float] = {"precision": 0.5, "recall": 0.5}
    if f1 is not None:
        metrics["f1"] = f1
    payload = {
        "condition_id": condition_id,
        "dataset_id": dataset_id,
        "pair_name": pair_name,
        "metrics": metrics,
    }
    if protocol is not None:
        payload["evaluation_protocol"] = protocol
    (directory / f"{pair_name}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_condition(
    results_dir: Path,
    condition_id: str,
    dataset_id: str,
    f1_by_pair: Dict[str, float],
) -> None:
    for pair_name, f1 in f1_by_pair.items():
        _write_result(results_dir, condition_id, dataset_id, pair_name, f1)


def _pairs(dataset_id: str, count: int, base: float) -> Dict[str, float]:
    return {f"{dataset_id}_p{index}": base + index * 0.01 for index in range(count)}


_UNITS = ("D1", "D2", "D3", "D4_schema", "D4_instance", "D5")
_ALL_DATASETS = ["D1", "D2", "D3", "D4", "D5"]


def _write_units(results_dir: Path, condition_id: str, values: List[float]) -> None:
    """Write one single-pair result per dataset column, in ``_UNITS`` order."""
    for unit, f1 in zip(_UNITS, values):
        _write_result(results_dir, condition_id, unit, f"{unit}_pair", f1)


def _compare_units(tmp_path: Path, values_a: List[float], values_b: List[float]) -> dict:
    """Compare C1 against C2 over the six dataset columns."""
    _write_units(tmp_path, "C1", values_a)
    _write_units(tmp_path, "C2", values_b)
    return wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))


_BASE = [0.30, 0.40, 0.50, 0.20, 0.60, 0.10]


def _shifted(differences: List[float]) -> List[float]:
    return [base + difference for base, difference in zip(_BASE, differences)]


# ---------------------------------------------------------------------------
# Package exports (regression)
# ---------------------------------------------------------------------------
def test_package_exports() -> None:
    expected = {
        "load_results_for_condition",
        "collect_f1_scores",
        "wilcoxon_comparison",
        "run_group_comparisons",
        "export_stats_table",
    }
    assert expected.issubset(set(evaluation.__all__))
    for name in expected:
        assert hasattr(evaluation, name)


def test_existing_exports_unchanged() -> None:
    for name in ("compute_all_metrics", "tune_threshold", "compute_mrr"):
        assert hasattr(evaluation, name)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def test_load_missing_directory_returns_none(tmp_path: Path) -> None:
    assert load_results_for_condition("C1", "D1", str(tmp_path)) is None


def test_load_empty_directory_returns_none(tmp_path: Path) -> None:
    (tmp_path / "C1" / "D1").mkdir(parents=True)
    assert load_results_for_condition("C1", "D1", str(tmp_path)) is None


def test_load_multiple_pairs(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D3", _pairs("D3", 3, 0.4))
    result = load_results_for_condition("C1", "D3", str(tmp_path))
    assert result is not None
    assert set(result) == {"D3_p0", "D3_p1", "D3_p2"}
    assert result["D3_p1"]["metrics"]["f1"] == pytest.approx(0.41)


def test_load_deterministic_ordering(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D3", "D3_p2", 0.3)
    _write_result(tmp_path, "C1", "D3", "D3_p0", 0.1)
    _write_result(tmp_path, "C1", "D3", "D3_p1", 0.2)
    result = load_results_for_condition("C1", "D3", str(tmp_path))
    assert list(result) == ["D3_p0", "D3_p1", "D3_p2"]


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    directory = tmp_path / "C1" / "D1"
    directory.mkdir(parents=True)
    (directory / "D1_p0_results.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError):
        load_results_for_condition("C1", "D1", str(tmp_path))


def test_load_duplicate_pair_name_raises(tmp_path: Path) -> None:
    directory = tmp_path / "C1" / "D1"
    directory.mkdir(parents=True)
    for filename in ("a_results.json", "b_results.json"):
        payload = {"pair_name": "shared", "metrics": {"f1": 0.5}}
        (directory / filename).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataError):
        load_results_for_condition("C1", "D1", str(tmp_path))


def test_load_missing_pair_name_raises(tmp_path: Path) -> None:
    directory = tmp_path / "C1" / "D1"
    directory.mkdir(parents=True)
    (directory / "D1_p0_results.json").write_text(
        json.dumps({"metrics": {"f1": 0.5}}), encoding="utf-8"
    )
    with pytest.raises(DataError):
        load_results_for_condition("C1", "D1", str(tmp_path))


# ---------------------------------------------------------------------------
# F1 collection
# ---------------------------------------------------------------------------
def test_collect_one_value_per_pair(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D3", _pairs("D3", 21, 0.4))
    scores = collect_f1_scores("C1", ["D3"], str(tmp_path))
    assert len(scores) == 21


def test_collect_spans_multiple_datasets(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 2, 0.5))
    _write_condition(tmp_path, "C1", "D3", _pairs("D3", 3, 0.6))
    scores = collect_f1_scores("C1", ["D1", "D3"], str(tmp_path))
    assert len(scores) == 5


def test_collect_expands_d4_into_schema_and_instance(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D4_schema", "schema", 0.3)
    _write_result(tmp_path, "C1", "D4_instance", "instance", 0.7)
    assert collect_f1_scores("C1", ["D4"], str(tmp_path)) == [0.3, 0.7]


def test_collect_deterministic_ordering(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p1", 0.2)
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.1)
    scores = collect_f1_scores("C1", ["D1"], str(tmp_path))
    assert scores == [0.1, 0.2]


def test_collect_missing_dataset_contributes_nothing(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 2, 0.5))
    scores = collect_f1_scores("C1", ["D1", "D2"], str(tmp_path))
    assert len(scores) == 2


def test_collect_missing_f1_raises(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", None)
    with pytest.raises(DataError):
        collect_f1_scores("C1", ["D1"], str(tmp_path))


def test_collect_non_numeric_f1_raises(tmp_path: Path) -> None:
    directory = tmp_path / "C1" / "D1"
    directory.mkdir(parents=True)
    payload = {
        "pair_name": "D1_p0",
        "metrics": {"f1": "not-a-number"},
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
    }
    (directory / "D1_p0_results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataError):
        collect_f1_scores("C1", ["D1"], str(tmp_path))


def test_collect_duplicate_pair_across_datasets_raises(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "shared", 0.5)
    _write_result(tmp_path, "C1", "D2", "shared", 0.6)
    with pytest.raises(DataError):
        collect_f1_scores("C1", ["D1", "D2"], str(tmp_path))


@pytest.mark.parametrize("protocol", [None, 1])
def test_collect_rejects_results_from_an_older_protocol(
    tmp_path: Path, protocol: Optional[int]
) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5, protocol=protocol)
    with pytest.raises(DataError, match="evaluation protocol"):
        collect_f1_scores("C1", ["D1"], str(tmp_path))


def test_expand_dataset_ids_replaces_d4_in_place() -> None:
    assert expand_dataset_ids(["D1", "D4", "D5"]) == [
        "D1", "D4_schema", "D4_instance", "D5",
    ]


# ---------------------------------------------------------------------------
# Experimental unit
# ---------------------------------------------------------------------------
def test_d3_pairs_collapse_into_one_unit(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _shifted([0.1] * 6))
    for name, value in _pairs("D3", 20, 0.3).items():
        _write_result(tmp_path, "C1", "D3", name, value)
        _write_result(tmp_path, "C2", "D3", name, value + 0.1)
    result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert result["n_units"] == 6
    assert result["n_pairs"] == 26
    assert result["units"] == list(_UNITS)


def test_d4_columns_take_part_in_comparisons(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.1] * 6))
    assert {"D4_schema", "D4_instance"} <= set(result["units"])


def test_unit_value_is_the_mean_of_its_pairs(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _BASE)
    _write_result(tmp_path, "C2", "D3", "D3_pair", 0.2)
    _write_result(tmp_path, "C1", "D3", "D3_extra", 0.5)
    _write_result(tmp_path, "C2", "D3", "D3_extra", 0.4)
    result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert result["delta_f1"] == pytest.approx(((0.2 + 0.4) / 2 - (0.5 + 0.5) / 2) / 6)


def test_pair_outcomes_count_every_alignment_pair(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.1, 0.0, -0.1, 0.2, 0.0, 0.3]))
    assert (result["pair_wins"], result["pair_ties"], result["pair_losses"]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# Wilcoxon comparison
# ---------------------------------------------------------------------------
def test_wilcoxon_identical_lists_pvalue_is_one(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _BASE)
    assert result["p_value"] == 1.0
    assert result["significant"] is False
    assert result["delta_f1"] == pytest.approx(0.0)
    assert result["effect_size_r"] == 0.0
    assert "zero" in result["wilcoxon_warning"]


def test_wilcoxon_floating_point_noise_counts_as_a_zero_difference(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, [0.1 + 0.2] * 6, [0.3] * 6)
    assert result["n_nonzero"] == 0
    assert result["p_value"] == 1.0


def test_wilcoxon_all_units_improved_reaches_exact_minimum(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]))
    assert result["p_value"] == pytest.approx(2 / 64)
    assert result["min_attainable_p"] == pytest.approx(2 / 64)
    assert result["effect_size_r"] == 1.0


def test_wilcoxon_fewer_than_five_units_is_reported_untested(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _shifted([0.1] * 6))
    result = wilcoxon_comparison("C1", "C2", ["D1", "D2"], str(tmp_path))
    assert result["testable"] is False
    assert result["p_value"] is None and result["statistic"] is None
    assert result["significant"] is False
    assert result["n_units"] == 2
    assert "Not tested" in result["wilcoxon_warning"]
    assert result["delta_f1"] == pytest.approx(0.1)


def test_wilcoxon_minimum_units_constant_is_five() -> None:
    assert _MIN_PAIRED_OBSERVATIONS == 5


def test_wilcoxon_delta_is_mean_b_minus_mean_a(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.2] * 6))
    assert result["delta_f1"] == pytest.approx(result["mean_f1_b"] - result["mean_f1_a"])
    assert result["delta_f1"] == pytest.approx(0.20)


def test_wilcoxon_uses_two_sided_permutation_test(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _shifted([0.2] * 6))
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=1.0, pvalue=0.5)
        wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert mocked.call_args.kwargs["alternative"] == "two-sided"
    assert isinstance(mocked.call_args.kwargs["method"], PermutationMethod)


def test_wilcoxon_is_deterministic(tmp_path: Path) -> None:
    differences = [0.1, 0.2, -0.05, 0.3, 0.4, -0.15]
    first = _compare_units(tmp_path / "a", _BASE, _shifted(differences))
    second = _compare_units(tmp_path / "b", _BASE, _shifted(differences))
    assert first == second


def test_wilcoxon_matches_by_pair_name_not_order(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _BASE)
    _write_condition(tmp_path, "C1", "D3", {"a": 0.9, "b": 0.8, "c": 0.7})
    _write_condition(tmp_path, "C2", "D3", {"c": 0.7, "b": 0.8, "a": 0.9})
    result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert result["pair_ties"] == result["n_pairs"]


def test_wilcoxon_mismatched_pair_names_raise(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _BASE)
    _write_result(tmp_path, "C2", "D3", "other", 0.4)
    with pytest.raises(DataError):
        wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))


def test_wilcoxon_mismatched_dataset_columns_raise(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _BASE[:5])
    with pytest.raises(DataError, match="columns"):
        wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))


def test_wilcoxon_without_any_results_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))


def test_wilcoxon_return_keys(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.2] * 6))
    assert set(result) == {
        "condition_a",
        "condition_b",
        "units",
        "n_units",
        "n_pairs",
        "n_nonzero",
        "testable",
        "statistic",
        "p_value",
        "significant",
        "min_attainable_p",
        "mean_f1_a",
        "mean_f1_b",
        "delta_f1",
        "effect_size_r",
        "pair_wins",
        "pair_ties",
        "pair_losses",
        "wilcoxon_warning",
    }
    assert result["n_units"] == 6


# ---------------------------------------------------------------------------
# Rank-biserial effect size
# ---------------------------------------------------------------------------
def test_effect_size_follows_matched_pairs_rank_biserial(tmp_path: Path) -> None:
    # |d| ranks: 0.05->1, 0.1->2, 0.15->3, 0.2->4, 0.3->5, 0.4->6.
    # R+ = 2 + 4 + 5 + 6 = 17, R- = 1 + 3 = 4, r = (17 - 4) / 21.
    result = _compare_units(tmp_path, _BASE, _shifted([0.1, 0.2, -0.05, 0.3, 0.4, -0.15]))
    assert result["effect_size_r"] == pytest.approx(13 / 21)


def test_effect_size_agrees_with_scipy_statistic(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.1, 0.2, -0.05, 0.3, 0.4, -0.15]))
    total = 6 * 7 / 2
    assert result["statistic"] == pytest.approx(4.0)
    assert result["effect_size_r"] == pytest.approx((total - 2 * result["statistic"]) / total)


def test_effect_size_is_signed_like_delta(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([-0.01, -0.02, -0.03, -0.04, -0.05, -0.06]))
    assert result["effect_size_r"] == -1.0
    assert result["delta_f1"] < 0


def test_effect_size_drops_zero_differences(tmp_path: Path) -> None:
    # Non-zero ranks: 0.1->1, 0.2->2, 0.3->3, 0.4->4, 0.5->5; R+ = 10, R- = 5.
    result = _compare_units(tmp_path, _BASE, _shifted([0.0, 0.1, 0.2, 0.3, 0.4, -0.5]))
    assert result["effect_size_r"] == pytest.approx(1 / 3)
    assert result["n_nonzero"] == 5
    assert result["min_attainable_p"] == pytest.approx(2 / 32)


def test_effect_size_averages_tied_ranks(tmp_path: Path) -> None:
    # Three |d| = 0.1 share rank 2; R+ = 2 + 2 + 4 + 5 + 6 = 19, R- = 2.
    result = _compare_units(tmp_path, _BASE, _shifted([0.1, 0.1, -0.1, 0.2, 0.3, 0.4]))
    assert result["effect_size_r"] == pytest.approx(17 / 21)


def test_effect_size_is_not_biased_towards_large_values(tmp_path: Path) -> None:
    result = _compare_units(tmp_path, _BASE, _shifted([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]))
    assert result["effect_size_r"] == pytest.approx((9 - 12) / 21)
    assert abs(result["effect_size_r"]) < 0.5


def test_effect_size_is_computed_for_untested_comparisons(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _shifted([0.1, 0.2] + [0.0] * 4))
    result = wilcoxon_comparison("C1", "C2", ["D1", "D2"], str(tmp_path))
    assert result["effect_size_r"] == 1.0


@pytest.mark.parametrize("n_nonzero,expected", [(0, 1.0), (1, 1.0), (5, 0.0625), (6, 0.03125)])
def test_minimum_attainable_p(n_nonzero: int, expected: float) -> None:
    assert minimum_attainable_p(n_nonzero) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Wilcoxon warning capture
# ---------------------------------------------------------------------------
def _warning_wilcoxon(message: str, category: type = RuntimeWarning):
    """Return a wilcoxon double that raises ``message`` before returning."""

    def _call(*_args, **_kwargs):
        warnings.warn(message, category)
        return mock.Mock(statistic=1.0, pvalue=0.5)

    return _call


def _seed_six_units(tmp_path: Path) -> None:
    _write_units(tmp_path, "C1", _BASE)
    _write_units(tmp_path, "C2", _shifted([0.2] * 6))


def _compare_with_warning(
    tmp_path: Path, message: str, category: type = RuntimeWarning
) -> dict:
    _seed_six_units(tmp_path)
    with mock.patch(
        "kgsemembed.evaluation.stats.wilcoxon", _warning_wilcoxon(message, category)
    ):
        return wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))


def test_wilcoxon_warning_is_captured_in_result(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "zero differences detected")
    assert result["wilcoxon_warning"] == "zero differences detected"


def test_wilcoxon_warning_is_preserved_as_string(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "sample too small")
    assert isinstance(result["wilcoxon_warning"], str)


def test_wilcoxon_warning_is_none_without_warning(tmp_path: Path) -> None:
    _seed_six_units(tmp_path)
    result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert result["wilcoxon_warning"] is None


def test_wilcoxon_warning_does_not_alter_statistics(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "ties detected")
    assert result["statistic"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.5)
    assert result["delta_f1"] == pytest.approx(0.20)


def test_repeated_warnings_are_joined_in_order(tmp_path: Path) -> None:
    def _two_warnings(*_args, **_kwargs):
        warnings.warn("first problem", RuntimeWarning)
        warnings.warn("second problem", RuntimeWarning)
        return mock.Mock(statistic=1.0, pvalue=0.5)

    _seed_six_units(tmp_path)
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon", _two_warnings):
        result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert result["wilcoxon_warning"] == "first problem; second problem"


def test_user_warning_is_treated_as_a_reliability_warning(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "zero differences", UserWarning)
    assert result["wilcoxon_warning"] == "zero differences"


def test_unrelated_warning_is_reissued_not_captured(tmp_path: Path) -> None:
    _seed_six_units(tmp_path)
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with mock.patch(
            "kgsemembed.evaluation.stats.wilcoxon",
            _warning_wilcoxon("api change", DeprecationWarning),
        ):
            result = wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))

    assert result["wilcoxon_warning"] is None
    assert [str(entry.message) for entry in escaped] == ["api change"]


def test_capture_does_not_change_global_warning_filters(tmp_path: Path) -> None:
    _seed_six_units(tmp_path)
    before = list(warnings.filters)
    wilcoxon_comparison("C1", "C2", _ALL_DATASETS, str(tmp_path))
    assert warnings.filters == before


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------
def _seed_group(results_dir: Path, group: str) -> None:
    """Write one result per column for every condition of a group, B ahead of A."""
    for comparison in _GROUP_COMPARISONS[group]:
        for offset, condition_id in enumerate((comparison.condition_a, comparison.condition_b)):
            for index, unit in enumerate(expand_dataset_ids(list(comparison.datasets))):
                f1 = 0.3 + 0.01 * index + 0.1 * offset
                _write_result(results_dir, condition_id, unit, f"{unit}_pair", f1)


@pytest.mark.parametrize(
    "group,expected",
    [("A", 3), ("B", 4), ("C", 3), ("D", 2), ("E", 1)],
)
def test_group_comparison_counts(tmp_path: Path, group: str, expected: int) -> None:
    _seed_group(tmp_path, group)
    results = run_group_comparisons(group, str(tmp_path))
    assert len(results) == expected


def test_group_comparisons_carry_warning_field(tmp_path: Path) -> None:
    _seed_group(tmp_path, "D")
    for result in run_group_comparisons("D", str(tmp_path)):
        assert "wilcoxon_warning" in result


def test_untestable_comparisons_do_not_abort_the_group(tmp_path: Path) -> None:
    _seed_group(tmp_path, "D")
    results = run_group_comparisons("D", str(tmp_path))
    assert [result["testable"] for result in results] == [False, False]
    assert all(result["corrected_significant"] is False for result in results)


def test_group_results_contain_correction_fields(tmp_path: Path) -> None:
    _seed_group(tmp_path, "B")
    results = run_group_comparisons("B", str(tmp_path))
    required = {
        "condition_a",
        "condition_b",
        "n_units",
        "p_value",
        "significant",
        "corrected_significant",
        "alpha_corrected",
        "underpowered",
        "delta_f1",
        "effect_size_r",
    }
    for result in results:
        assert required.issubset(set(result))
        assert result["alpha_corrected"] == pytest.approx(0.05 / 4)


def test_group_comparisons_match_canonical_matrix(tmp_path: Path) -> None:
    expected = {
        "A": [("C1", "C2"), ("C1", "C17"), ("C2", "C17")],
        "B": [("C3", "C10"), ("C9", "C10"), ("C12", "C1"), ("C18", "C10")],
        "C": [("C10", "C13"), ("C11", "C13"), ("C6", "C16")],
        "D": [("C5", "C14"), ("C10", "C15")],
        "E": [("C20", "C10")],
    }
    for group, pairs in expected.items():
        _seed_group(tmp_path / group, group)
        results = run_group_comparisons(group, str(tmp_path / group))
        actual = [(r["condition_a"], r["condition_b"]) for r in results]
        assert actual == pairs


def test_six_unit_comparison_is_underpowered_under_a_four_way_correction(
    tmp_path: Path,
) -> None:
    _seed_group(tmp_path, "B")
    c3_vs_c10 = run_group_comparisons("B", str(tmp_path))[0]
    assert c3_vs_c10["p_value"] == pytest.approx(2 / 64)
    assert c3_vs_c10["significant"] is True
    assert c3_vs_c10["underpowered"] is True
    assert c3_vs_c10["corrected_significant"] is False


def test_group_e_alone_can_reach_significance(tmp_path: Path) -> None:
    _seed_group(tmp_path, "E")
    (result,) = run_group_comparisons("E", str(tmp_path))
    assert result["n_units"] == 6
    assert result["underpowered"] is False
    assert result["corrected_significant"] is True


def _run_group_with_fixed_pvalue(
    tmp_path: Path, group: str, p_value: float, alpha: float = 0.05
) -> List[dict]:
    _seed_group(tmp_path, group)
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=1.0, pvalue=p_value)
        results = run_group_comparisons(group, str(tmp_path), alpha=alpha)
    return [result for result in results if result["testable"]]


def test_bonferroni_uses_group_comparison_count(tmp_path: Path) -> None:
    # Group B has 4 comparisons; the corrected threshold is 0.05 / 4 = 0.0125.
    # Its C12 vs C1 comparison spans three columns and is not tested, yet it
    # still counts towards the divisor.
    below = _run_group_with_fixed_pvalue(tmp_path / "below", "B", 0.012)
    above = _run_group_with_fixed_pvalue(tmp_path / "above", "B", 0.013)
    assert len(below) == 3 and len(above) == 3
    assert all(result["corrected_significant"] is True for result in below)
    assert all(result["corrected_significant"] is False for result in above)


def test_bonferroni_correction_is_group_local(tmp_path: Path) -> None:
    # p=0.04 passes group E's alpha / 1 but fails group B's alpha / 4.
    in_e = _run_group_with_fixed_pvalue(tmp_path / "e", "E", 0.04)
    in_b = _run_group_with_fixed_pvalue(tmp_path / "b", "B", 0.04)
    assert all(result["corrected_significant"] is True for result in in_e)
    assert all(result["corrected_significant"] is False for result in in_b)


def test_corrected_significant_does_not_overwrite_significant(tmp_path: Path) -> None:
    results = _run_group_with_fixed_pvalue(tmp_path, "B", 0.03)
    for result in results:
        assert result["significant"] is True
        assert result["corrected_significant"] is False


def test_unknown_group_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_group_comparisons("Z", str(tmp_path))


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------
def _example_results() -> List[dict]:
    return [
        {
            "condition_a": "C1",
            "condition_b": "C2",
            "units": list(_UNITS),
            "n_units": 6,
            "n_pairs": 26,
            "n_nonzero": 6,
            "testable": True,
            "statistic": 3.0,
            "p_value": 0.0625,
            "significant": False,
            "min_attainable_p": 0.03125,
            "corrected_significant": False,
            "mean_f1_a": 0.40,
            "mean_f1_b": 0.42,
            "delta_f1": 0.02,
            "effect_size_r": 0.8,
            "pair_wins": 20,
            "pair_ties": 2,
            "pair_losses": 4,
            "wilcoxon_warning": None,
        }
    ]


def test_export_creates_file_and_parents(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dir" / "stats.md"
    export_stats_table(_example_results(), str(output))
    assert output.is_file()


def test_export_header_matches_schema(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_example_results(), str(output))
    content = output.read_text(encoding="utf-8")
    header = (
        "Condition A | Condition B | n | pairs (W/T/L) | p-value | min p "
        "| Corrected sig. | delta-F1 | effect_size_r | Warning"
    )
    assert header in content


def test_export_has_dedicated_effect_size_column(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_example_results(), str(output))
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(" | ").index("effect_size_r") == 8


def test_export_row_maps_values_to_columns(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_example_results(), str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1] == (
        "| C1 | C2 | 6 | 20/2/4 | 0.0625 | 0.0312 | False | 0.0200 | 0.8000 | - |"
    )


def test_export_renders_untested_p_value_as_not_applicable(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    results = _example_results()
    results[0].update(testable=False, p_value=None, statistic=None)
    export_stats_table(results, str(output))
    row = output.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert row.split(" | ")[4] == "n/a"


def test_export_one_row_per_comparison(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    results = _example_results() * 3
    export_stats_table(results, str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 + 3


def _results_with_warning(message: object) -> List[dict]:
    results = _example_results()
    results[0]["wilcoxon_warning"] = message
    return results


def test_export_surfaces_warning_message(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_results_with_warning("zero differences detected"), str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1].endswith("| zero differences detected |")


def test_export_tolerates_result_without_warning_field(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    results = _example_results()
    del results[0]["wilcoxon_warning"]
    export_stats_table(results, str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1].endswith("| - |")


def test_export_marks_absent_warning(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_results_with_warning(None), str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1].endswith("| - |")


def _cell_count(row: str) -> int:
    """Count table cells, ignoring pipes escaped for Markdown."""
    return len(re.split(r"(?<!\\)\|", row)) - 2


def test_export_keeps_warning_inside_one_cell(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_results_with_warning("ties |\nand zeros"), str(output))
    header, _, row = output.read_text(encoding="utf-8").strip().splitlines()
    assert _cell_count(row) == _cell_count(header)
    assert row.endswith("| ties \\| and zeros |")


def test_export_keeps_warning_object_inside_one_cell(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_results_with_warning(RuntimeWarning("ties | here")), str(output))
    header, _, row = output.read_text(encoding="utf-8").strip().splitlines()
    assert _cell_count(row) == _cell_count(header)
    assert row.endswith("| ties \\| here |")


@pytest.mark.parametrize("effect_size", [0.0, 0.5, 1.0, -0.25])
def test_export_renders_effect_size_verbatim(tmp_path: Path, effect_size: float) -> None:
    output = tmp_path / "stats.md"
    results = _example_results()
    results[0]["effect_size_r"] = effect_size
    export_stats_table(results, str(output))
    row = output.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert row.split(" | ")[8] == f"{effect_size:.4f}"


def test_export_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    export_stats_table(_example_results(), str(first))
    export_stats_table(_example_results(), str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
