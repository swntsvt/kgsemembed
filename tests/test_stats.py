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

import kgsemembed.evaluation as evaluation
from kgsemembed.evaluation import (
    collect_f1_scores,
    export_stats_table,
    load_results_for_condition,
    run_group_comparisons,
    wilcoxon_comparison,
)
from kgsemembed.evaluation.stats import _GROUP_COMPARISONS, _MIN_PAIRED_OBSERVATIONS
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
    payload = {"pair_name": "D1_p0", "metrics": {"f1": "not-a-number"}}
    (directory / "D1_p0_results.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataError):
        collect_f1_scores("C1", ["D1"], str(tmp_path))


def test_collect_duplicate_pair_across_datasets_raises(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "shared", 0.5)
    _write_result(tmp_path, "C1", "D2", "shared", 0.6)
    with pytest.raises(DataError):
        collect_f1_scores("C1", ["D1", "D2"], str(tmp_path))


# ---------------------------------------------------------------------------
# Wilcoxon comparison
# ---------------------------------------------------------------------------
def test_wilcoxon_identical_lists_pvalue_near_one(tmp_path: Path) -> None:
    pairs = _pairs("D1", 6, 0.4)
    _write_condition(tmp_path, "C1", "D1", pairs)
    _write_condition(tmp_path, "C2", "D1", pairs)
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False
    assert result["delta_f1"] == pytest.approx(0.0)


def test_wilcoxon_fewer_than_five_pairs_raises(tmp_path: Path) -> None:
    pairs = _pairs("D1", 4, 0.4)
    _write_condition(tmp_path, "C1", "D1", pairs)
    _write_condition(tmp_path, "C2", "D1", pairs)
    with pytest.raises(ValueError):
        wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))


def test_wilcoxon_delta_is_mean_b_minus_mean_a(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["delta_f1"] == pytest.approx(
        result["mean_f1_b"] - result["mean_f1_a"]
    )
    assert result["delta_f1"] == pytest.approx(0.20)


def test_wilcoxon_uses_two_sided_alternative(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=1.0, pvalue=0.5)
        wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert mocked.call_args.kwargs["alternative"] == "two-sided"


def test_wilcoxon_matches_by_pair_name_not_order(tmp_path: Path) -> None:
    _write_condition(
        tmp_path, "C1", "D1", {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
    )
    _write_condition(
        tmp_path, "C2", "D1", {"e": 0.5, "d": 0.6, "c": 0.7, "b": 0.8, "a": 0.9}
    )
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=0.0, pvalue=1.0)
        wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    scores_a, scores_b = mocked.call_args.args
    assert scores_a == scores_b


def test_wilcoxon_mismatched_pair_names_raise(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.4))
    _write_condition(
        tmp_path, "C2", "D1", {f"other_{i}": 0.4 for i in range(6)}
    )
    with pytest.raises(DataError):
        wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))


def test_wilcoxon_return_keys(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert set(result) == {
        "condition_a",
        "condition_b",
        "n_pairs",
        "statistic",
        "p_value",
        "significant",
        "mean_f1_a",
        "mean_f1_b",
        "delta_f1",
        "effect_size_r",
        "wilcoxon_warning",
    }
    assert result["n_pairs"] == 6


# ---------------------------------------------------------------------------
# Rank-biserial effect size
# ---------------------------------------------------------------------------
def _compare_with_statistic(tmp_path: Path, statistic: float) -> dict:
    """Compare six-pair conditions with a fixed Wilcoxon statistic."""
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=statistic, pvalue=0.5)
        return wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))


def test_effect_size_follows_rank_biserial_formula(tmp_path: Path) -> None:
    # n = 6 pairs and W = 3 give r = 1 - (2 * 3) / (6 * 7) = 0.857142...
    result = _compare_with_statistic(tmp_path, 3.0)
    assert result["effect_size_r"] == pytest.approx(1.0 - 6.0 / 42.0)


def test_effect_size_derives_from_returned_statistic_and_n(tmp_path: Path) -> None:
    result = _compare_with_statistic(tmp_path, 7.0)
    statistic, n_pairs = result["statistic"], result["n_pairs"]
    assert result["effect_size_r"] == pytest.approx(
        1.0 - (2.0 * statistic) / (n_pairs * (n_pairs + 1))
    )


def test_effect_size_is_one_when_statistic_is_zero(tmp_path: Path) -> None:
    result = _compare_with_statistic(tmp_path, 0.0)
    assert result["effect_size_r"] == pytest.approx(1.0)


def test_effect_size_is_zero_at_the_full_rank_sum(tmp_path: Path) -> None:
    # W = n(n + 1) / 2 = 21 is the formula's lower bound for six pairs; a
    # two-sided test never returns it, so the statistic is injected directly.
    result = _compare_with_statistic(tmp_path, 21.0)
    assert result["effect_size_r"] == pytest.approx(0.0)


def test_effect_size_falls_as_the_statistic_rises(tmp_path: Path) -> None:
    small = _compare_with_statistic(tmp_path / "small", 2.0)
    large = _compare_with_statistic(tmp_path / "large", 8.0)
    assert small["effect_size_r"] > large["effect_size_r"]


def test_effect_size_present_on_group_comparisons(tmp_path: Path) -> None:
    _seed_group(tmp_path, "D")
    for result in run_group_comparisons("D", str(tmp_path)):
        assert isinstance(result["effect_size_r"], float)


def _write_diverging_conditions(results_dir: Path, count: int) -> None:
    """Write two conditions whose paired F1 values differ in both directions."""
    scores_a = {f"p{index}": 0.10 + index * 0.10 for index in range(count)}
    scores_b = {"p0": 0.15, "p1": 0.18, "p2": 0.42, "p3": 0.44, "p4": 0.49, "p5": 0.75}
    _write_condition(results_dir, "C1", "D1", scores_a)
    _write_condition(results_dir, "C2", "D1", {name: scores_b[name] for name in scores_a})


def test_effect_size_matches_unmocked_scipy_statistic(tmp_path: Path) -> None:
    _write_diverging_conditions(tmp_path, 6)
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    statistic, n_pairs = result["statistic"], result["n_pairs"]
    assert result["effect_size_r"] == pytest.approx(
        1.0 - (2.0 * statistic) / (n_pairs * (n_pairs + 1))
    )
    assert 0.5 <= result["effect_size_r"] <= 1.0


def test_effect_size_uses_pair_count_not_dataset_count(tmp_path: Path) -> None:
    # Six pairs spread over two datasets: n must be 6, never 2.
    _write_condition(tmp_path, "C1", "D1", {"x0": 0.10, "x1": 0.20, "x2": 0.30})
    _write_condition(tmp_path, "C1", "D2", {"y0": 0.40, "y1": 0.50, "y2": 0.60})
    _write_condition(tmp_path, "C2", "D1", {"x0": 0.15, "x1": 0.18, "x2": 0.42})
    _write_condition(tmp_path, "C2", "D2", {"y0": 0.44, "y1": 0.49, "y2": 0.75})
    result = wilcoxon_comparison("C1", "C2", ["D1", "D2"], str(tmp_path))
    statistic = result["statistic"]
    assert result["n_pairs"] == 6
    assert result["effect_size_r"] == pytest.approx(1.0 - (2.0 * statistic) / 42.0)
    assert result["effect_size_r"] != pytest.approx(1.0 - (2.0 * statistic) / 6.0)


def test_effect_size_at_minimum_paired_observations(tmp_path: Path) -> None:
    _write_diverging_conditions(tmp_path, _MIN_PAIRED_OBSERVATIONS)
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["n_pairs"] == 5
    assert result["effect_size_r"] == pytest.approx(
        1.0 - (2.0 * result["statistic"]) / 30.0
    )


def test_effect_size_does_not_alter_existing_fields(tmp_path: Path) -> None:
    result = _compare_with_statistic(tmp_path, 3.0)
    assert result["statistic"] == pytest.approx(3.0)
    assert result["p_value"] == pytest.approx(0.5)
    assert result["significant"] is False
    assert result["delta_f1"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Wilcoxon warning capture
# ---------------------------------------------------------------------------
def _warning_wilcoxon(message: str, category: type = RuntimeWarning):
    """Return a wilcoxon double that raises ``message`` before returning."""

    def _call(*_args, **_kwargs):
        warnings.warn(message, category)
        return mock.Mock(statistic=1.0, pvalue=0.5)

    return _call


def _compare_with_warning(
    tmp_path: Path, message: str, category: type = RuntimeWarning
) -> dict:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    with mock.patch(
        "kgsemembed.evaluation.stats.wilcoxon", _warning_wilcoxon(message, category)
    ):
        return wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))


def test_wilcoxon_warning_is_captured_in_result(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "zero differences detected")
    assert result["wilcoxon_warning"] == "zero differences detected"


def test_wilcoxon_warning_is_preserved_as_string(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "sample too small")
    assert isinstance(result["wilcoxon_warning"], str)


def test_wilcoxon_warning_is_none_without_warning(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["wilcoxon_warning"] is None


def test_wilcoxon_warning_does_not_alter_statistics(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "ties detected")
    assert result["statistic"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.5)
    assert result["delta_f1"] == pytest.approx(0.20)


def test_scipy_runtime_warning_is_no_longer_suppressed(tmp_path: Path) -> None:
    # Identical score lists make SciPy divide by a zero-valued statistic, which
    # previously vanished behind a module-level RuntimeWarning filter.
    pairs = _pairs("D1", 6, 0.4)
    _write_condition(tmp_path, "C1", "D1", pairs)
    _write_condition(tmp_path, "C2", "D1", pairs)
    result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["wilcoxon_warning"] is not None
    assert result["p_value"] == pytest.approx(1.0)


def test_repeated_warnings_are_joined_in_order(tmp_path: Path) -> None:
    def _two_warnings(*_args, **_kwargs):
        warnings.warn("first problem", RuntimeWarning)
        warnings.warn("second problem", RuntimeWarning)
        return mock.Mock(statistic=1.0, pvalue=0.5)

    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon", _two_warnings):
        result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert result["wilcoxon_warning"] == "first problem; second problem"


def test_user_warning_is_treated_as_a_reliability_warning(tmp_path: Path) -> None:
    result = _compare_with_warning(tmp_path, "zero differences", UserWarning)
    assert result["wilcoxon_warning"] == "zero differences"


def test_unrelated_warning_is_reissued_not_captured(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with mock.patch(
            "kgsemembed.evaluation.stats.wilcoxon",
            _warning_wilcoxon("api change", DeprecationWarning),
        ):
            result = wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))

    assert result["wilcoxon_warning"] is None
    assert [str(entry.message) for entry in escaped] == ["api change"]


def test_capture_does_not_change_global_warning_filters(tmp_path: Path) -> None:
    _write_condition(tmp_path, "C1", "D1", _pairs("D1", 6, 0.30))
    _write_condition(tmp_path, "C2", "D1", _pairs("D1", 6, 0.50))
    before = list(warnings.filters)
    wilcoxon_comparison("C1", "C2", ["D1"], str(tmp_path))
    assert warnings.filters == before


def test_group_comparisons_carry_warning_field(tmp_path: Path) -> None:
    _seed_group(tmp_path, "D")
    for result in run_group_comparisons("D", str(tmp_path)):
        assert "wilcoxon_warning" in result


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------
def _seed_group(results_dir: Path, group: str) -> None:
    """Write identical six-pair results for every condition used in a group."""
    for comparison in _GROUP_COMPARISONS[group]:
        for condition_id in (comparison.condition_a, comparison.condition_b):
            for dataset_id in comparison.datasets:
                _write_condition(
                    results_dir, condition_id, dataset_id, _pairs(dataset_id, 6, 0.4)
                )


@pytest.mark.parametrize(
    "group,expected",
    [("A", 3), ("B", 4), ("C", 3), ("D", 2)],
)
def test_group_comparison_counts(tmp_path: Path, group: str, expected: int) -> None:
    _seed_group(tmp_path, group)
    results = run_group_comparisons(group, str(tmp_path))
    assert len(results) == expected


def test_group_results_contain_corrected_significant(tmp_path: Path) -> None:
    _seed_group(tmp_path, "B")
    results = run_group_comparisons("B", str(tmp_path))
    required = {
        "condition_a",
        "condition_b",
        "n_pairs",
        "statistic",
        "p_value",
        "significant",
        "corrected_significant",
        "mean_f1_a",
        "mean_f1_b",
        "delta_f1",
    }
    for result in results:
        assert required.issubset(set(result))


def test_group_comparisons_match_canonical_matrix(tmp_path: Path) -> None:
    expected = {
        "A": [("C1", "C2"), ("C1", "C17"), ("C2", "C17")],
        "B": [("C3", "C10"), ("C9", "C10"), ("C12", "C1"), ("C18", "C10")],
        "C": [("C10", "C13"), ("C11", "C13"), ("C6", "C16")],
        "D": [("C5", "C14"), ("C10", "C15")],
    }
    for group, pairs in expected.items():
        _seed_group(tmp_path, group)
        results = run_group_comparisons(group, str(tmp_path))
        actual = [(r["condition_a"], r["condition_b"]) for r in results]
        assert actual == pairs


def _run_group_with_fixed_pvalue(
    tmp_path: Path, group: str, p_value: float, alpha: float = 0.05
) -> List[dict]:
    _seed_group(tmp_path, group)
    with mock.patch("kgsemembed.evaluation.stats.wilcoxon") as mocked:
        mocked.return_value = mock.Mock(statistic=1.0, pvalue=p_value)
        return run_group_comparisons(group, str(tmp_path), alpha=alpha)


def test_bonferroni_uses_group_comparison_count(tmp_path: Path) -> None:
    # Group B has 4 comparisons; the corrected threshold is 0.05 / 4 = 0.0125.
    # p=0.012 is below it and p=0.013 above it, so this pair of assertions is
    # only satisfied when the divisor is exactly the group's comparison count.
    below = _run_group_with_fixed_pvalue(tmp_path / "below", "B", 0.012)
    above = _run_group_with_fixed_pvalue(tmp_path / "above", "B", 0.013)
    assert len(below) == 4 and len(above) == 4
    assert all(result["corrected_significant"] is True for result in below)
    assert all(result["corrected_significant"] is False for result in above)


def test_bonferroni_correction_is_group_local(tmp_path: Path) -> None:
    # Group D has 2 comparisons; corrected threshold is 0.05 / 2 = 0.025.
    # p=0.02 is significant here, but would not be under a larger (cross-group)
    # divisor, confirming correction is applied within the group only.
    results = _run_group_with_fixed_pvalue(tmp_path, "D", 0.02)
    assert all(result["corrected_significant"] is True for result in results)


def test_corrected_significant_does_not_overwrite_significant(tmp_path: Path) -> None:
    # p=0.03 is significant at alpha=0.05 but not at the corrected 0.05/4.
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
            "n_pairs": 5,
            "statistic": 3.0,
            "p_value": 0.0625,
            "significant": False,
            "corrected_significant": False,
            "mean_f1_a": 0.40,
            "mean_f1_b": 0.42,
            "delta_f1": 0.02,
            "effect_size_r": 0.8,
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
        "Condition A | Condition B | n | p-value | Corrected sig. | delta-F1 "
        "| effect_size_r | Warning"
    )
    assert header in content


def test_export_has_dedicated_effect_size_column(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_example_results(), str(output))
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(" | ").index("effect_size_r") == 6


def test_export_row_maps_values_to_columns(tmp_path: Path) -> None:
    output = tmp_path / "stats.md"
    export_stats_table(_example_results(), str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1] == "| C1 | C2 | 5 | 0.0625 | False | 0.0200 | 0.8000 | - |"


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
    assert row.split(" | ")[6] == f"{effect_size:.4f}"


def test_export_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    export_stats_table(_example_results(), str(first))
    export_stats_table(_example_results(), str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
