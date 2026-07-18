"""Tests for the Phase 2 evaluation metrics module.

All tests use lightweight synthetic scored pairs and ranked lists; no real
embedding models or repository datasets are involved.
"""

import numpy as np
import pytest

import kgsemembed.evaluation as evaluation
from kgsemembed.evaluation import (
    compute_all_metrics,
    compute_f1_at_threshold,
    compute_mrr,
    compute_recall_at_k,
    tune_threshold,
)


# ---------------------------------------------------------------------------
# Package exports (regression)
# ---------------------------------------------------------------------------
def test_package_exports() -> None:
    expected = {
        "EntityPair",
        "ScoredPair",
        "RankedList",
        "compute_f1_at_threshold",
        "tune_threshold",
        "compute_mrr",
        "compute_recall_at_k",
        "compute_all_metrics",
    }
    assert expected.issubset(set(evaluation.__all__))
    for name in expected:
        assert hasattr(evaluation, name)


def test_type_alias_definitions() -> None:
    from typing import get_args

    assert get_args(evaluation.EntityPair) == (str, str)
    assert get_args(evaluation.ScoredPair) == (str, str, float)
    assert get_args(evaluation.RankedList)[0] is evaluation.ScoredPair


# ---------------------------------------------------------------------------
# Threshold-based evaluation
# ---------------------------------------------------------------------------
def test_perfect_predictions() -> None:
    scored = [("a", "x", 0.9), ("b", "y", 0.8)]
    references = [("a", "x"), ("b", "y")]
    result = compute_f1_at_threshold(scored, references, 0.5)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_no_predictions_above_threshold() -> None:
    scored = [("a", "x", 0.2), ("b", "y", 0.1)]
    references = [("a", "x"), ("b", "y")]
    result = compute_f1_at_threshold(scored, references, 0.9)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 2


def test_true_false_positive_and_negative_counts() -> None:
    scored = [("a", "x", 0.9), ("b", "wrong", 0.8)]
    references = [("a", "x"), ("c", "z")]
    result = compute_f1_at_threshold(scored, references, 0.5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_exact_uri_comparison_no_normalisation() -> None:
    scored = [("A", "X", 0.9)]
    references = [("a", "x")]
    result = compute_f1_at_threshold(scored, references, 0.5)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_score_equal_to_threshold_is_a_match() -> None:
    scored = [("a", "x", 0.5)]
    references = [("a", "x")]
    result = compute_f1_at_threshold(scored, references, 0.5)
    assert result["tp"] == 1


def test_empty_scored_pairs_and_references() -> None:
    result = compute_f1_at_threshold([], [], 0.5)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["tp"] == result["fp"] == result["fn"] == 0


def test_duplicate_scored_pairs_collapse() -> None:
    scored = [("a", "x", 0.9), ("a", "x", 0.95)]
    references = [("a", "x")]
    result = compute_f1_at_threshold(scored, references, 0.5)
    assert result["tp"] == 1
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0


def test_threshold_metrics_within_unit_interval() -> None:
    scored = [("a", "x", 0.9), ("b", "wrong", 0.8), ("c", "z", 0.3)]
    references = [("a", "x"), ("c", "z"), ("d", "w")]
    for threshold in (0.0, 0.4, 0.85, 1.0):
        result = compute_f1_at_threshold(scored, references, threshold)
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def test_tune_threshold_uses_default_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    original = evaluation.metrics.compute_f1_at_threshold

    def spy(scored_pairs, references, threshold):
        seen.append(threshold)
        return original(scored_pairs, references, threshold)

    monkeypatch.setattr(evaluation.metrics, "compute_f1_at_threshold", spy)
    tune_threshold([("a", "x", 0.9)], [("a", "x")])
    assert seen == pytest.approx(np.arange(0.1, 1.0, 0.05).tolist())


def test_tune_threshold_returns_value_within_grid_bounds() -> None:
    scored = [("a", "x", 0.9), ("b", "y", 0.4)]
    references = [("a", "x"), ("b", "y")]
    result = tune_threshold(scored, references)
    assert 0.1 <= result["best_threshold"] <= 0.95


def test_tune_threshold_best_f1_matches_selected_threshold() -> None:
    scored = [("a", "x", 0.9), ("b", "wrong", 0.3)]
    references = [("a", "x"), ("b", "y")]
    result = tune_threshold(scored, references)
    recomputed = compute_f1_at_threshold(scored, references, result["best_threshold"])
    assert result["best_f1"] == recomputed["f1"]
    assert result["precision"] == recomputed["precision"]
    assert result["recall"] == recomputed["recall"]


def test_tune_threshold_selects_maximum_f1() -> None:
    scored = [("a", "x", 0.6), ("b", "wrong", 0.3)]
    references = [("a", "x")]
    result = tune_threshold(scored, references)
    assert result["best_f1"] == 1.0
    assert 0.3 < result["best_threshold"] <= 0.6


def test_tune_threshold_tie_selects_first() -> None:
    scored = [("a", "x", 0.95)]
    references = [("a", "x")]
    thresholds = [0.2, 0.4, 0.6, 0.8]
    result = tune_threshold(scored, references, thresholds)
    assert result["best_f1"] == 1.0
    assert result["best_threshold"] == 0.2


def test_tune_threshold_top_grid_value_stays_within_bounds() -> None:
    scored = [
        ("a", "x", 0.96),
        ("b", "w1", 0.94),
        ("c", "w2", 0.93),
    ]
    references = [("a", "x")]
    result = tune_threshold(scored, references)
    assert result["best_f1"] == 1.0
    assert result["best_threshold"] == 0.95
    assert 0.1 <= result["best_threshold"] <= 0.95


def test_tune_threshold_within_unit_interval() -> None:
    scored = [("a", "x", 0.7), ("b", "wrong", 0.7)]
    references = [("a", "x"), ("b", "y")]
    result = tune_threshold(scored, references)
    for key in ("best_f1", "precision", "recall"):
        assert 0.0 <= result[key] <= 1.0


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------
def test_mrr_gold_at_rank_one() -> None:
    ranked = [[("a", "x", 0.9), ("a", "y", 0.5)]]
    references = [("a", "x")]
    assert compute_mrr(ranked, references) == 1.0


def test_mrr_gold_at_rank_two() -> None:
    ranked = [[("a", "y", 0.9), ("a", "x", 0.5)]]
    references = [("a", "x")]
    assert compute_mrr(ranked, references) == 0.5


def test_mrr_missing_gold_contributes_zero() -> None:
    ranked = [[("a", "y", 0.9), ("a", "z", 0.5)]]
    references = [("a", "x")]
    assert compute_mrr(ranked, references) == 0.0


def test_mrr_source_without_reference_contributes_zero() -> None:
    ranked = [[("a", "x", 0.9)], [("b", "y", 0.9)]]
    references = [("a", "x")]
    assert compute_mrr(ranked, references) == 0.5


def test_mrr_arithmetic_mean_over_sources() -> None:
    ranked = [
        [("a", "x", 0.9), ("a", "n", 0.5)],
        [("b", "n", 0.9), ("b", "y", 0.5)],
    ]
    references = [("a", "x"), ("b", "y")]
    assert compute_mrr(ranked, references) == pytest.approx((1.0 + 0.5) / 2)


def test_mrr_empty_ranked_lists() -> None:
    assert compute_mrr([], [("a", "x")]) == 0.0


def test_mrr_within_unit_interval() -> None:
    ranked = [[("a", "n", 0.9), ("a", "x", 0.5)]]
    references = [("a", "x")]
    assert 0.0 <= compute_mrr(ranked, references) <= 1.0


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------
def _ranked_fixture() -> list:
    return [
        [("a", f"c{i}", 1.0 - i * 0.05) for i in range(12)],
    ]


def test_recall_at_1_only_when_gold_first() -> None:
    ranked = [[("a", "x", 0.9), ("a", "y", 0.5)]]
    assert compute_recall_at_k(ranked, [("a", "x")], 1) == 1.0
    assert compute_recall_at_k(ranked, [("a", "y")], 1) == 0.0


def test_recall_at_various_k() -> None:
    ranked = _ranked_fixture()
    ranked[0][4] = ("a", "gold", 0.6)
    references = [("a", "gold")]
    assert compute_recall_at_k(ranked, references, 1) == 0.0
    assert compute_recall_at_k(ranked, references, 5) == 1.0
    assert compute_recall_at_k(ranked, references, 10) == 1.0
    assert compute_recall_at_k(ranked, references, 20) == 1.0


def test_recall_at_20_equals_all_candidates_considered() -> None:
    ranked = _ranked_fixture()
    ranked[0][9] = ("a", "gold", 0.6)
    references = [("a", "gold"), ("a", "absent")]
    full_k = len(ranked[0])
    at_20 = compute_recall_at_k(ranked, references, 20)
    at_full = compute_recall_at_k(ranked, references, full_k)
    assert at_20 == at_full == 0.5


def test_recall_at_k_larger_than_list() -> None:
    ranked = [[("a", "x", 0.9)]]
    assert compute_recall_at_k(ranked, [("a", "x")], 50) == 1.0


def test_recall_at_k_empty_ranked_lists() -> None:
    assert compute_recall_at_k([], [("a", "x")], 5) == 0.0


def test_recall_at_k_empty_references() -> None:
    assert compute_recall_at_k(_ranked_fixture(), [], 5) == 0.0


def test_recall_at_k_within_unit_interval() -> None:
    ranked = [[("a", "x", 0.9)], [("b", "n", 0.9)]]
    references = [("a", "x"), ("b", "y")]
    assert compute_recall_at_k(ranked, references, 5) == 0.5


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------
def _combined_fixture() -> tuple:
    ranked = [
        [("a", "x", 0.9), ("a", "n1", 0.4)],
        [("b", "n2", 0.85), ("b", "y", 0.6)],
    ]
    references = [("a", "x"), ("b", "y")]
    return ranked, references


def test_compute_all_metrics_contains_every_key() -> None:
    ranked, references = _combined_fixture()
    result = compute_all_metrics(ranked, references)
    expected = {
        "f1",
        "precision",
        "recall",
        "threshold",
        "mrr",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
    }
    assert set(result) == expected


def test_compute_all_metrics_tuning_path_matches_components() -> None:
    ranked, references = _combined_fixture()
    scored = [pair for group in ranked for pair in group]
    tuned = tune_threshold(scored, references)
    result = compute_all_metrics(ranked, references)
    assert result["threshold"] == tuned["best_threshold"]
    assert result["mrr"] == compute_mrr(ranked, references)
    assert result["recall_at_5"] == compute_recall_at_k(ranked, references, 5)


def test_compute_all_metrics_fixed_threshold_path() -> None:
    ranked, references = _combined_fixture()
    scored = [pair for group in ranked for pair in group]
    result = compute_all_metrics(ranked, references, threshold=0.5)
    fixed = compute_f1_at_threshold(scored, references, 0.5)
    assert result["threshold"] == 0.5
    assert result["f1"] == fixed["f1"]
    assert result["precision"] == fixed["precision"]
    assert result["recall"] == fixed["recall"]


def test_compute_all_metrics_within_unit_interval() -> None:
    ranked, references = _combined_fixture()
    result = compute_all_metrics(ranked, references)
    for value in result.values():
        assert 0.0 <= value <= 1.0
