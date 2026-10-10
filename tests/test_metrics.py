"""Tests for the Phase 2 evaluation metrics module.

All tests use lightweight synthetic scored pairs and ranked lists; no real
embedding models or repository datasets are involved.
"""

import numpy as np
import pytest

import kgsemembed.evaluation as evaluation
import kgsemembed.pipeline.run_experiment as runner
from kgsemembed.evaluation import (
    compute_all_metrics,
    compute_f1_at_threshold,
    compute_mrr,
    compute_recall_at_k,
    top_ranked_pairs,
    tune_threshold,
)
from kgsemembed.evaluation.metrics import _median_threshold, _reference_map
from kgsemembed.pipeline.conditions import get_condition
from kgsemembed.pipeline.run_experiment import _resolve_threshold


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
        "top_ranked_pairs",
        "EVALUATION_PROTOCOL_VERSION",
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
    assert seen == pytest.approx(np.arange(0.10, 0.995, 0.01).tolist())
    assert seen[0] == 0.10 and seen[-1] == 0.99 and len(seen) == 90


def test_default_grid_values_are_exact_two_decimal_floats() -> None:
    grid = evaluation.metrics._default_thresholds()
    assert grid == [round(value, 2) for value in grid]


def test_tune_threshold_returns_value_within_grid_bounds() -> None:
    scored = [("a", "x", 0.9), ("b", "y", 0.4)]
    references = [("a", "x"), ("b", "y")]
    result = tune_threshold(scored, references)
    assert 0.1 <= result["best_threshold"] <= 0.99


def test_tune_threshold_rejects_empty_grid() -> None:
    with pytest.raises(ValueError):
        tune_threshold([("a", "x", 0.9)], [("a", "x")], [])


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


def test_tune_threshold_tie_selects_lower_median() -> None:
    scored = [("a", "x", 0.95)]
    references = [("a", "x")]
    thresholds = [0.2, 0.4, 0.6, 0.8]
    result = tune_threshold(scored, references, thresholds)
    assert result["best_f1"] == 1.0
    assert result["best_threshold"] == 0.4


def test_tune_threshold_tie_with_odd_count_selects_middle() -> None:
    result = tune_threshold([("a", "x", 0.95)], [("a", "x")], [0.2, 0.4, 0.6])
    assert result["best_threshold"] == 0.4


def test_tune_threshold_single_winner_is_not_moved_by_tie_break() -> None:
    scored = [("a", "x", 0.65), ("b", "w", 0.45)]
    result = tune_threshold(scored, [("a", "x")], [0.4, 0.5, 0.7])
    assert result["best_threshold"] == 0.5


def test_median_threshold_of_one_value_is_that_value() -> None:
    assert _median_threshold([0.3]) == 0.3


def test_tune_threshold_reaches_above_former_grid_ceiling() -> None:
    scored = [
        ("a", "x", 0.985),
        ("b", "w1", 0.97),
        ("c", "w2", 0.96),
    ]
    references = [("a", "x")]
    result = tune_threshold(scored, references)
    assert result["best_f1"] == 1.0
    assert 0.97 < result["best_threshold"] <= 0.98


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


def test_mrr_ignores_ranked_source_without_reference() -> None:
    ranked = [[("a", "x", 0.9)], [("b", "y", 0.9)]]
    references = [("a", "x")]
    assert compute_mrr(ranked, references) == 1.0


def test_mrr_reference_source_without_ranked_list_contributes_zero() -> None:
    ranked = [[("a", "x", 0.9)]]
    references = [("a", "x"), ("b", "y")]
    assert compute_mrr(ranked, references) == 0.5


def test_mrr_empty_references() -> None:
    assert compute_mrr([[("a", "x", 0.9)]], []) == 0.0


def test_mrr_is_at_least_recall_at_1_for_one_to_one_references() -> None:
    ranked = [
        [("a", "x", 0.9), ("a", "n", 0.5)],
        [("b", "n", 0.9), ("b", "y", 0.5)],
        [("c", "z", 0.9)],
        [("u", "v", 0.8)],
    ]
    references = [("a", "x"), ("b", "y"), ("c", "z"), ("d", "w")]
    assert compute_mrr(ranked, references) >= compute_recall_at_k(ranked, references, 1)


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
# 1:N reference alignments
# ---------------------------------------------------------------------------
def test_reference_map_retains_every_gold_target() -> None:
    references = [("a", "x"), ("a", "y"), ("b", "z")]
    assert _reference_map(references) == {"a": ["x", "y"], "b": ["z"]}


def test_reference_map_deduplicates_repeated_targets() -> None:
    assert _reference_map([("a", "x"), ("a", "x")]) == {"a": ["x"]}


def test_reference_map_preserves_single_target_behaviour() -> None:
    assert _reference_map([("a", "x")]) == {"a": ["x"]}


def test_mrr_uses_best_ranked_gold_target() -> None:
    ranked = [[("a", "y", 0.9), ("a", "x", 0.5)]]
    references = [("a", "x"), ("a", "y")]
    assert compute_mrr(ranked, references) == 1.0


def test_mrr_second_gold_target_used_when_first_is_absent() -> None:
    ranked = [[("a", "n", 0.9), ("a", "y", 0.5)]]
    references = [("a", "x"), ("a", "y")]
    assert compute_mrr(ranked, references) == 0.5


def test_mrr_zero_when_no_gold_target_is_ranked() -> None:
    ranked = [[("a", "n", 0.9), ("a", "m", 0.5)]]
    references = [("a", "x"), ("a", "y")]
    assert compute_mrr(ranked, references) == 0.0


def test_mrr_one_to_n_scores_at_least_as_high_as_one_to_one() -> None:
    ranked = [[("a", "y", 0.9), ("a", "x", 0.5)]]
    single = compute_mrr(ranked, [("a", "x")])
    multi = compute_mrr(ranked, [("a", "x"), ("a", "y")])
    assert multi >= single


def test_recall_at_k_counts_every_gold_target() -> None:
    ranked = [[("a", "x", 0.9), ("a", "y", 0.5), ("a", "n", 0.1)]]
    references = [("a", "x"), ("a", "y")]
    assert compute_recall_at_k(ranked, references, 1) == 0.5
    assert compute_recall_at_k(ranked, references, 2) == 1.0


def test_recall_at_k_partial_credit_for_one_to_n() -> None:
    ranked = [[("a", "x", 0.9), ("a", "n", 0.5)]]
    references = [("a", "x"), ("a", "absent")]
    assert compute_recall_at_k(ranked, references, 5) == 0.5


# ---------------------------------------------------------------------------
# Validation-only threshold selection
# ---------------------------------------------------------------------------
def test_resolve_threshold_tunes_on_top_ranked_validation_pairs() -> None:
    val_lists = [[("a", "x", 0.9), ("a", "n", 0.85)], [("u", "w", 0.6)]]
    val_refs = [("a", "x")]
    expected = tune_threshold([("a", "x", 0.9), ("u", "w", 0.6)], val_refs)
    assert _resolve_threshold(val_lists, val_refs) == expected["best_threshold"]


def test_resolve_threshold_is_pushed_up_by_unmatched_validation_sources() -> None:
    val_refs = [("a", "x")]
    alone = _resolve_threshold([[("a", "x", 0.9)]], val_refs)
    with_unmatched = _resolve_threshold([[("a", "x", 0.9)], [("u", "w", 0.7)]], val_refs)
    assert with_unmatched > 0.7
    assert alone <= with_unmatched


def test_resolve_threshold_defaults_when_validation_empty() -> None:
    assert _resolve_threshold([[("a", "x", 0.9)]], []) == 0.5


def test_resolve_threshold_skips_tuning_when_validation_empty(monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("tune_threshold must not run without validation refs")

    monkeypatch.setattr(runner, "tune_threshold", _fail)
    assert _resolve_threshold([[("a", "x", 0.9)]], []) == 0.5


def test_resolve_threshold_warns_when_validation_empty(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        runner._LOGGER, "warning", lambda message, *args: warnings.append(message % args)
    )
    _resolve_threshold([[("a", "x", 0.9)]], [])
    assert len(warnings) == 1
    assert "validation" in warnings[0].lower()


# ---------------------------------------------------------------------------
# Experiment-level seeding
# ---------------------------------------------------------------------------
def test_seed_random_state_sets_python_and_numpy_generators() -> None:
    import random

    runner._seed_random_state()
    observed = (random.random(), float(np.random.random()))

    random.seed(42)
    np.random.seed(42)
    assert observed == (random.random(), float(np.random.random()))


def test_condition_run_is_isolated_from_preceding_conditions(monkeypatch) -> None:
    import random

    condition = get_condition("C1")
    runs: list[list[float]] = []
    model_info = {
        "model_id": "test",
        "model_key": "M1",
        "device": "cpu",
        "hf_revision": "abc123",
    }

    def _record(*args, **kwargs):
        runs[-1].append(random.random())
        return None

    monkeypatch.setattr(runner, "_run_dataset", _record)
    for hostile_seed in (999, 12345):
        random.seed(hostile_seed)  # state a preceding condition would leave behind
        runs.append([])
        runner._run_condition_with_encoder(
            condition, object(), None, "data/", "results/", False, model_info
        )

    assert runs[0] and runs[0] == runs[1]


def test_run_condition_seeds_before_loading_the_model(monkeypatch) -> None:
    import random

    seeded_before_load: list[bool] = []

    def _fake_load(model_key):
        seeded_before_load.append(random.random() == _seeded_first_draw())
        raise RuntimeError("stop after seeding check")

    monkeypatch.setattr(runner, "load_sentence_transformer", _fake_load)
    random.seed(999)
    with pytest.raises(RuntimeError):
        runner.run_condition("C1")
    assert seeded_before_load == [True]


def _seeded_first_draw() -> float:
    import random

    generator = random.Random(42)
    return generator.random()


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
        "tp",
        "fp",
        "fn",
    }
    assert set(result) == expected


def test_top_ranked_pairs_keeps_first_candidate_per_source() -> None:
    ranked, _ = _combined_fixture()
    assert top_ranked_pairs(ranked + [[]]) == [("a", "x", 0.9), ("b", "n2", 0.85)]


def test_compute_all_metrics_predicts_one_target_per_source() -> None:
    ranked = [[("a", "x", 0.9), ("a", "n1", 0.8), ("a", "n2", 0.7)]]
    result = compute_all_metrics(ranked, [("a", "x")], threshold=0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert result["precision"] == 1.0


def test_compute_all_metrics_lower_ranked_gold_is_not_a_prediction() -> None:
    ranked = [[("b", "n2", 0.85), ("b", "y", 0.6)]]
    result = compute_all_metrics(ranked, [("b", "y")], threshold=0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["recall_at_5"] == 1.0


def test_compute_all_metrics_counts_reconcile_with_references() -> None:
    ranked, references = _combined_fixture()
    result = compute_all_metrics(ranked, references, threshold=0.5)
    assert result["tp"] + result["fn"] == len(references)
    assert result["tp"] + result["fp"] == len(ranked)


def test_compute_all_metrics_tuning_path_matches_components() -> None:
    ranked, references = _combined_fixture()
    scored = top_ranked_pairs(ranked)
    tuned = tune_threshold(scored, references)
    result = compute_all_metrics(ranked, references)
    assert result["threshold"] == tuned["best_threshold"]
    assert result["mrr"] == compute_mrr(ranked, references)
    assert result["recall_at_5"] == compute_recall_at_k(ranked, references, 5)


def test_compute_all_metrics_fixed_threshold_path() -> None:
    ranked, references = _combined_fixture()
    scored = top_ranked_pairs(ranked)
    result = compute_all_metrics(ranked, references, threshold=0.5)
    fixed = compute_f1_at_threshold(scored, references, 0.5)
    assert result["threshold"] == 0.5
    assert result["f1"] == fixed["f1"]
    assert result["precision"] == fixed["precision"]
    assert result["recall"] == fixed["recall"]


def test_compute_all_metrics_within_unit_interval() -> None:
    ranked, references = _combined_fixture()
    result = compute_all_metrics(ranked, references)
    for key, value in result.items():
        if key not in ("tp", "fp", "fn"):
            assert 0.0 <= value <= 1.0
