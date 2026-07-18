"""Tests for the Phase 2 experiment condition registry."""

import pytest

from kgsemembed.embeddings.models import MODEL_REGISTRY
from kgsemembed.pipeline import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    get_condition,
    get_conditions_for_dataset,
    get_conditions_for_group,
)
from kgsemembed.pipeline.conditions import _validate_registry
from kgsemembed.verbalisation.registry import VALID_STRATEGY_NAMES

_NON_PPAS_IDS = {"C14", "C15"}


def _make_condition(**overrides) -> ExperimentCondition:
    defaults = dict(
        condition_id="C1",
        strategy_name="V1",
        model_key="M1",
        datasets=["D1"],
        apply_ppas=True,
        description="synthetic condition.",
        ablation_group="A",
    )
    defaults.update(overrides)
    return ExperimentCondition(**defaults)


def test_registry_holds_eighteen_conditions():
    assert len(EXPERIMENT_CONDITIONS) == 18


def test_condition_identifiers_are_unique():
    ids = [c.condition_id for c in EXPERIMENT_CONDITIONS]
    assert len(ids) == len(set(ids))


def test_every_strategy_is_registered():
    for condition in EXPERIMENT_CONDITIONS:
        assert condition.strategy_name in VALID_STRATEGY_NAMES


def test_every_model_is_registered():
    for condition in EXPERIMENT_CONDITIONS:
        assert condition.model_key in MODEL_REGISTRY


def test_ppas_disabled_only_for_c14_and_c15():
    disabled = {c.condition_id for c in EXPERIMENT_CONDITIONS if not c.apply_ppas}
    assert disabled == _NON_PPAS_IDS


def test_all_other_conditions_enable_ppas():
    for condition in EXPERIMENT_CONDITIONS:
        if condition.condition_id not in _NON_PPAS_IDS:
            assert condition.apply_ppas is True


def test_get_condition_returns_matching_condition():
    condition = get_condition("C1")
    assert isinstance(condition, ExperimentCondition)
    assert condition.condition_id == "C1"


def test_get_condition_strategy_lookup():
    assert get_condition("C10").strategy_name == "V2+V8"


def test_get_condition_ppas_flags():
    assert get_condition("C14").apply_ppas is False
    assert get_condition("C15").apply_ppas is False
    assert get_condition("C1").apply_ppas is True


def test_get_condition_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_condition("C999")


def test_get_conditions_for_dataset_includes_expected():
    ids = [c.condition_id for c in get_conditions_for_dataset("D5")]
    assert "C1" in ids
    assert "C14" in ids


def test_get_conditions_for_dataset_preserves_registry_order():
    expected = [c for c in EXPERIMENT_CONDITIONS if "D5" in c.datasets]
    assert get_conditions_for_dataset("D5") == expected


def test_get_conditions_for_dataset_unknown_returns_empty():
    assert get_conditions_for_dataset("D99") == []


def test_get_conditions_for_group_d_is_c14_and_c15():
    ids = [c.condition_id for c in get_conditions_for_group("D")]
    assert ids == ["C14", "C15"]


@pytest.mark.parametrize(
    "group, expected_ids",
    [
        ("A", ["C1", "C2", "C17"]),
        ("B", ["C3", "C9", "C10", "C12", "C18"]),
        ("C", ["C4", "C5", "C6", "C7", "C8", "C11", "C13", "C16"]),
    ],
)
def test_get_conditions_for_group_membership(group, expected_ids):
    ids = [c.condition_id for c in get_conditions_for_group(group)]
    assert ids == expected_ids


def test_get_conditions_for_group_preserves_registry_order():
    for group in {c.ablation_group for c in EXPERIMENT_CONDITIONS}:
        expected = [c for c in EXPERIMENT_CONDITIONS if c.ablation_group == group]
        assert get_conditions_for_group(group) == expected


def test_get_conditions_for_group_unknown_returns_empty():
    assert get_conditions_for_group("Z") == []


def test_validation_rejects_wrong_count():
    with pytest.raises(ValueError):
        _validate_registry([_make_condition()])


def test_validation_rejects_duplicate_ids():
    conditions = [_make_condition() for _ in range(18)]
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_unknown_strategy():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(18)]
    conditions[0] = _make_condition(condition_id="C0", strategy_name="V999")
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_unknown_model():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(18)]
    conditions[0] = _make_condition(condition_id="C0", model_key="M999")
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_incorrect_ppas_assignment():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(18)]
    conditions[0] = _make_condition(condition_id="C0", apply_ppas=False)
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_package_exports_are_available():
    import kgsemembed.pipeline as pipeline

    for name in (
        "EXPERIMENT_CONDITIONS",
        "ExperimentCondition",
        "get_condition",
        "get_conditions_for_dataset",
        "get_conditions_for_group",
    ):
        assert hasattr(pipeline, name)
