"""Phase 2 pipeline package: experiment condition registry and entrypoint."""

from kgsemembed.pipeline.conditions import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    get_condition,
    get_conditions_for_dataset,
    get_conditions_for_group,
)

__all__ = [
    "EXPERIMENT_CONDITIONS",
    "ExperimentCondition",
    "get_condition",
    "get_conditions_for_dataset",
    "get_conditions_for_group",
]
