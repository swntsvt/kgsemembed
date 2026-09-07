"""Phase 2 pipeline package: experiment condition registry and entrypoint."""

from typing import Any

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
    "run_condition",
    "run_all_conditions",
]

_LAZY_EXPORTS = {"run_condition", "run_all_conditions"}


def __getattr__(name: str) -> Any:
    """Lazily expose the runner so ``python -m`` avoids a double import."""
    if name in _LAZY_EXPORTS:
        from kgsemembed.pipeline import run_experiment

        return getattr(run_experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
