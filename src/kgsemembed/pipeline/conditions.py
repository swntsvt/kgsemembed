"""Centralised registry of Phase 2 experimental conditions.

This module is the single authoritative source of experiment definitions for
Phase 2: which verbalisation strategy, embedding model, datasets, PPAS setting
and ablation group each condition uses.  It is purely declarative — importing it
validates the registry against the registered verbalisation strategies and
embedding models but never loads a model, performs verbalisation, or executes an
experiment.  Invalid definitions fail fast at import time rather than surfacing
during a run.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List

from kgsemembed.embeddings.models import MODEL_REGISTRY
from kgsemembed.verbalisation.registry import VALID_STRATEGY_NAMES

_EXPECTED_CONDITION_COUNT = 19
_NON_PPAS_CONDITIONS = frozenset({"C14", "C15", "C19"})


@dataclass(frozen=True)
class ExperimentCondition:
    """
    Declarative definition of a single Phase 2 experimental condition.

    Attributes
    ----------
    condition_id : str
        Unique experiment identifier, e.g. ``"C1"``.
    strategy_name : str
        Verbalisation strategy key; must exist in ``VALID_STRATEGY_NAMES``.
    model_key : str
        Embedding model key; must exist in ``MODEL_REGISTRY``.
    datasets : List[str]
        Dataset identifiers this condition is evaluated on.
    apply_ppas : bool
        Whether Predicate-Priority Adaptive Sampling is applied.
    description : str
        Human-readable summary of the condition's purpose.
    ablation_group : str
        Ablation group this condition belongs to, e.g. ``"A"``.
    """

    condition_id: str
    strategy_name: str
    model_key: str
    datasets: List[str]
    apply_ppas: bool
    description: str
    ablation_group: str


EXPERIMENT_CONDITIONS: List[ExperimentCondition] = [
    ExperimentCondition(
        condition_id="C1",
        strategy_name="V1",
        model_key="M1",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Label-only lower bound on MiniLM.",
        ablation_group="A",
    ),
    ExperimentCondition(
        condition_id="C2",
        strategy_name="V2",
        model_key="M1",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Annotation lower bound on MiniLM.",
        ablation_group="A",
    ),
    ExperimentCondition(
        condition_id="C17",
        strategy_name="V1",
        model_key="M4",
        datasets=["D1", "D2"],
        apply_ppas=True,
        description="Label-only lower bound on BioLORD.",
        ablation_group="A",
    ),
    ExperimentCondition(
        condition_id="C3",
        strategy_name="V2+V6",
        model_key="M2",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Annotation with schema-aware context on BGE (V8 ablation control).",
        ablation_group="B",
    ),
    ExperimentCondition(
        condition_id="C9",
        strategy_name="V8",
        model_key="M2",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Relational-signature only on BGE.",
        ablation_group="B",
    ),
    ExperimentCondition(
        condition_id="C10",
        strategy_name="V2+V8",
        model_key="M2",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Annotation with relational signature on BGE.",
        ablation_group="B",
    ),
    ExperimentCondition(
        condition_id="C12",
        strategy_name="V8",
        model_key="M2",
        datasets=["D3", "D4"],
        apply_ppas=True,
        description="Relational-signature only on BGE over a dataset subset.",
        ablation_group="B",
    ),
    ExperimentCondition(
        condition_id="C18",
        strategy_name="V2+V8",
        model_key="M1",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Annotation with relational signature on MiniLM.",
        ablation_group="B",
    ),
    ExperimentCondition(
        condition_id="C4",
        strategy_name="V3",
        model_key="M2",
        datasets=["D1", "D2", "D3"],
        apply_ppas=True,
        description="Template natural-language verbalisation on BGE.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C5",
        strategy_name="V4",
        model_key="M2",
        datasets=["D4", "D5"],
        apply_ppas=True,
        description="Structured key-value verbalisation on BGE.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C6",
        strategy_name="V2+V7",
        model_key="M4",
        datasets=["D1", "D2"],
        apply_ppas=True,
        description="Annotation with hierarchical context on BioLORD.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C7",
        strategy_name="V6+V3",
        model_key="M2",
        datasets=["D3", "D4"],
        apply_ppas=True,
        description="Schema-aware and template natural-language verbalisation on BGE.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C8",
        strategy_name="V5",
        model_key="M2",
        datasets=["D4", "D5"],
        apply_ppas=True,
        description="Neighbourhood-walk verbalisation on BGE.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C11",
        strategy_name="V2+V8",
        model_key="M4",
        datasets=["D1", "D2"],
        apply_ppas=True,
        description="Annotation with relational signature on BioLORD.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C13",
        strategy_name="V2+V8",
        model_key="M5",
        datasets=["D1", "D2", "D3", "D4", "D5"],
        apply_ppas=True,
        description="Annotation with relational signature on Stella.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C16",
        strategy_name="V2+V8+V7",
        model_key="M4",
        datasets=["D1", "D2"],
        apply_ppas=True,
        description="Annotation, relational signature and hierarchical context on BioLORD.",
        ablation_group="C",
    ),
    ExperimentCondition(
        condition_id="C14",
        strategy_name="V4+V6",
        model_key="M3",
        datasets=["D5"],
        apply_ppas=False,
        description="Structured key-value and schema-aware verbalisation on GTE without PPAS.",
        ablation_group="D",
    ),
    ExperimentCondition(
        condition_id="C15",
        strategy_name="V2+V8",
        model_key="M3",
        datasets=["D1"],
        apply_ppas=False,
        description="Annotation with relational signature on GTE without PPAS.",
        ablation_group="D",
    ),
    # Note: V2+V8 does not invoke PPAS under any model (V2 and V8 are
    # annotation/relational verbalisers with no budget logic). C19 isolates the
    # model effect of M2 with ppas_budget=None and should reproduce C10 metrics.
    # A genuine PPAS ablation for V2+V8 is not possible without redesigning the
    # strategy.
    ExperimentCondition(
        condition_id="C19",
        strategy_name="V2+V8",
        model_key="M2_uncapped",
        datasets=["D1"],
        apply_ppas=False,
        description="PPAS ablation: V2+V8 on bge-large without token budget cap",
        ablation_group="D",
    ),
]


def _validate_count(conditions: List[ExperimentCondition]) -> None:
    """Raise if the registry does not hold exactly the expected condition count."""
    if len(conditions) != _EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"EXPERIMENT_CONDITIONS must contain exactly "
            f"{_EXPECTED_CONDITION_COUNT} conditions, found {len(conditions)}."
        )


def _validate_unique_ids(conditions: List[ExperimentCondition]) -> None:
    """Raise if any condition identifier is repeated."""
    counts = Counter(c.condition_id for c in conditions)
    duplicates = sorted(cid for cid, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate condition identifiers: {duplicates}.")


def _validate_references(condition: ExperimentCondition) -> None:
    """Raise if a condition references an unregistered strategy or model."""
    if condition.strategy_name not in VALID_STRATEGY_NAMES:
        raise ValueError(
            f"Condition {condition.condition_id!r} uses unknown strategy "
            f"{condition.strategy_name!r}."
        )
    if condition.model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Condition {condition.condition_id!r} uses unknown model "
            f"{condition.model_key!r}."
        )


def _validate_ppas(condition: ExperimentCondition) -> None:
    """Raise if a condition's PPAS setting does not match the invariant."""
    expected = condition.condition_id not in _NON_PPAS_CONDITIONS
    if condition.apply_ppas is not expected:
        raise ValueError(
            f"Condition {condition.condition_id!r} has apply_ppas="
            f"{condition.apply_ppas}; PPAS may be disabled only for "
            f"{sorted(_NON_PPAS_CONDITIONS)}."
        )


def _validate_registry(conditions: List[ExperimentCondition]) -> None:
    """Validate every registry invariant, raising on the first violation."""
    _validate_count(conditions)
    _validate_unique_ids(conditions)
    for condition in conditions:
        _validate_references(condition)
        _validate_ppas(condition)


_validate_registry(EXPERIMENT_CONDITIONS)

_CONDITIONS_BY_ID = {c.condition_id: c for c in EXPERIMENT_CONDITIONS}


def get_condition(condition_id: str) -> ExperimentCondition:
    """
    Return the condition registered under ``condition_id``.

    Parameters
    ----------
    condition_id : str
        Experiment identifier, e.g. ``"C1"``.

    Returns
    -------
    ExperimentCondition
        The matching condition definition.

    Raises
    ------
    KeyError
        If ``condition_id`` is not registered.
    """
    try:
        return _CONDITIONS_BY_ID[condition_id]
    except KeyError:
        raise KeyError(f"Unknown condition identifier: {condition_id!r}.") from None


def get_conditions_for_dataset(dataset_id: str) -> List[ExperimentCondition]:
    """
    Return conditions evaluated on ``dataset_id``, in registry order.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier, e.g. ``"D5"``.

    Returns
    -------
    List[ExperimentCondition]
        Matching conditions, or an empty list when none apply.
    """
    return [c for c in EXPERIMENT_CONDITIONS if dataset_id in c.datasets]


def get_conditions_for_group(group: str) -> List[ExperimentCondition]:
    """
    Return conditions in ablation ``group``, in registry order.

    Parameters
    ----------
    group : str
        Ablation group identifier, e.g. ``"A"``.

    Returns
    -------
    List[ExperimentCondition]
        Matching conditions, or an empty list when none apply.
    """
    return [c for c in EXPERIMENT_CONDITIONS if c.ablation_group == group]
