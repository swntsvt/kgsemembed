"""Registry of the controlled PPAS ablation conditions (#122).

The ablation is kept apart from :mod:`kgsemembed.pipeline.conditions`, so the
historical C1-C20 registry, its count invariant, the main results tables and the
ablation-group statistics are unchanged.  Every condition here uses the same
model (M2, ``BAAI/bge-large-en-v1.5``), the same structured key-value
representation, and the same candidate files; within one budget the selection
policy, plus the seed of the random policy, is the only difference.

Budgets
-------
``PRIMARY_BUDGET`` is M2's nominal 420-token budget, now counted with the real
tokenizer.  At 420, sampling is invoked for 2 of 14,762 D5 sources and 241 of
13,656 D5 candidates, and for no D1 entity, so D1 is not run at 420.

``STRESS_BUDGETS`` were fixed before any ablation run by an outcome-blind rule
on input lengths only: the median full-representation length over a dataset's
pooled source and candidate entities, floored to a multiple of 8.  D5 gives 80
(sampling 28% of sources, 82% of candidates) and D1 gives 48 (25% / 81%).
``scripts/ppas_ablation_inventory.py`` recomputes the rule.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from kgsemembed.embeddings.models import get_model_config
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC,
    FULL,
    PPAS,
    RANDOM,
    SELECTION_POLICIES,
)

ABLATION_MODEL_KEY = "M2"
PRIMARY_BUDGET = 420
STRESS_BUDGETS = {"D5": 80, "D1": 48}
RANDOM_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
STRESS_BUDGET_MULTIPLE = 8

_POLICY_LABELS = {PPAS: "ppas", DETERMINISTIC: "det", RANDOM: "rand"}


@dataclass(frozen=True)
class AblationCondition:
    """
    One selection policy at one token budget.

    Attributes
    ----------
    condition_id : str
        Identifier, e.g. ``"PA-ppas-420"`` or ``"PA-rand-420-s3"``.
    policy : str
        One of :data:`~kgsemembed.verbalisation.selection.SELECTION_POLICIES`.
    token_budget : Optional[int]
        Encoded-length budget, special tokens included; ``None`` for ``"full"``.
    seed : Optional[int]
        Seed of the random policy, ``None`` otherwise.
    datasets : Tuple[str, ...]
        Datasets the condition runs on.
    model_key : str
        Embedding model, shared by every ablation condition.
    """

    condition_id: str
    policy: str
    token_budget: Optional[int]
    seed: Optional[int]
    datasets: Tuple[str, ...]
    model_key: str = ABLATION_MODEL_KEY


def _budget_family(budget: int, datasets: Tuple[str, ...]) -> List[AblationCondition]:
    """Return the PPAS, deterministic and seeded random conditions of a budget."""
    family = [
        AblationCondition(f"PA-{_POLICY_LABELS[policy]}-{budget}", policy, budget, None, datasets)
        for policy in (PPAS, DETERMINISTIC)
    ]
    family += [
        AblationCondition(f"PA-rand-{budget}-s{seed}", RANDOM, budget, seed, datasets)
        for seed in RANDOM_SEEDS
    ]
    return family


ABLATION_CONDITIONS: List[AblationCondition] = [
    AblationCondition("PA-full", FULL, None, None, ("D5", "D1")),
    *_budget_family(PRIMARY_BUDGET, ("D5",)),
    *_budget_family(STRESS_BUDGETS["D5"], ("D5",)),
    *_budget_family(STRESS_BUDGETS["D1"], ("D1",)),
]


def _validate_condition(condition: AblationCondition) -> None:
    """Raise if a condition's policy, budget, seed or model is inconsistent."""
    if condition.policy not in SELECTION_POLICIES:
        raise ValueError(f"{condition.condition_id}: unknown policy {condition.policy!r}.")
    if (condition.policy == FULL) != (condition.token_budget is None):
        raise ValueError(f"{condition.condition_id}: only 'full' may omit a budget.")
    if (condition.policy == RANDOM) != (condition.seed is not None):
        raise ValueError(f"{condition.condition_id}: only 'random' takes a seed.")
    config = get_model_config(condition.model_key)
    if condition.token_budget is not None and condition.token_budget > config.max_tokens:
        raise ValueError(f"{condition.condition_id}: budget exceeds the model limit.")
    if config.query_prefix or config.doc_prefix:
        raise ValueError(f"{condition.condition_id}: model prefixes would escape the budget.")


def _validate_registry(conditions: List[AblationCondition]) -> None:
    """Raise if identifiers repeat, models differ, or any condition is invalid."""
    counts = Counter(c.condition_id for c in conditions)
    duplicates = sorted(cid for cid, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate ablation condition identifiers: {duplicates}.")
    if len({c.model_key for c in conditions}) != 1:
        raise ValueError("Every ablation condition must share one embedding model.")
    for condition in conditions:
        _validate_condition(condition)


_validate_registry(ABLATION_CONDITIONS)

_CONDITIONS_BY_ID = {c.condition_id: c for c in ABLATION_CONDITIONS}


def get_ablation_condition(condition_id: str) -> AblationCondition:
    """
    Return the ablation condition registered under ``condition_id``.

    Parameters
    ----------
    condition_id : str
        Identifier, e.g. ``"PA-ppas-420"``.

    Returns
    -------
    AblationCondition
        The matching condition.

    Raises
    ------
    KeyError
        If ``condition_id`` is not registered.
    """
    try:
        return _CONDITIONS_BY_ID[condition_id]
    except KeyError:
        raise KeyError(f"Unknown ablation condition: {condition_id!r}.") from None


def stress_budget(lengths: List[int]) -> int:
    """
    Apply the outcome-blind stress-budget rule to full-representation lengths.

    Parameters
    ----------
    lengths : List[int]
        Encoded lengths of every source and candidate entity's full text.

    Returns
    -------
    int
        The median length floored to a multiple of
        :data:`STRESS_BUDGET_MULTIPLE`.
    """
    ordered = sorted(lengths)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return int(median // STRESS_BUDGET_MULTIPLE * STRESS_BUDGET_MULTIPLE)
