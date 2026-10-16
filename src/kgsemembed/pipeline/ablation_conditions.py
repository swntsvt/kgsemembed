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

Controlled policies (#128)
--------------------------
P1-P7 vary predicate eligibility (``filter_mode``), predicate vocabulary and
visiting order one at a time, on D1 at 48 tokens; see
:data:`CONTROLLED_POLICIES`.  They are identified by ``policy_label`` and their
condition IDs name every dimension, so no two collide when grouped by policy.
"""


from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from kgsemembed.embeddings.models import get_model_config
from kgsemembed.verbalisation.predicate_config import PER_PREDICATE_CONFIGS
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC,
    FILTER_ALL,
    FILTER_COVERED,
    FILTER_MODES,
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
    filter_mode : str
        Predicate eligibility: ``"all"`` or ``"covered"``.
    order_mode : str
        Visiting order; defaults to ``policy``, which it must equal.
    predicate_config : str
        Predicate vocabulary name, ``"base"`` or ``"fma_extended"``.
    policy_label : str
        Controlled-policy label (``"P1"``-``"P7"``); empty for #122 conditions.
    """

    condition_id: str
    policy: str
    token_budget: Optional[int]
    seed: Optional[int]
    datasets: Tuple[str, ...]
    model_key: str = ABLATION_MODEL_KEY
    filter_mode: str = FILTER_ALL
    order_mode: str = ""
    predicate_config: str = "base"
    policy_label: str = ""

    def __post_init__(self) -> None:
        if not self.order_mode:
            object.__setattr__(self, "order_mode", self.policy)

    @property
    def is_controlled(self) -> bool:
        """Whether the condition belongs to the #128 controlled-policy family."""
        return bool(self.policy_label)


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


CONTROLLED_BUDGET = STRESS_BUDGETS["D1"]
CONTROLLED_DATASETS: Tuple[str, ...] = ("D1",)

_FILTER_LABELS = {FILTER_ALL: "all", FILTER_COVERED: "cov"}
_VOCABULARY_LABELS = {"base": "base", "fma_extended": "fmax"}

CONTROLLED_POLICIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("P1", FILTER_ALL, "base", DETERMINISTIC),
    ("P2", FILTER_ALL, "base", PPAS),
    ("P3", FILTER_COVERED, "base", DETERMINISTIC),
    ("P4", FILTER_COVERED, "base", PPAS),
    ("P5", FILTER_ALL, "fma_extended", PPAS),
    ("P6", FILTER_COVERED, "fma_extended", DETERMINISTIC),
    ("P7", FILTER_COVERED, "fma_extended", PPAS),
)
"""Controlled policies (#128): (label, filter mode, vocabulary, order mode).

P1 and P2 are the #122 deterministic and PPAS policies re-registered with an
explicit vocabulary, so they are rerun from the same commit as P3-P7.  The
filter "all" with deterministic order never consults a vocabulary, so an
``all``/``fma_extended``/deterministic condition would equal P1 and is not
registered; P6 and P7 are the extended-vocabulary controls instead.
"""


def controlled_condition_id(filter_mode: str, vocabulary: str, order_mode: str, budget: int) -> str:
    """
    Name a controlled condition by every dimension it fixes.

    Parameters
    ----------
    filter_mode : str
        ``"all"`` or ``"covered"``.
    vocabulary : str
        Predicate vocabulary name.
    order_mode : str
        ``"ppas"`` or ``"deterministic"``.
    budget : int
        Token budget.

    Returns
    -------
    str
        E.g. ``"PA-cov-fmax-ppas-48"``.
    """
    order = _POLICY_LABELS[order_mode]
    return f"PA-{_FILTER_LABELS[filter_mode]}-{_VOCABULARY_LABELS[vocabulary]}-{order}-{budget}"


def _controlled_family(budget: int, datasets: Tuple[str, ...]) -> List[AblationCondition]:
    """Return the P1-P7 controlled conditions at one budget."""
    return [
        AblationCondition(
            controlled_condition_id(filter_mode, vocabulary, order_mode, budget),
            order_mode, budget, None, datasets,
            filter_mode=filter_mode, order_mode=order_mode,
            predicate_config=vocabulary, policy_label=label,
        )
        for label, filter_mode, vocabulary, order_mode in CONTROLLED_POLICIES
    ]


ABLATION_CONDITIONS: List[AblationCondition] = [
    AblationCondition("PA-full", FULL, None, None, ("D5", "D1")),
    *_budget_family(PRIMARY_BUDGET, ("D5",)),
    *_budget_family(STRESS_BUDGETS["D5"], ("D5",)),
    *_budget_family(STRESS_BUDGETS["D1"], ("D1",)),
    *_controlled_family(CONTROLLED_BUDGET, CONTROLLED_DATASETS),
]


def _validate_condition(condition: AblationCondition) -> None:
    """Raise if a condition's policy, budget, seed or model is inconsistent."""
    if condition.policy not in SELECTION_POLICIES:
        raise ValueError(f"{condition.condition_id}: unknown policy {condition.policy!r}.")
    if (condition.policy == FULL) != (condition.token_budget is None):
        raise ValueError(f"{condition.condition_id}: only 'full' may omit a budget.")
    if (condition.policy == RANDOM) != (condition.seed is not None):
        raise ValueError(f"{condition.condition_id}: only 'random' takes a seed.")
    if condition.order_mode != condition.policy:
        raise ValueError(f"{condition.condition_id}: order_mode must equal policy.")
    if condition.filter_mode not in FILTER_MODES:
        raise ValueError(f"{condition.condition_id}: unknown filter mode {condition.filter_mode!r}.")
    if condition.predicate_config not in PER_PREDICATE_CONFIGS:
        raise ValueError(f"{condition.condition_id}: unknown vocabulary {condition.predicate_config!r}.")
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
    labels = Counter(c.policy_label for c in conditions if c.is_controlled)
    if any(count > 1 for count in labels.values()):
        raise ValueError(f"Duplicate controlled policy labels: {sorted(labels)}.")
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
