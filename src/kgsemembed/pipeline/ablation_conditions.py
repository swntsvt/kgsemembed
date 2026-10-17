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

Validation (#128 follow-up)
---------------------------
The base-vocabulary policies P1-P4 are also registered on D1 at
:data:`VALIDATION_BUDGETS` (32 and 64) and on D2 at 48, and P5-P7 on D2 with
the D2 vocabulary ``obo_extended`` (:data:`EXTENDED_VOCABULARIES`).  A label
names the same filter/vocabulary-role/order design everywhere; it is unique
per (dataset, budget), the unit a report groups by.

Predicate-role ablation (#131)
------------------------------
The ``PR-*`` conditions (:data:`ROLE_ARMS`, :data:`ROLE_MATRIX`) remove items by
predicate role (``roles_v1``) with every predicate otherwise eligible, with no
budget (``full``: every remaining item, only the model's 512-token input limit)
or at a budget filled in ``core_first`` order.  They are identified by
``role_arm`` and never carry a ``policy_label``, so the #128 family and its
reports are unaffected.
"""


from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from kgsemembed.embeddings.models import get_model_config
from kgsemembed.verbalisation.predicate_config import PER_PREDICATE_CONFIGS
from kgsemembed.verbalisation.predicate_roles import (
    CORE,
    DEFINITION,
    DISJOINTNESS,
    IDENTIFIER,
    NON_CORE_ROLES,
    NON_ENGLISH,
    PREFERRED_NAME,
    ROLE_MAPS,
    ROLES,
    SYNONYM,
)
from kgsemembed.verbalisation.selection import (
    CORE_FIRST,
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
    role_arm : str
        Predicate-role arm (#131), e.g. ``"R-syn"``; empty otherwise.
    excluded_roles : Tuple[str, ...]
        Roles the arm removes, sorted.
    role_map : str
        Role-map name; empty unless ``role_arm`` is set.
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
    role_arm: str = ""
    excluded_roles: Tuple[str, ...] = ()
    role_map: str = ""

    def __post_init__(self) -> None:
        if not self.order_mode:
            object.__setattr__(self, "order_mode", self.policy)

    @property
    def is_controlled(self) -> bool:
        """Whether the condition belongs to the #128 controlled-policy family."""
        return bool(self.policy_label)

    @property
    def is_role_ablation(self) -> bool:
        """Whether the condition belongs to the #131 predicate-role family."""
        return bool(self.role_arm)


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
VALIDATION_BUDGETS: Tuple[int, ...] = (32, 64)
VALIDATION_DATASET = "D2"
EXTENDED_VOCABULARIES = {"D1": "fma_extended", "D2": "obo_extended"}

_FILTER_LABELS = {FILTER_ALL: "all", FILTER_COVERED: "cov"}
_VOCABULARY_LABELS = {"base": "base", "fma_extended": "fmax", "obo_extended": "obox"}

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

BASE_POLICIES = tuple(p for p in CONTROLLED_POLICIES if p[2] == "base")
"""P1-P4: the base-vocabulary policies, run at every validation budget."""


def extended_policies(vocabulary: str) -> Tuple[Tuple[str, str, str, str], ...]:
    """
    Return P5-P7 with their extended vocabulary replaced by ``vocabulary``.

    Parameters
    ----------
    vocabulary : str
        A dataset's extended vocabulary, e.g. ``"obo_extended"``.

    Returns
    -------
    Tuple[Tuple[str, str, str, str], ...]
        ``(label, filter mode, vocabulary, order mode)`` of P5-P7.
    """
    return tuple((label, f, vocabulary, o) for label, f, v, o in CONTROLLED_POLICIES if v != "base")


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


def _controlled_family(
    budget: int, datasets: Tuple[str, ...], policies: Tuple[Tuple[str, str, str, str], ...]
) -> List[AblationCondition]:
    """Return the given controlled policies at one budget."""
    return [
        AblationCondition(
            controlled_condition_id(filter_mode, vocabulary, order_mode, budget),
            order_mode, budget, None, datasets,
            filter_mode=filter_mode, order_mode=order_mode,
            predicate_config=vocabulary, policy_label=label,
        )
        for label, filter_mode, vocabulary, order_mode in policies
    ]


ROLE_MAP = "roles_v1"
ROLE_BUDGETED_ORDER = CORE_FIRST
_REMOVED_ARMS = {
    "R-pref": PREFERRED_NAME, "R-syn": SYNONYM, "R-def": DEFINITION, "R-id": IDENTIFIER,
    "R-nonEn": NON_ENGLISH, "R-disj": DISJOINTNESS,
}
_ADDED_ARMS = {"C+pref": PREFERRED_NAME, "C+syn": SYNONYM, "C+def": DEFINITION, "C+id": IDENTIFIER}
ROLE_ARMS: dict[str, Tuple[str, ...]] = {
    "A0": (),
    **{arm: (role,) for arm, role in _REMOVED_ARMS.items()},
    "A-core": tuple(sorted(NON_CORE_ROLES)),
    **{arm: tuple(sorted(set(NON_CORE_ROLES) - {role})) for arm, role in _ADDED_ARMS.items()},
}
"""Arm to the roles it removes: A0 none, R-x one role, A-core every non-core
role, C+x every non-core role except x (core plus one role)."""

_D1_BOTH = ("A0", "R-pref", "R-syn", "R-def", "R-id", "R-nonEn", "R-disj", "A-core")
_D2_BOTH = ("A0", "R-syn", "R-def", "R-id", "R-disj", "A-core")
ROLE_MATRIX: dict[str, dict[Optional[int], Tuple[str, ...]]] = {
    "D1": {None: (*_D1_BOTH, "C+pref", "C+syn", "C+def", "C+id"), 48: _D1_BOTH},
    "D2": {None: (*_D2_BOTH, "C+syn", "C+def"), 48: _D2_BOTH, 24: ("A0", "R-syn", "R-def", "A-core")},
}
"""Dataset to budget (``None`` = no budget) to the arms run there: 38 runs."""

_ARM_IDS = {"A0": "all", "A-core": "core"}


def role_condition_id(arm: str, budget: Optional[int]) -> str:
    """
    Name a predicate-role condition by its arm and budget.

    Parameters
    ----------
    arm : str
        A key of :data:`ROLE_ARMS`.
    budget : Optional[int]
        Token budget; ``None`` for no budget.

    Returns
    -------
    str
        E.g. ``"PR-no-syn-full"``, ``"PR-core-48"``, ``"PR-core-plus-def-full"``.
    """
    name = _ARM_IDS.get(arm) or arm.replace("R-", "no-").replace("C+", "core-plus-")
    return f"PR-{name.lower()}-{'full' if budget is None else budget}"


def _role_family() -> List[AblationCondition]:
    """Return one condition per (arm, budget), listing every dataset that runs it."""
    cells = sorted({(arm, budget) for plan in ROLE_MATRIX.values() for budget, arms in plan.items() for arm in arms},
                   key=lambda cell: (cell[1] is not None, cell[1] or 0, list(ROLE_ARMS).index(cell[0])))
    return [
        AblationCondition(
            role_condition_id(arm, budget), FULL if budget is None else ROLE_BUDGETED_ORDER, budget, None,
            tuple(d for d, plan in ROLE_MATRIX.items() if arm in plan.get(budget, ())),
            role_arm=arm, excluded_roles=ROLE_ARMS[arm], role_map=ROLE_MAP,
        )
        for arm, budget in cells
    ]


ABLATION_CONDITIONS: List[AblationCondition] = [
    AblationCondition("PA-full", FULL, None, None, ("D5", "D1")),
    *_budget_family(PRIMARY_BUDGET, ("D5",)),
    *_budget_family(STRESS_BUDGETS["D5"], ("D5",)),
    *_budget_family(STRESS_BUDGETS["D1"], ("D1",)),
    *_controlled_family(CONTROLLED_BUDGET, (*CONTROLLED_DATASETS, VALIDATION_DATASET), BASE_POLICIES),
    *_controlled_family(CONTROLLED_BUDGET, CONTROLLED_DATASETS, extended_policies("fma_extended")),
    *_controlled_family(
        CONTROLLED_BUDGET, (VALIDATION_DATASET,), extended_policies(EXTENDED_VOCABULARIES[VALIDATION_DATASET])
    ),
    *(c for budget in VALIDATION_BUDGETS
      for c in _controlled_family(budget, CONTROLLED_DATASETS, BASE_POLICIES)),
    *_role_family(),
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
    if condition.is_role_ablation:
        _validate_role_condition(condition)
    config = get_model_config(condition.model_key)
    if condition.token_budget is not None and condition.token_budget > config.max_tokens:
        raise ValueError(f"{condition.condition_id}: budget exceeds the model limit.")
    if config.query_prefix or config.doc_prefix:
        raise ValueError(f"{condition.condition_id}: model prefixes would escape the budget.")


def _validate_role_condition(condition: AblationCondition) -> None:
    """Raise if a predicate-role condition's map, roles, order or label is inconsistent."""
    cid = condition.condition_id
    if condition.role_map not in ROLE_MAPS:
        raise ValueError(f"{cid}: unknown role map {condition.role_map!r}.")
    if not set(condition.excluded_roles) <= set(ROLES) - {CORE}:
        raise ValueError(f"{cid}: excluded roles must be non-core roles.")
    if condition.policy not in (FULL, ROLE_BUDGETED_ORDER) or condition.policy_label:
        raise ValueError(f"{cid}: role arms use 'full' or '{ROLE_BUDGETED_ORDER}' and no policy label.")
    if (condition.filter_mode, condition.predicate_config) != (FILTER_ALL, "base"):
        raise ValueError(f"{cid}: role arms keep every predicate eligible before role removal.")


def _validate_registry(conditions: List[AblationCondition]) -> None:
    """Raise if identifiers repeat, models differ, or any condition is invalid."""
    counts = Counter(c.condition_id for c in conditions)
    duplicates = sorted(cid for cid, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate ablation condition identifiers: {duplicates}.")
    labels = Counter(
        (c.policy_label, dataset, c.token_budget)
        for c in conditions if c.is_controlled for dataset in c.datasets
    )
    duplicated = sorted(key for key, count in labels.items() if count > 1)
    if duplicated:
        raise ValueError(f"Duplicate controlled policy labels per (dataset, budget): {duplicated}.")
    arms = Counter((c.role_arm, d, c.token_budget) for c in conditions if c.is_role_ablation for d in c.datasets)
    repeated = sorted(key for key, count in arms.items() if count > 1)
    if repeated:
        raise ValueError(f"Duplicate role arms per (dataset, budget): {repeated}.")
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
