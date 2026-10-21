"""Registry of the text-serialization-order experiment (#134).

The experiment asks how much the order in which selected items are written
changes matching performance when the selected item set is held fixed.  Its
conditions are kept out of :data:`~kgsemembed.pipeline.ablation_conditions.ABLATION_CONDITIONS`,
so every earlier registry, runner default and report is unchanged.

Logical conditions and executions
---------------------------------
A *logical condition* is a (dataset, representation, budget, selection order,
render order) combination the analysis refers to.  An *execution* is one run
of a registered condition on one dataset.  Two logical conditions share an
execution when they are identical by construction:

* Block S (no budget): 8 dataset-representation rows x 4 render orders = 32
  logical conditions.  In A-core only core items remain, and canonical order
  already lists them ``type``, ``label``, ``subClassOf``, so ``core`` equals
  ``uri`` (2 shared): 30 executions.
* Block B (A0 at a budget): per budget configuration, 4 fixed-set conditions
  (``core`` selection, each render order), 4 coupled ones (selection order =
  render order) and 2 selection-only ones (``label``/``role`` selection,
  ``uri`` render) = 10 logical conditions.  Coupled ``core`` *is* fixed
  ``core`` (1 shared): 9 executions.  Three configurations (D1@48, D2@24,
  D2@48) give 30 logical conditions and 27 executions.

In total 62 logical conditions and 57 executions; :func:`check_registry`
asserts both counts at import time.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kgsemembed.pipeline.ablation_conditions import ABLATION_MODEL_KEY, ROLE_ARMS, ROLE_MAP, AblationCondition
from kgsemembed.verbalisation.selection import CORE_FIRST, DETERMINISTIC, FULL, LABEL_FIRST, ROLE_PRIORITY
from kgsemembed.verbalisation.serialization import (
    RENDER_CORE,
    RENDER_LABEL,
    RENDER_ROLE,
    RENDER_URI,
    SERIALIZATION_ORDERS,
)

BLOCK_S = "S"
BLOCK_B = "B"
S_REPRESENTATIONS: Dict[str, Tuple[str, ...]] = {
    "D1": ("A0", "A-core", "C+syn", "C+id"),
    "D2": ("A0", "A-core", "C+syn", "C+def"),
}
B_BUDGETS: Tuple[Tuple[str, int], ...] = (("D1", 48), ("D2", 24), ("D2", 48))
SELECTION_MODE = {RENDER_URI: DETERMINISTIC, RENDER_CORE: CORE_FIRST, RENDER_LABEL: LABEL_FIRST,
                  RENDER_ROLE: ROLE_PRIORITY}
"""Selection order name to the selector's order mode; ``uri`` selection is
lexicographic, i.e. #128's deterministic order."""

REFERENCE_SELECTION = RENDER_CORE
B_LOGICAL: Tuple[Tuple[str, str, str], ...] = (
    ("F-uri", RENDER_CORE, RENDER_URI),
    ("F-core", RENDER_CORE, RENDER_CORE),
    ("F-label", RENDER_CORE, RENDER_LABEL),
    ("F-role", RENDER_CORE, RENDER_ROLE),
    ("C-uri", RENDER_URI, RENDER_URI),
    ("C-core", RENDER_CORE, RENDER_CORE),
    ("C-label", RENDER_LABEL, RENDER_LABEL),
    ("C-role", RENDER_ROLE, RENDER_ROLE),
    ("X-label", RENDER_LABEL, RENDER_URI),
    ("X-role", RENDER_ROLE, RENDER_URI),
)
"""Block B per budget: (name, selection order, render order)."""

EXECUTION_LABELS = {
    (RENDER_CORE, RENDER_URI): "B1", (RENDER_CORE, RENDER_CORE): "B2", (RENDER_CORE, RENDER_LABEL): "B3",
    (RENDER_CORE, RENDER_ROLE): "B4", (RENDER_URI, RENDER_URI): "B5", (RENDER_LABEL, RENDER_LABEL): "B6",
    (RENDER_ROLE, RENDER_ROLE): "B7", (RENDER_LABEL, RENDER_URI): "B8", (RENDER_ROLE, RENDER_URI): "B9",
}
"""(selection, render) to the execution label B1-B9 used in #134."""

EXPECTED_LOGICAL = 62
EXPECTED_EXECUTIONS = 57
_ARM_NAMES = {"A0": "all", "A-core": "core"}


@dataclass(frozen=True)
class LogicalCondition:
    """
    One condition the analysis refers to, and the execution that produces it.

    Attributes
    ----------
    name : str
        Unique name, e.g. ``"D1/S/C+syn/label"`` or ``"D2/B24/C-role"``.
    dataset : str
        ``"D1"`` or ``"D2"``.
    block : str
        ``"S"`` (no budget) or ``"B"`` (budgeted).
    representation : str
        Role arm from #131, e.g. ``"A0"``.
    budget : Optional[int]
        Token budget; ``None`` in Block S.
    selection : str
        Selection order; ``"all"`` in Block S, where every item is kept.
    render : str
        Render order.
    condition_id : str
        The registered condition that executes it.
    shared_reason : str
        Why it shares an execution with another logical condition; empty if it does not.
    """

    name: str
    dataset: str
    block: str
    representation: str
    budget: Optional[int]
    selection: str
    render: str
    condition_id: str
    shared_reason: str = ""


def so_condition_id(representation: str, budget: Optional[int], selection: str, render: str) -> str:
    """
    Name a #134 condition by its representation, budget and orders.

    Parameters
    ----------
    representation : str
        Role arm, e.g. ``"C+syn"``.
    budget : Optional[int]
        Token budget; ``None`` for Block S.
    selection : str
        Selection order (ignored without a budget, where every item is kept).
    render : str
        Render order.

    Returns
    -------
    str
        E.g. ``"SO-core-plus-syn-label-full"`` or ``"SO-all-sel_core-ren_uri-48"``.
    """
    arm = (_ARM_NAMES.get(representation) or representation.replace("C+", "core-plus-")).lower()
    if budget is None:
        return f"SO-{arm}-{render}-full"
    return f"SO-{arm}-sel_{selection}-ren_{render}-{budget}"


def _block_s(dataset: str, representation: str) -> List[LogicalCondition]:
    """Return the four render orders of one Block S row."""
    rows = []
    for render in SERIALIZATION_ORDERS:
        shared = representation == "A-core" and render == RENDER_CORE
        executed = RENDER_URI if shared else render
        rows.append(LogicalCondition(
            f"{dataset}/S/{representation}/{render}", dataset, BLOCK_S, representation, None, "all", render,
            so_condition_id(representation, None, "all", executed),
            "A-core keeps only core items, whose canonical order is already core-first" if shared else "",
        ))
    return rows


def _block_b(dataset: str, budget: int) -> List[LogicalCondition]:
    """Return the ten logical conditions of one budget configuration."""
    return [
        LogicalCondition(
            f"{dataset}/B{budget}/{name}", dataset, BLOCK_B, "A0", budget, selection, render,
            so_condition_id("A0", budget, selection, render),
            "coupled core-first is the fixed core-first selection rendered core-first" if name == "C-core" else "",
        )
        for name, selection, render in B_LOGICAL
    ]


LOGICAL_CONDITIONS: List[LogicalCondition] = [
    *(c for dataset, reps in S_REPRESENTATIONS.items() for rep in reps for c in _block_s(dataset, rep)),
    *(c for dataset, budget in B_BUDGETS for c in _block_b(dataset, budget)),
]


def executions() -> List[Tuple[str, str]]:
    """
    Return every unique execution as ``(condition_id, dataset)``, in matrix order.

    Returns
    -------
    List[Tuple[str, str]]
        One entry per run.
    """
    return list(dict.fromkeys((c.condition_id, c.dataset) for c in LOGICAL_CONDITIONS))


def _condition_for(condition_id: str) -> AblationCondition:
    """Build the registered condition behind an execution identifier."""
    first = next(c for c in LOGICAL_CONDITIONS if c.condition_id == condition_id and not c.shared_reason)
    datasets = tuple(dict.fromkeys(d for cid, d in executions() if cid == condition_id))
    policy = FULL if first.budget is None else SELECTION_MODE[first.selection]
    return AblationCondition(
        condition_id, policy, first.budget, None, datasets,
        model_key=ABLATION_MODEL_KEY, role_arm=first.representation,
        excluded_roles=ROLE_ARMS[first.representation], role_map=ROLE_MAP, render_order=first.render,
    )


SERIALIZATION_CONDITIONS: List[AblationCondition] = [
    _condition_for(cid) for cid in dict.fromkeys(c.condition_id for c in LOGICAL_CONDITIONS)
]

FIXED_SET_GROUPS: List[Tuple[str, ...]] = [
    *(tuple(f"{d}/S/{rep}/{r}" for r in SERIALIZATION_ORDERS) for d, reps in S_REPRESENTATIONS.items() for rep in reps),
    *(tuple(f"{d}/B{b}/{n}" for n in ("F-uri", "F-core", "F-label", "F-role", "C-core")) for d, b in B_BUDGETS),
    *(tuple(f"{d}/B{b}/{n}" for n in ("C-label", "X-label")) for d, b in B_BUDGETS),
    *(tuple(f"{d}/B{b}/{n}" for n in ("C-role", "X-role")) for d, b in B_BUDGETS),
]
"""Logical conditions whose selected item sets must be identical, entity by entity."""


def _role_reference(representation: str) -> str:
    """Return the #131 no-budget condition of a representation."""
    name = (_ARM_NAMES.get(representation) or representation.replace("C+", "core-plus-")).lower()
    return f"PR-{name}-full"


ROLE_RESULTS_DIR = "data/results_predicate_roles"
P1_RESULTS_DIRS = {"D1": "data/results_ppas_controlled", "D2": "data/results_ppas_validation"}


def reproduction_target(logical: LogicalCondition) -> Optional[Tuple[str, str]]:
    """
    Return the stored run a canonically rendered condition must reproduce.

    Parameters
    ----------
    logical : LogicalCondition
        A logical condition.

    Returns
    -------
    Optional[Tuple[str, str]]
        ``(results_dir, condition_id)`` of the #131 or #128 run with identical
        texts, or ``None`` when there is none (non-canonical render, or coupled
        ``uri`` at D2@24, which #128 never ran).
    """
    if logical.render != RENDER_URI or logical.shared_reason:
        return None
    if logical.block == BLOCK_S:
        return ROLE_RESULTS_DIR, _role_reference(logical.representation)
    if logical.selection == REFERENCE_SELECTION:
        return ROLE_RESULTS_DIR, f"PR-all-{logical.budget}"
    if logical.selection == RENDER_URI and logical.budget == 48:
        return P1_RESULTS_DIRS[logical.dataset], "PA-all-base-det-48"
    return None


def get_logical(name: str) -> LogicalCondition:
    """
    Return a logical condition by name.

    Parameters
    ----------
    name : str
        E.g. ``"D2/S/A0/label"``.

    Returns
    -------
    LogicalCondition
        The matching logical condition.
    """
    matches = [c for c in LOGICAL_CONDITIONS if c.name == name]
    if not matches:
        raise KeyError(f"Unknown logical condition {name!r}.")
    return matches[0]


def get_serialization_condition(condition_id: str) -> AblationCondition:
    """
    Return a registered #134 condition by identifier.

    Parameters
    ----------
    condition_id : str
        E.g. ``"SO-all-label-full"``.

    Returns
    -------
    AblationCondition
        The matching condition.
    """
    matches = [c for c in SERIALIZATION_CONDITIONS if c.condition_id == condition_id]
    if not matches:
        raise KeyError(f"Unknown serialization condition {condition_id!r}.")
    return matches[0]


def check_registry() -> None:
    """Raise if the matrix counts, sharing or names deviate from #134."""
    names = Counter(c.name for c in LOGICAL_CONDITIONS)
    if max(names.values()) > 1:
        raise ValueError(f"Duplicate logical conditions: {sorted(n for n, k in names.items() if k > 1)}.")
    if len(LOGICAL_CONDITIONS) != EXPECTED_LOGICAL or len(executions()) != EXPECTED_EXECUTIONS:
        raise ValueError(f"Expected {EXPECTED_LOGICAL} logical conditions and {EXPECTED_EXECUTIONS} executions, "
                         f"got {len(LOGICAL_CONDITIONS)} and {len(executions())}.")
    shared = [c for c in LOGICAL_CONDITIONS if c.shared_reason]
    if len(shared) != EXPECTED_LOGICAL - EXPECTED_EXECUTIONS:
        raise ValueError(f"Expected {EXPECTED_LOGICAL - EXPECTED_EXECUTIONS} shared conditions, got {len(shared)}.")
    known = {c.name for c in LOGICAL_CONDITIONS}
    if any(name not in known for group in FIXED_SET_GROUPS for name in group):
        raise ValueError("A fixed-set group names an unknown logical condition.")


check_registry()
