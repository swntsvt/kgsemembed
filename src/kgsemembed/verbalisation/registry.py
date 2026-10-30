"""Central factory for constructing configured verbaliser instances.

Provides a single authoritative mapping between experimental strategy
identifiers and their verbaliser implementations, covering both individual
strategies (V1-V8) and their supported compositions.  Strategy-selection logic
lives here alone so experiments can request a verbaliser declaratively by name
rather than embedding construction logic across the codebase.

Budgeted strategies (#183) are built for one encoder role, since the budget
counts the encoder's input (M5's source prefix included), and are wrapped in
:class:`~kgsemembed.verbalisation.budget.BudgetedVerbaliser` so the whole text
holds the budget.
"""

from typing import Callable

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.budget import BudgetedVerbaliser, strategy_is_budgeted, token_budget
from kgsemembed.verbalisation.combined import CombinedVerbaliser
from kgsemembed.verbalisation.v1 import LabelVerbaliser
from kgsemembed.verbalisation.v2 import AnnotationVerbaliser
from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser
from kgsemembed.verbalisation.v4 import StructuredKVVerbaliser
from kgsemembed.verbalisation.v5 import NeighbourhoodWalkVerbaliser
from kgsemembed.verbalisation.v6 import SchemaAwareVerbaliser
from kgsemembed.verbalisation.v7 import HierarchicalContextVerbaliser
from kgsemembed.verbalisation.v8 import RelationalSignatureVerbaliser

_VerbaliserFactory = Callable[[str, str], VerbaliserBase]

_REGISTRY: dict[str, _VerbaliserFactory] = {
    "V1": lambda _model_key, _role: LabelVerbaliser(),
    "V2": lambda _model_key, _role: AnnotationVerbaliser(),
    "V3": lambda model_key, role: TemplateNLVerbaliser(model_key, role),
    "V4": lambda model_key, role: StructuredKVVerbaliser(model_key, role),
    "V5": lambda model_key, role: NeighbourhoodWalkVerbaliser(model_key, role),
    "V6": lambda model_key, role: SchemaAwareVerbaliser(model_key, role),
    "V7": lambda model_key, role: HierarchicalContextVerbaliser(model_key, role),
    "V8": lambda _model_key, _role: RelationalSignatureVerbaliser(),
    "V2+V6": lambda model_key, role: CombinedVerbaliser(
        [AnnotationVerbaliser(), SchemaAwareVerbaliser(model_key, role)]
    ),
    "V2+V8": lambda _model_key, _role: CombinedVerbaliser(
        [AnnotationVerbaliser(), RelationalSignatureVerbaliser()]
    ),
    "V2+V7": lambda model_key, role: CombinedVerbaliser(
        [AnnotationVerbaliser(), HierarchicalContextVerbaliser(model_key, role)]
    ),
    "V2+V8+V7": lambda model_key, role: CombinedVerbaliser(
        [
            AnnotationVerbaliser(),
            RelationalSignatureVerbaliser(),
            HierarchicalContextVerbaliser(model_key, role),
        ]
    ),
    "V6+V3": lambda model_key, role: CombinedVerbaliser(
        [SchemaAwareVerbaliser(model_key, role), TemplateNLVerbaliser(model_key, role)]
    ),
    "V4+V6": lambda model_key, role: CombinedVerbaliser(
        [StructuredKVVerbaliser(model_key, role), SchemaAwareVerbaliser(model_key, role)]
    ),
    "V8+V6": lambda model_key, role: CombinedVerbaliser(
        [RelationalSignatureVerbaliser(), SchemaAwareVerbaliser(model_key, role)]
    ),
}

VALID_STRATEGY_NAMES: list[str] = list(_REGISTRY)


def build_verbaliser(
    strategy_name: str, model_key: str = "M2", role: str = "candidate"
) -> VerbaliserBase:
    """
    Construct the verbaliser registered under *strategy_name*.

    Parameters
    ----------
    strategy_name : str
        Supported strategy identifier; see :data:`VALID_STRATEGY_NAMES`.
    model_key : str
        Model key forwarded to model-aware strategies (e.g. "M1", "M2").
        Strategies that do not require a model key ignore it.
    role : str
        ``"source"`` or ``"candidate"``: the encoder role whose input a
        budgeted strategy's budget counts (default ``"candidate"``).

    Returns
    -------
    VerbaliserBase
        A freshly constructed verbaliser instance; a budgeted strategy comes
        wrapped in :class:`BudgetedVerbaliser`.

    Raises
    ------
    ValueError
        If *strategy_name* is not a supported strategy.
    """
    try:
        factory = _REGISTRY[strategy_name]
    except KeyError:
        supported = ", ".join(VALID_STRATEGY_NAMES)
        raise ValueError(
            f"Unknown verbalisation strategy: {strategy_name!r}. "
            f"Supported strategies are: {supported}."
        ) from None
    verbaliser = factory(model_key, role)
    if strategy_is_budgeted(strategy_name, model_key):
        return BudgetedVerbaliser(verbaliser, token_budget(model_key, role))
    return verbaliser
