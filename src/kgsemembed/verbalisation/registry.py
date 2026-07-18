"""Central factory for constructing configured verbaliser instances.

Provides a single authoritative mapping between experimental strategy
identifiers and their verbaliser implementations, covering both individual
strategies (V1-V8) and their supported compositions.  Strategy-selection logic
lives here alone so experiments can request a verbaliser declaratively by name
rather than embedding construction logic across the codebase.
"""

from typing import Callable

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.combined import CombinedVerbaliser
from kgsemembed.verbalisation.v1 import LabelVerbaliser
from kgsemembed.verbalisation.v2 import AnnotationVerbaliser
from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser
from kgsemembed.verbalisation.v4 import StructuredKVVerbaliser
from kgsemembed.verbalisation.v5 import NeighbourhoodWalkVerbaliser
from kgsemembed.verbalisation.v6 import SchemaAwareVerbaliser
from kgsemembed.verbalisation.v7 import HierarchicalContextVerbaliser
from kgsemembed.verbalisation.v8 import RelationalSignatureVerbaliser

_VerbaliserFactory = Callable[[str], VerbaliserBase]

_REGISTRY: dict[str, _VerbaliserFactory] = {
    "V1": lambda _model_key: LabelVerbaliser(),
    "V2": lambda _model_key: AnnotationVerbaliser(),
    "V3": lambda model_key: TemplateNLVerbaliser(model_key),
    "V4": lambda model_key: StructuredKVVerbaliser(model_key),
    "V5": lambda model_key: NeighbourhoodWalkVerbaliser(model_key),
    "V6": lambda model_key: SchemaAwareVerbaliser(model_key),
    "V7": lambda model_key: HierarchicalContextVerbaliser(model_key),
    "V8": lambda _model_key: RelationalSignatureVerbaliser(),
    "V2+V6": lambda model_key: CombinedVerbaliser(
        [AnnotationVerbaliser(), SchemaAwareVerbaliser(model_key)]
    ),
    "V2+V8": lambda _model_key: CombinedVerbaliser(
        [AnnotationVerbaliser(), RelationalSignatureVerbaliser()]
    ),
    "V2+V7": lambda model_key: CombinedVerbaliser(
        [AnnotationVerbaliser(), HierarchicalContextVerbaliser(model_key)]
    ),
    "V2+V8+V7": lambda model_key: CombinedVerbaliser(
        [
            AnnotationVerbaliser(),
            RelationalSignatureVerbaliser(),
            HierarchicalContextVerbaliser(model_key),
        ]
    ),
    "V6+V3": lambda model_key: CombinedVerbaliser(
        [SchemaAwareVerbaliser(model_key), TemplateNLVerbaliser(model_key)]
    ),
    "V4+V6": lambda model_key: CombinedVerbaliser(
        [StructuredKVVerbaliser(model_key), SchemaAwareVerbaliser(model_key)]
    ),
    "V8+V6": lambda model_key: CombinedVerbaliser(
        [RelationalSignatureVerbaliser(), SchemaAwareVerbaliser(model_key)]
    ),
}

VALID_STRATEGY_NAMES: list[str] = list(_REGISTRY)


def build_verbaliser(
    strategy_name: str, model_key: str = "M2"
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

    Returns
    -------
    VerbaliserBase
        A freshly constructed verbaliser instance.

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
    return factory(model_key)
