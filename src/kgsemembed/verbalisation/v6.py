"""Schema-aware verbalisation strategy (V6).

Serialises entities as deterministic pipe-separated predicate-value pairs,
always curating triples in schema-aware predicate-tier priority order.

Unlike V4, tier ordering is applied unconditionally -- it is never gated on
triple count -- and the model token budget (when one exists) is applied only
after the tier ordering has been established: rendered pairs are kept in tier
order while they fit, counted with the model's tokenizer (#183).  Models
without a budget (e.g. ``"M3"``) retain every tier-ordered triple.
"""

from functools import cached_property
from typing import Optional

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.budget import TokenBudget, fit_items, token_budget
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
    tier_order,
)

_PAIR_SEPARATOR = " | "

_TIER_LISTS: dict[str, list[list[str]]] = {
    "class": CLASS_TIER_LIST,
    "instance": INSTANCE_TIER_LIST,
    "predicate": PREDICATE_TIER_LIST,
}


class SchemaAwareVerbaliser(VerbaliserBase):
    """Serialise entities using schema-aware predicate-tier prioritisation.

    Output format (identical to V4)::

        predicate_label: value | predicate_label: value

    Triples are always ordered by their predicate tier -- highest-priority
    tier first -- with any untiered predicates appended in graph order.  When
    the model defines a token budget, the ordered triples are greedily capped
    to fit; models without a budget retain every ordered triple.
    """

    def __init__(self, model_key: str = "M2", role: str = "candidate") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to look up the token budget (e.g. "M1", "M2").
        role : str
            ``"source"`` or ``"candidate"``: the encoder role whose input the
            budget counts.
        """
        self.model_key = model_key
        self.role = role

    @cached_property
    def budget(self) -> Optional[TokenBudget]:
        """The model's token budget for this role, or ``None`` when uncapped."""
        return token_budget(self.model_key, self.role)

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        """
        Convert entity triples into schema-ordered key-value pairs.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.

        Returns
        -------
        str
            Non-empty key-value string for entities with retrievable data;
            empty string only for entities with no retrievable information.

        Raises
        ------
        ValueError
            If *entity_type* is not a supported entity type.
        """
        return self.verbalise_after(graph, entity_uri, entity_type, "")

    def verbalise_after(
        self, graph: Graph, entity_uri: URIRef, entity_type: str, preceding: str
    ) -> str:
        """
        Render tier-ordered pairs that fit what ``preceding`` left of the budget.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.
        preceding : str
            Joined text of the earlier components of a composite, or ``""``.

        Returns
        -------
        str
            The kept pairs joined with `` | ``.
        """
        tier_list = self._tier_list_for(entity_type)
        ordered = self._tier_order(self._collect_triples(graph, entity_uri), tier_list)
        pairs = [pair for _subj, pred, obj in ordered if (pair := self._format_pair(graph, pred, obj))]
        return _PAIR_SEPARATOR.join(fit_items(pairs, _PAIR_SEPARATOR, self.budget, preceding))

    @staticmethod
    def _tier_list_for(entity_type: str) -> list[list[str]]:
        """Return the predicate-tier list for *entity_type*."""
        if entity_type not in _TIER_LISTS:
            raise ValueError(
                f"Unsupported entity type: {entity_type!r}. Expected one of "
                "'class', 'instance', 'predicate'."
            )
        return _TIER_LISTS[entity_type]

    @staticmethod
    def _tier_order(
        triples: list[tuple], tier_list: list[list[str]]
    ) -> list[tuple]:
        """Order triples by predicate tier, appending untiered ones last."""
        return tier_order(triples, tier_list)
