"""Schema-aware verbalisation strategy (V6).

Serialises entities as deterministic pipe-separated predicate-value pairs,
always curating triples in schema-aware predicate-tier priority order.

Unlike V4, tier ordering is applied unconditionally -- it is never gated on
triple count -- and the model token budget (when one exists) is applied only
after the tier ordering has been established.  Models without a budget (e.g.
``"M3"``) retain every tier-ordered triple.
"""

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PPAS_BUDGETS,
    PREDICATE_TIER_LIST,
    estimate_tokens,
)

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

    def __init__(self, model_key: str = "M2") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to look up the token budget (e.g. "M1", "M2").
        """
        self.model_key = model_key

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
        tier_list = self._tier_list_for(entity_type)
        triples = self._collect_triples(graph, entity_uri)
        ordered = self._tier_order(triples, tier_list)

        budget = PPAS_BUDGETS.get(self.model_key)
        if budget is not None:
            ordered = self._apply_budget(ordered, budget)

        pairs: list[str] = []
        for _subj, pred, obj in ordered:
            pair = self._format_pair(graph, pred, obj)
            if pair:
                pairs.append(pair)

        return " | ".join(pairs)

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
        ordered: list[tuple] = []
        selected: set[int] = set()
        for tier in tier_list:
            if not tier:
                continue
            tier_predicates = set(tier)
            for index, triple in enumerate(triples):
                if index not in selected and str(triple[1]) in tier_predicates:
                    ordered.append(triple)
                    selected.add(index)
        ordered.extend(
            triple
            for index, triple in enumerate(triples)
            if index not in selected
        )
        return ordered

    def _apply_budget(
        self, triples: list[tuple], budget: int
    ) -> list[tuple]:
        """Greedily keep tier-ordered triples that fit within *budget*."""
        selected: list[tuple] = []
        tokens_used = 0
        for triple in triples:
            cost = estimate_tokens(self._verbalise_triple(*triple))
            if tokens_used + cost <= budget:
                selected.append(triple)
                tokens_used += cost
        return selected
